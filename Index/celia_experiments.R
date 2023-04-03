
library(pso)
library(ggplot2)
library(dplyr)
# library(tidyquant)
library(quantmod)
library(TTR)
library(tidyr)
library(keras)
library(tensorflow)
library(gridExtra)

# tensorflow::install_tensorflow(version = "2.11.0", gpu = TRUE)
# tf_config()

prepare_stock_data = function(stock_data) {
  # Close price
  close_price <- Cl(stock_data)
  
  # Simple Moving Average (SMA) - 10-day period
  sma <- SMA(close_price, n = 10)
  
  # Exponential Moving Average (EMA) - 10-day period
  ema <- EMA(close_price, n = 10)
  
  # Relative Strength Index (RSI) - 14-day period
  rsi <- RSI(close_price, n = 14)
  
  # Moving Average Convergence Divergence (MACD) - fast = 12 days, slow = 26 days, signal = 9 days
  macd_obj <- MACD(close_price, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
  macd_signal <- macd_obj$signal
  
  # Bollinger Bands - 20-day period
  bbands <- BBands(close_price, n = 20)
  bbands_pct <- (close_price - bbands[,1]) / (bbands[,3] - bbands[,1])
  
  # Add volume for the stock
  volume = Vo(stock_data)
  
  # Merge all the features into a single dataset
  features <- merge(close_price, sma, ema, rsi, macd_signal, bbands_pct)
  colnames(features) <- c("Close_Price", "SMA", "EMA", "RSI", "MACD_Signal", "BBands_Pct")
  
  # Remove rows with NA values
  features_complete <- na.omit(features)
  return(features_complete)
}

plot_stock = function(prepped_stock_data) {
  # Convert xts object to data frame
  features_df <- data.frame(Date = index(prepped_stock_data), coredata(prepped_stock_data))
  colnames(features_df) <- c("Date", "Close_Price", "SMA", "EMA", "RSI", "MACD_Signal", "BBands_Pct")
  
  # Reshape data frame to long format
  features_long <- features_df %>%
    tidyr::gather(key = "Feature", value = "Value", -Date)
  
  # Update the scale for the "Volume" feature
  features_long$Value <- ifelse(features_long$Feature == "Volume",
                                features_long$Value / 1e6, # Adjust the scale factor as needed
                                features_long$Value)
  
  # Add a suffix to the "Volume" feature name to indicate the new scale
  features_long$Feature <- ifelse(features_long$Feature == "Volume",
                                  "Volume (Millions)",
                                  features_long$Feature)
  
  ggplot(features_long, aes(x = Date, y = Value, color = Feature, group = Feature)) +
    geom_line() +
    scale_color_discrete(name = "Features") +
    labs(title = "Stock Features over Time",
         x = "Date",
         y = "Value") +
    theme_minimal()
  
}

train_test_split = function(stock_data) {
  # Calculate the number of rows for each set
  n_rows <- nrow(stock_data$data)
  train_size <- floor(0.70 * n_rows)
  validation_size <- floor(0.15 * n_rows)
  test_size <- n_rows - train_size - validation_size
  
  train_data <- stock_data$data[1:train_size]
  validation_data <- stock_data$data[(train_size + 1):(train_size + validation_size)]
  test_data <- stock_data$data[(train_size + validation_size + 1):n_rows]
  
  list(
    train = train_data,
    val = validation_data,
    test = test_data
  )
}

normalize_data <- function(data) {
  col_mins <- apply(data, 2, min)
  col_maxs <- apply(data, 2, max)
  col_ranges <- col_maxs - col_mins
  
  normalized_data <- sweep(data, 2, col_mins, FUN = "-")
  normalized_data <- sweep(normalized_data, 2, col_ranges, FUN = "/")
  
  list(data = normalized_data, col_mins = col_mins, col_ranges = col_ranges)
}

denormalize_data <- function(normalized_data, col_mins, col_ranges) {
  denormalized_data <- sweep(normalized_data, 2, col_ranges, FUN = "*")
  denormalized_data <- sweep(denormalized_data, 2, col_mins, FUN = "+")
  
  denormalized_data
}
# Split the data into training, validation, and test sets

symbol <- "AMD"
start_date <- as.Date("2018-01-01")
end_date <- as.Date("2023-01-01")
stock_data <- getSymbols(symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)

prepped_stock_data = prepare_stock_data(stock_data)

plot_stock(prepped_stock_data)
normal_data = normalize_data(prepped_stock_data)
split_data = train_test_split(normal_data)

train_data = split_data$train
validation_data = split_data$val
test_data = split_data$test

preprocess_data <- function(data, lookback_window, horizon) {
  # Normalize the data
  normalized_data <- data
  num_samples <- nrow(data) - lookback_window - horizon + 1
  x <- array(0, dim = c(num_samples, lookback_window, ncol(normalized_data)))
  y <- array(0, dim = c(num_samples, horizon))
  
  for (i in 1:num_samples) {
    x[i, , ] <- normalized_data[i:(i + lookback_window - 1), ]
    y[i, ] <- normalized_data[(i + lookback_window):(i + lookback_window + horizon - 1), "Close_Price"]
  }
  
  list(x = x, y = y)
}

build_lstm_model <- function(input_shape, learning_rate, regularization) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 20, input_shape = input_shape, return_sequences=TRUE, kernel_regularizer = regularizer_l2(l = regularization)) %>%
    layer_dropout(rate=0.5) %>%
    layer_lstm(units = 10, kernel_regularizer = regularizer_l2(l = regularization)) %>%
    layer_dropout(rate=0.5) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(learning_rate = learning_rate)
  )
  
  return(model)
}

mse_fitness_function <- function(pso_params, train_data, val_data, horizon) {
  learning_rate <- pso_params[1]
  regularization <- pso_params[2]
  epochs = pso_params[3]
  lookback_window = pso_params[4]

  processed_train <- preprocess_data(train_data, lookback_window, horizon)
  processed_val <- preprocess_data(val_data, lookback_window, horizon)
  
  model <- build_lstm_model(
    input_shape = dim(processed_train$x)[-1],
    learning_rate = learning_rate,
    regularization = regularization
  )
  
  current_time <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
  log_dir <- paste0("logs/fit/", current_time, "_lr_", learning_rate, "_reg_", regularization, "_epochs_", epochs, "_lookback_", lookback_window)
  
  # Initialize tensorboard callback with the unique log directory
  tensorboard_callback <- callback_tensorboard(log_dir = log_dir)
  
  history <- model %>% fit(
    x = processed_train$x,
    y = processed_train$y,
    validation_data = list(processed_val$x, processed_val$y),
    epochs = epochs,
    batch_size = 32,
    verbose = 0,
    callbacks=list(tensorboard_callback)
  )
  
  val_mse <- tail(history$metrics$val_loss, 1)
  return(val_mse)
}

lower_bounds <- c(0.0001, 0.001, 10, 5)
upper_bounds <- c(0.005, 0.01, 100, 50)
initial_weights <- runif(4, min=lower_bounds, max=upper_bounds)

result <- psoptim( 
  par = initial_weights,
  fn = mse_fitness_function,
  lower = lower_bounds,
  upper = upper_bounds,
  train_data = train_data,
  val_data = validation_data,
  # lookback_window = 20,
  horizon = 1,
  control = list(maxit = 20)
)

optimal_params <- result$par

train_final_model = function(optimal_params, train_data, val_data, horizon) {
  # Extract the optimal parameters from the PSO result
  learning_rate <- optimal_params[1]
  regularization <- optimal_params[2]
  epochs = optimal_params[3]
  lookback_window = optimal_params[4]
  
  processed_train <- preprocess_data(train_data, lookback_window, horizon)
  processed_val <- preprocess_data(val_data, lookback_window, horizon)
  
  # Build the LSTM model with the optimal parameters
  model <- build_lstm_model(input_shape = dim(processed_train$x)[-1], 
                            learning_rate = learning_rate,
                            regularization = regularization)
  
  current_time <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
  log_dir <- paste0("logs/fit/", current_time, "_lr_", learning_rate, "_reg_", regularization, "_epochs_", epochs, "_lookback_", lookback_window)
  
  # Initialize tensorboard callback with the unique log directory
  tensorboard_callback <- callback_tensorboard(log_dir = log_dir)
  
  # Train the LSTM model with the optimal parameters
  history <- model %>% fit(
    processed_train$x, processed_train$y,
    epochs = epochs,
    batch_size = 32,
    validation_data = list(processed_val$x, processed_val$y),
    callbacks = list(tensorboard_callback)
  )
  return(model)
}

train_model = train_final_model(optimal_params, train_data, validation_data, horizon=1)

evaluate_final_model = function(optimal_params, model, test_data, horizon) {
  learning_rate <- optimal_params[1]
  regularization <- optimal_params[2]
  lookback_window = optimal_params[4]

  processed_test <- preprocess_data(test_data, lookback_window, horizon)
  
  print(processed_test)
  # Evaluate the performance of the LSTM model on the test data
  test_loss <- model %>% evaluate(processed_test$x, processed_test$y)
  
  # Generate predictions for the test data
  predicted_test_prices <- model %>% predict(processed_test$x)
  
  # Convert the test data and predictions to data frames
  actual_test_prices <- as.data.frame(processed_test$y)
  colnames(actual_test_prices) <- c("Actual_Price")
  predicted_test_prices <- as.data.frame(predicted_test_prices)
  colnames(predicted_test_prices) <- c("Predicted_Price")
  
  # Combine the actual and predicted test prices
  price_comparison <- cbind(actual_test_prices, predicted_test_prices)
  
  # Calculate the error between the actual and predicted test prices
  error <- mean((price_comparison$Actual_Price - price_comparison$Predicted_Price)^2)
  
  # Print the error
  cat("MSE on test data:", error)
  
  list(actual = actual_test_prices, predicted = predicted_test_prices)
}

backtest_results = function(dates, predicted, actual, normalized_data, buy_threshold = 0.05, sell_threshold = -0.05, initial_cash = 10000) {
  denormalize_data <- function(normalized_data, col_min, col_range) {
    denormalized_data <- normalized_data * col_range + col_min
    denormalized_data
  }
  
  shift_and_denormalize = function(predicted, actual) {
    # Implement your trading strategy and backtest it using the predicted_test_prices
    shifted_predicted <- predicted[-1,] # Remove the first element
    actual_trimmed <- actual[-length(actual),] # Remove the last element
    
    # This is so that instead of it being today predicted : today actual for signal, it is
    # tomorrow predicted : today actual
    
    shifted_predicted = denormalize_data(shifted_predicted, normalized_data$col_mins["Close_Price"], normalized_data$col_ranges["Close_Price"])
    actual_trimmed = denormalize_data(actual_trimmed, normalized_data$col_mins["Close_Price"], normalized_data$col_ranges["Close_Price"])
    list(predicted = shifted_predicted, actual = actual_trimmed)
  }

  shifted_and_trimmed = shift_and_denormalize(predicted, actual)
  shifted_predicted = shifted_and_trimmed$predicted
  actual_trimmed = shifted_and_trimmed$actual
  
  # Calculate the predicted profit percentage
  predicted_profit_pct <- (shifted_predicted - actual_trimmed) / actual_trimmed
  trading_signals <- ifelse(predicted_profit_pct > buy_threshold, "Buy",
                            ifelse(predicted_profit_pct < sell_threshold, "Sell", "Hold"))
  
  backtest_series = backtest_loop(predicted_profit_pct, initial_cash, trading_signals, actual_trimmed)
  # The rest of the code for calculating the final portfolio value and plotting remains the same
  # Add the final cash value
  
  
  plot_backtest(dates, backtest_series, shifted_predicted)
  
  # Calculate final portfolio value
  cat("Initial Portfolio Value:", initial_cash, "\n")
  cat("Final Portfolio Value:", backtest_series$Portfolio[nrow(backtest_series)], "\n")
  cat("Buy and Hold Final Value:", backtest_series$BuyAndHold[nrow(backtest_series)], "\n")
}

full_evaluation = function(optimal_params, model, test_data) {
  lookback_window = optimal_params[4]
  horizon = 1
  results = evaluate_final_model(optimal_params, model, test_data, horizon)
  # Combine the actual prices, predicted prices, and trading signals into one data frame
  dates <- as.Date(index(test_data)[(lookback_window + horizon + 1):(nrow(test_data))])
  
  backtest_results(dates, results$predicted, results$actual, normal_data, 
                   buy_threshold = 0.02, 
                   sell_threshold = -0.02,
                   initial_cash = 100000)
}

full_evaluation(optimal_params, train_model, test_data)

normalize_data2 <- function(data_to_normalize, col_mins, col_ranges) {
  normalized_data <- sweep(data_to_normalize, 2, col_mins, FUN = "-")
  normalized_data <- sweep(normalized_data, 2, col_ranges, FUN = "/")
  return(normalized_data)
}























plot_backtest = function(dates, backtest_series, predicted) {
  plot_data <- data.frame(Date = rep(dates, 6),
                          Value = c(backtest_series$Actual, predicted, backtest_series$Cash, backtest_series$Assets, backtest_series$Portfolio, backtest_series$BuyAndHold),
                          Type = factor(rep(c("Actual", "Cash Value", "Asset Value", "Portfolio Value"), each = length(dates))),
                          Signal = factor(rep(backtest_series$Signals, 6)))
  # Create a ggplot for the cash value, asset value, and portfolio value
  value_plot <- ggplot(data = subset(plot_data, Type %in% c("Cash Value", "Asset Value", "Portfolio Value")),
                       aes(x = Date, y = Value, color = Type, group = Type)) +
    geom_line() +
    scale_color_manual(values = c("Cash Value" = "black",
                                  "Portfolio Value" = "blue",
                                  "Asset Value" = "cyan",
                                  "Buy and Hold Value" = "red")) +
    labs(y = "Value ($)", color = "Series")
  
  # Print the combined ggplot
  plot(value_plot)
}

backtest_loop = function(predicted_profit_pct,portfolio_value, signals, actual_value) {
  cash <- portfolio_value
  num_shares <- 0
  # Initialize the cash_value vector
  cash_value <- rep(NA, length(signals))
  asset_value = rep(NA, length(signals))
  total_portfolio_value = rep(NA, length(signals))
  
  # Backtesting loop
  for (i in 1:(length(signals) - 1)) {
    cash_value[i] <- cash

    if (signals[i] == "Buy" && cash >= actual_value[i]) {
      investment_amount <- cash * predicted_profit_pct[i] # TODO choose and explain
      num_shares_bought <- floor(investment_amount / actual_value[i])
      num_shares <- num_shares + num_shares_bought
      cash <- cash - (num_shares_bought * actual_value[i])
    } else if (signals[i] == "Sell" && num_shares > 0) {
      sell_amount <- num_shares * actual_value[i] * (-predicted_profit_pct[i])
      num_shares_sold <- floor(sell_amount / actual_value[i])
      num_shares <- num_shares - num_shares_sold
      cash <- cash + (num_shares_sold * actual_value[i])
    }
    
    asset_value[i] = num_shares * actual_value[i]
    total_portfolio_value[i] = cash + asset_value[i]
  }
  
  # Add the final cash value
  cash_value[length(signals)] <- cash
  asset_value[length(signals)] = num_shares * actual_value[length(actual_value)]
  total_portfolio_value[length(signals)] = cash + (num_shares * actual_value[length(actual_value)])

  return(
    data.frame(Cash = cash_value, 
               Assets = asset_value, 
               Portfolio = total_portfolio_value,
               Signals = signals)
  )
}

maTypes <- function(number) {
  return('EMA')
}

fitness_of_one_index_one_stock <- function(params,close_p) {
  # type_of_index <- params[1]
  # conditions on the type of index; TODO add more
  nFast <- params[1]
  nSlow <- params[2]
  nSig <- params[3]
  maType <- maTypes(params[4])
  macd <- MACD(close_p, nFast = nFast, nSlow = nSlow, nSig = nSig, maType = maType)
  signal <- macd$signal
  
  # removing NA
  signal <- tail(signal,-40)
  close_p <- tail(close_p,-40)
  
  backtest <- backtest_loop(rep(0.1, length(signal)),10000,signal,close_p)
  return((backtest$Portfolio)[length(signal)])
}

fitness_of_one_index <- function(params,close_ps) {
  value <- 0
  for (close_p in close_ps) {
    value <- value + fitness_of_one_index_one_stock(params,close_p)
  }
  return(value)
}

symbol <- "^GSPC"
start_date <- as.Date("2021-08-01")
end_date <- as.Date("2023-04-01")
stock_data <- getSymbols(symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
close_prices <- Cl(stock_data)

lower_bounds <- c(1, 16, 1, 0)
upper_bounds <- c(15, 30, 10, 0)
initial_weights <- runif(4, min=lower_bounds, max=upper_bounds)

result <- psoptim( 
  par = initial_weights,
  fn = function(x) fitness_of_one_index_one_stock(x,close_prices),
  lower = lower_bounds,
  upper = upper_bounds,
  control = list(maxit = 20)
)
optimal_params <- result$par

nFast <- optimal_params[1]
nSlow <- optimal_params[2]
nSig <- optimal_params[3]
maType <- maTypes(optimal_params[4])
macd <- MACD(close_prices, nFast = nFast, nSlow = nSlow, nSig = nSig, maType = maType)
signal <- macd$signal

signal <- tail(signal,-40)
signal <- data.frame(Date = index(signal), coredata(signal))
colnames(signal) <- c("Date","Signal")

close_prices <- tail(close_prices,-40)
close_prices <- data.frame(Date = index(close_prices), coredata(close_prices))
colnames(close_prices) <- c("Date","Close")

trading_signals <- ifelse(signal > 0.1, "Buy",
                          ifelse(signal < -0.1, "Sell", "Hold"))

backtest <- backtest_loop(rep(0.1, length(signal)),10000,trading_signals,close_prices)
print(backtest$Portfolio)
