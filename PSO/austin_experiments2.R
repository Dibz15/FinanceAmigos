
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

symbol <- "AMD"
start_date <- as.Date("2018-01-01")
end_date <- as.Date("2023-01-01")
stock_data <- getSymbols(symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)

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

# Convert xts object to data frame
features_df <- data.frame(Date = index(features_complete), coredata(features_complete))
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


# Calculate the number of rows for each set
n_rows <- nrow(features_complete)
train_size <- floor(0.70 * n_rows)
validation_size <- floor(0.15 * n_rows)
test_size <- n_rows - train_size - validation_size

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

normal_data = normalize_data(features_complete)

train_data <- normal_data$data[1:train_size]
validation_data <- normal_data$data[(train_size + 1):(train_size + validation_size)]
test_data <- normal_data$data[(train_size + validation_size + 1):n_rows]

window_size = 25 # days
prediction_size = 1
num_features = ncol(train_data)

tensorboard_callback <- callback_tensorboard(log_dir = "logs/fit")

preprocess_data <- function(data, lookback_window, horizon) {
  # Normalize the data
  normalized_data <- data
  print(nrow(data))
  num_samples <- nrow(data) - lookback_window - horizon + 1
  print(num_samples)
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
  lookback_window <- 20
  epochs <- 50
  
  processed_train <- preprocess_data(train_data, lookback_window, horizon)
  processed_val <- preprocess_data(val_data, lookback_window, horizon)
  
  # Build the LSTM model with the optimal parameters
  model <- build_lstm_model(input_shape = dim(processed_train$x)[-1], 
                            learning_rate = learning_rate,
                            regularization = regularization)
  
  # Train the LSTM model with the optimal parameters
  history <- model %>% fit(
    processed_train$x, processed_train$y,
    epochs = epochs,
    batch_size = 32,
    validation_data = list(processed_val$x, processed_val$y),
    callbacks = list(callback_tensorboard(log_dir = "logs/test"))
  )
  return(model)
}

train_model = train_final_model(optimal_params, train_data, validation_data, horizon=1)

evaluate_final_model = function(optimal_params, test_data, horizon) {
  learning_rate <- optimal_params[1]
  regularization <- optimal_params[2]
  lookback_window <- 20

  processed_test <- preprocess_data(test_data, lookback_window, horizon)
  
  print(processed_test)
  # Evaluate the performance of the LSTM model on the test data
  test_loss <- train_model %>% evaluate(processed_test$x, processed_test$y)
  
  # Generate predictions for the test data
  predicted_test_prices <- train_model %>% predict(processed_test$x)
  
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

results = evaluate_final_model(optimal_params, test_data, horizon=1)
# Implement your trading strategy and backtest it using the predicted_test_prices

shifted_predicted <- results$predicted[-1,] # Remove the first element
actual_trimmed <- results$actual[-length(results$actual),] # Remove the last element

denormalize_data <- function(normalized_data, col_min, col_range) {
  denormalized_data <- normalized_data * col_range + col_min
  denormalized_data
}

shifted_predicted = denormalize_data(shifted_predicted, normal_data$col_mins["Close_Price"], normal_data$col_ranges["Close_Price"])
actual_trimmed = denormalize_data(actual_trimmed, normal_data$col_mins["Close_Price"], normal_data$col_ranges["Close_Price"])

portfolio_value <- 10000 # Initial portfolio value
cash <- portfolio_value
num_shares <- 0

# Initialize the cash_value vector
# cash_value <- numeric(length(trading_signals))

# Calculate the predicted profit percentage
predicted_profit_pct <- (shifted_predicted - actual_trimmed) / actual_trimmed

# Modify the trading signal to include the "hold" option
threshold <- 0.05  # You can adjust this threshold to control when to hold
trading_signals <- ifelse(predicted_profit_pct > threshold, "Buy",
                          ifelse(predicted_profit_pct < -threshold, "Sell", "Hold"))

# Initialize the cash_value vector
cash_value <- rep(NA, length(trading_signals))
asset_value = rep(NA, length(trading_signals))
total_portfolio_value = rep(NA, length(trading_signals))
buy_and_hold_value = rep(NA, length(trading_signals))

buy_and_hold_shares = floor(portfolio_value / actual_trimmed[1])

# Backtesting loop
for (i in 1:(length(trading_signals) - 1)) {
  cash_value[i] <- cash
  buy_and_hold_value[i] = buy_and_hold_shares * actual_trimmed[i]
  
  if (trading_signals[i] == "Buy" && cash >= actual_trimmed[i]) {
    investment_amount <- cash * predicted_profit_pct[i]
    num_shares_bought <- floor(investment_amount / actual_trimmed[i])
    num_shares <- num_shares + num_shares_bought
    cash <- cash - (num_shares_bought * actual_trimmed[i])
  } else if (trading_signals[i] == "Sell" && num_shares > 0) {
    sell_amount <- num_shares * actual_trimmed[i] * (-predicted_profit_pct[i])
    num_shares_sold <- floor(sell_amount / actual_trimmed[i])
    num_shares <- num_shares - num_shares_sold
    cash <- cash + (num_shares_sold * actual_trimmed[i])
  }
  
  asset_value[i] = num_shares * actual_trimmed[i]
  total_portfolio_value[i] = cash + asset_value[i]
}

# Add the final cash value
cash_value[length(trading_signals)] <- cash
asset_value[length(trading_signals)] = num_shares * actual_trimmed[length(actual_trimmed)]
total_portfolio_value[length(trading_signals)] = cash + (num_shares * actual_trimmed[length(actual_trimmed)])
buy_and_hold_value[length(trading_signals)] = buy_and_hold_shares * actual_trimmed[length(actual_trimmed)]

# The rest of the code for calculating the final portfolio value and plotting remains the same
# Add the final cash value
cash_value[length(trading_signals)] <- cash
lookback_window = 20
horizon = 1

# Calculate final portfolio value
portfolio_value_final <- cash + (num_shares * actual_trimmed[length(actual_trimmed)])

# Combine the actual prices, predicted prices, and trading signals into one data frame
plot_data <- data.frame(Date = dates,
                        Value = c(actual_trimmed, shifted_predicted, cash_value, asset_value, total_portfolio_value, buy_and_hold_value),
                        Type = factor(rep(c("Actual", "Predicted", "Cash Value", "Asset Value", "Portfolio Value", "Buy and Hold Value"), each = length(dates))),
                        Signal = factor(rep(trading_signals, 6)))

# Create a ggplot for the actual and predicted prices
price_plot <- ggplot(data = subset(plot_data, Type %in% c("Actual", "Predicted")),
                     aes(x = Date, y = Value, color = Type, group = Type)) +
  geom_line() +
  geom_point(data = subset(plot_data, Type == "Actual"), aes(color = Signal), shape = 24, size = 3) +
  scale_color_manual(values = c("Actual" = "blue",
                                "Predicted" = "orange",
                                "Cash Value" = "black",
                                "Portfolio Value" = "gold",
                                "Asset Value" = "cyan",
                                "Buy" = "green",
                                "Sell" = "red",
                                "Hold" = "purple")) +
  labs(y = "Price", color = "Series")

# Create a ggplot for the cash value, asset value, and portfolio value
value_plot <- ggplot(data = subset(plot_data, Type %in% c("Cash Value", "Asset Value", "Portfolio Value", "Buy and Hold Value")),
                     aes(x = Date, y = Value, color = Type, group = Type)) +
  geom_line() +
  scale_color_manual(values = c("Cash Value" = "black",
                                "Portfolio Value" = "gold",
                                "Asset Value" = "cyan",
                                "Buy and Hold Value" = "red")) +
  labs(y = "Value", color = "Series")

# Combine the two ggplots with a common x-axis
combined_plot <- grid.arrange(price_plot, value_plot, ncol = 1, heights = c(1, 1))

# Print the combined ggplot
plot(combined_plot)

cat("Initial Portfolio Value:", portfolio_value, "\n")
cat("Final Portfolio Value:", portfolio_value_final, "\n")
cat("Buy and Hold Final Value:", buy_and_hold_value[length(buy_and_hold_value)], "\n")