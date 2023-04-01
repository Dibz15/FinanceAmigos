
library(pso)
library(ggplot2)
library(dplyr)
# library(tidyquant)
library(quantmod)
library(TTR)
library(tidyr)
library(keras)
library(tensorflow)

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
train_size <- floor(0.60 * n_rows)
validation_size <- floor(0.20 * n_rows)
test_size <- n_rows - train_size - validation_size

# Split the data into training, validation, and test sets
train_data <- features_complete[1:train_size]
validation_data <- features_complete[(train_size + 1):(train_size + validation_size)]
test_data <- features_complete[(train_size + validation_size + 1):n_rows]

window_size = 25 # days
prediction_size = 1
num_features = ncol(train_data)

tensorboard_callback <- callback_tensorboard(log_dir = "logs/fit")

normalize_data <- function(data) {
  min_values <- apply(data, 2, min)
  max_values <- apply(data, 2, max)
  normalized_data <- (data - min_values) / (max_values - min_values)
  return(normalized_data)
}

preprocess_data <- function(data, lookback_window, horizon) {
  # Normalize the data
  normalized_data <- normalize_data(data)
  
  num_samples <- nrow(data) - lookback_window - horizon + 1
  x <- array(0, dim = c(num_samples, lookback_window, ncol(normalized_data)))
  y <- array(0, dim = c(num_samples, horizon))
  
  for (i in 1:num_samples) {
    x[i, , ] <- normalized_data[i:(i + lookback_window - 1), ]
    y[i, ] <- normalized_data[(i + lookback_window):(i + lookback_window + horizon - 1), "Close_Price"]
  }
  
  list(x = x, y = y)
}

# preprocess_data <- function(data, lookback_window, horizon) {
#   num_samples <- nrow(data) - lookback_window - horizon + 1
#   x <- array(0, dim = c(num_samples, lookback_window, ncol(data)))
#   y <- array(0, dim = c(num_samples, horizon))
#   
#   for (i in 1:num_samples) {
#     x[i, , ] <- data[i:(i + lookback_window - 1), ]
#     y[i, ] <- data[(i + lookback_window):(i + lookback_window + horizon - 1), "Close_Price"]
#   }
#   
#   list(x = x, y = y)
# }

build_lstm_model <- function(input_shape, learning_rate, regularization) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 10, input_shape = input_shape, return_sequences=TRUE, kernel_regularizer = regularizer_l2(l = regularization)) %>%
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
upper_bounds <- c(0.005, 0.01, 50, 50)
initial_weights <- runif(4, upper_bounds=upper_bounds, lower_bounds=lower_bounds)

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
  lookback_window <- optimal_params[4]
  epochs <- optimal_params[3]
  
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
  lookback_window <- optimal_params[4]
  epochs <- optimal_params[3]
  
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


trading_signals <- ifelse(results$predicted > results$actual[-length(results$actual)], "Buy", "Sell")

portfolio_value <- 100000 # Initial portfolio value
cash <- portfolio_value
num_shares <- 0

for (i in 1:(length(trading_signals) - 1)) {
  if (trading_signals[i] == "Buy" && cash >= actual_test_prices[i]) {
    num_shares_bought <- floor(cash / actual_test_prices[i])
    num_shares <- num_shares + num_shares_bought
    cash <- cash - (num_shares_bought * actual_test_prices[i])
  } else if (trading_signals[i] == "Sell" && num_shares > 0) {
    cash <- cash + (num_shares * actual_test_prices[i])
    num_shares <- 0
  }
}

# Calculate final portfolio value
portfolio_value_final <- cash + (num_shares * actual_test_prices[length(actual_test_prices)])
cat("Initial Portfolio Value:", portfolio_value, "\n")
cat("Final Portfolio Value:", portfolio_value_final, "\n")