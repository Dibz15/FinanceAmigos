# install.packages(c("quantmod", "TTR"))
# install.packages(c("ggplot2", "dplyr", "tidyquant"))
# install.packages("pso")
library(pso)
library(ggplot2)
library(dplyr)
# library(tidyquant)
library(quantmod)
library(TTR)
library(tidyr)

# getStockReturns = function(symbols, startDate, endDate) {
#   stocks = lapply(symbols, function(sym) {
#     dailyReturn(na.omit(getSymbols(sym, src="yahoo", from=startDate, to=endDate, 
#                                    auto.assign=FALSE)))
#   })
#   return(do.call(merge.xts, stocks))
# }
# 
# Symbols = c("TSLA", "AAPL", "NVDA", "SPY", "XOM", "MRNA", "VOO", "VONE", "AMD", "UMC")
# 
# StartDate ="2021-01-01"
# EndDate = '2022-01-01'
# 
# dailyReturns = getStockReturns(Symbols, StartDate, EndDate)

symbol <- "AMD"
start_date <- as.Date("2022-01-01")
end_date <- as.Date("2022-04-01")
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
features <- merge(close_price, volume, sma, ema, rsi, macd_signal, bbands_pct)
colnames(features) <- c("Close_Price", "Volume", "SMA", "EMA", "RSI", "MACD_Signal", "BBands_Pct")

# Remove rows with NA values
features_complete <- na.omit(features)

# Convert xts object to data frame
features_df <- data.frame(Date = index(features_complete), coredata(features_complete))
colnames(features_df) <- c("Date", "Close_Price", "Volume", "SMA", "EMA", "RSI", "MACD_Signal", "BBands_Pct")

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

# Split the data into training, validation, and test sets
train_data <- features_complete[1:train_size]
validation_data <- features_complete[(train_size + 1):(train_size + validation_size)]
test_data <- features_complete[(train_size + validation_size + 1):n_rows]



# 
# # Similar fitness function to GA
# # But without weight normalisation
# psofitnessFunction <- function(wt){
#   returnVal = sum(wt * mu)
#   riskVal = sum(wt * sigma)
#   return (returnVal/riskVal)  
# }
# 
# # Negate fitness function as PSO will aim to minimise
# # Several other parameters that can be configured
# # Constrained iterations and swarm size as sometimes generating NAs
# PSOResults <- psoptim(c(0.5,0.5,0.5), fn = function(x){-psofitnessFunction(x)}, lower = c(0.0,0.0,0.0), upper = c(1.0,1.0,1.0), control = list(maxit = 50, s = 10))
# # Or
# # PSOResults <- psoptim(rep(NA,3), fn = function(x){-psofitnessFunction(x)}, lower = c(0.0,0.0,0.0), upper = c(1.0,1.0,1.0), control = list(maxit = 50, s = 10))
# 
# PSOResults$par
# PSOweights <- PSOResults$par
# PSOweights <- PSOweights / sum(PSOweights)
# PSOweights

mse_values = rep(0, 1000)
it_value = 0
window_size = 2 # days
prediction_size = 2

mse_fitness_function <- function(weights, data) {
  start_index <- sample(1:(nrow(data) - window_size), 1)
  # Extract a portion of the data with the given window size
  window_data <- data[start_index:(start_index + window_size - 1),]
  actual_data = data[(start_index + window_size):(start_index + window_size - 1) + prediction_size,]
  print(window_data)
  print(actual_data)
  actual_prices = actual_data[, "Close_Price"]
  # actual_price <- window_data[-1, "Close_Price"]
  # print(actual_price)
  # actual_price <- data[, "Close_Price"]
  # # print(actual_price)
  # predicted_price <- sum(weights * data[, -1]) # -1 to exclude the Close_Price column
  # mse <- mean((actual_price - predicted_price) ^ 2)
  # mse_values[it_value] = mse
  # it_value = it_value + 1
  mse = 0
  return(mse)
}

# Set the lower and upper bounds for the weights
lower_bounds <- rep(-1, 6) # -1 for each feature weight
upper_bounds <- rep(1, 6)  # 1 for each feature weight
initial_weights = runif(6, min=lower_bounds, max=upper_bounds)

mse_fitness_function(initial_weights, train_data)

# Run the PSO optimization
result <- psoptim(
  par = initial_weights,
  fn = mse_fitness_function,
  lower = lower_bounds,
  upper = upper_bounds,
  data = train_data, # Pass the train_data as additional argument to the fitness function
  control = list(maxit = 1000, print_iter=100) # Set the maximum number of iterations, adjust as needed
)

# Extract the optimal weights
optimal_weights <- result$par

# Create a data frame with the iteration numbers and MSE values
mse_data <- data.frame(Iteration = 1:length(mse_values), MSE = mse_values)

# Plot the MSE values
ggplot(mse_data, aes(x = Iteration, y = MSE)) +
  geom_line() +
  labs(title = "MSE during PSO Training", x = "Iteration", y = "MSE")