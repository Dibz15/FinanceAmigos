# install.packages("quantmod")
# install.packages("dplyr")
# install.packages("hydroPSO")

library(quantmod)
library(hydroPSO)

get_profit <- function(trading_signal,index) {
  daily_returns <- dailyReturn(index)
  strategy_returns <- daily_returns * trading_signal
  end_equity <- cumprod(1 + strategy_returns)
  end_profit <- last(end_equity)
  return(end_profit)
}

get_signal <- function(params,index) {
  fast = as.integer(params[1])
  slow = as.integer(params[2])
  signal = as.integer(params[3])
  
  macd <- MACD(index, nFast = fast, nSlow = slow, nSig = signal, maType = "EMA")
  print(macd)
  
  trading_signal <- lag(ifelse(macd$macd - macd$signal > 0.1, 1, ifelse(macd$macd - macd$signal < -0.1, -1, 0)))
  trading_signal[is.na(trading_signal)] <- 0

  return(trading_signal)
}

fitness_function <- function(params,index) {
  trading_signal <- get_signal(params,index)
  return(-get_profit(trading_signal,index))
}

get_optimal_signal <- function(lower_bounds,upper_bounds,index) {
  this_fitness <- function(params) {return(fitness_function(params,index))}
  opt_results <- hydroPSO(fn = this_fitness, lower = lower_bounds, upper = upper_bounds, control = list(c1 = 2, c2 = 2, maxit = 100))
  optimal_params <- as.integer(opt_results$par)
  return(get_signal(optimal_params,index))
}

get_pool_of_signals <- function(index) {
  set.seed(42)
  signal_small <- get_optimal_signal(c(1,5,5),c(5,20,50),index)
  signal_med <- get_optimal_signal(c(1,15,5),c(15,50,50),index)
  signal_big <- get_optimal_signal(c(1,50,5),c(50,200,50),index)
  signal_common <- get_signal(c(12,26,9),index)

  return(c(signal_small, signal_med, signal_big, signal_common))
}

get_majority_signal <- function(pool) {
  sum <- rowsum(pool)
  new_signal <- lag(ifelse(sum >= 2, 1, ifelse(sum <= -2, -1, 0)))
  return(new_signal)
}

# Training data: SPY index from 2018 to 2021
getSymbols("SPY", src = "yahoo", from = "2018-01-01", to = "2021-12-31")
training_index <- Cl(SPY)

pool_signals <- get_pool_of_signals(training_index)

# eval here

majority_signal <- get_majority_signal(pool_signals)

# eval here

all_signals <- c(pool_signals,majority_signal)

print(pool_signals)

# daily_returns_opt <- dailyReturn(sp500)
# strategy_returns_opt <- daily_returns_opt * trading_signal_opt
# end_equity_opt <- cumprod(1 + strategy_returns_opt)

# starting_amount <- 10000
# portfolio_value <- starting_amount * end_equity_opt

# # Fetching additional data for backtesting
# getSymbols("^GSPC", src = "yahoo", from = "2021-01-01", to = "2021-12-31")
# sp500_test <- Cl(GSPC)

# # Apply MACD with optimal parameters to test data
# macd_test <- MACD(sp500_test, nFast = fast_opt, nSlow = slow_opt, nSig = signal_opt, maType = "EMA")

# # Generate trading signals for test data
# trading_signal_test <- lag(ifelse(macd_test$macd > macd_test$signal, 1, -1))
# trading_signal_test[is.na(trading_signal_test)] <- 0

# # Calculate daily returns and strategy returns for test data
# daily_returns_test <- dailyReturn(sp500_test)
# strategy_returns_test <- daily_returns_test * trading_signal_test
# end_equity_test <- cumprod(1 + strategy_returns_test)

# # Calculate the portfolio value for test data using the starting amount
# portfolio_value_test <- starting_amount * end_equity_test

# # Calculate the performance metrics
# total_return <- last(portfolio_value_test) - starting_amount
# annualized_return <- (total_return / starting_amount) / length(sp500_test) * 252