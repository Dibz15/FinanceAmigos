# install.packages("quantmod")
# install.packages("dplyr")
# install.packages("hydroPSO")

library(quantmod)
library(hydroPSO)

getSymbols("^GSPC", src = "yahoo", from = "2018-01-01", to = "2021-12-31")
sp500 <- Cl(GSPC)

fitness_function <- function(params) {
  fast = as.integer(params[1])
  slow = as.integer(params[2])
  signal = as.integer(params[3])
 
  # if(slow - fast <= 5) {
  #   return(-1e10)
  # }
  
  macd <- MACD(sp500, nFast = fast, nSlow = slow, nSig = signal, maType = "EMA")
  print(macd)
  
  trading_signal <- lag(ifelse(macd$macd > macd$signal, 1, -1))
  trading_signal[is.na(trading_signal)] <- 0
  
  daily_returns <- dailyReturn(sp500)
  strategy_returns <- daily_returns * trading_signal
  end_equity <- cumprod(1 + strategy_returns)
  end_profit <- last(end_equity)
  
  return(-end_profit)
}

lower_bounds <- c(1, 50, 5)
upper_bounds <- c(50, 200, 50)

set.seed(123)
opt_results <- hydroPSO(fn = fitness_function, lower = lower_bounds, upper = upper_bounds, control = list(c1 = 2, c2 = 2, maxit = 100))

optimal_params <- as.integer(opt_results$par)

fast_opt <- optimal_params[1]
slow_opt <- optimal_params[2]
signal_opt <- optimal_params[3]

macd_opt <- MACD(sp500, nFast = fast_opt, nSlow = slow_opt, nSig = signal_opt, maType = "EMA")

trading_signal_opt <- lag(ifelse(macd_opt$macd > macd_opt$signal, 1, -1))
trading_signal_opt[is.na(trading_signal_opt)] <- 0

daily_returns_opt <- dailyReturn(sp500)
strategy_returns_opt <- daily_returns_opt * trading_signal_opt
end_equity_opt <- cumprod(1 + strategy_returns_opt)

starting_amount <- 10000
portfolio_value <- starting_amount * end_equity_opt

# Fetching additional data for backtesting
getSymbols("^GSPC", src = "yahoo", from = "2021-01-01", to = "2021-12-31")
sp500_test <- Cl(GSPC)

# Apply MACD with optimal parameters to test data
macd_test <- MACD(sp500_test, nFast = fast_opt, nSlow = slow_opt, nSig = signal_opt, maType = "EMA")

# Generate trading signals for test data
trading_signal_test <- lag(ifelse(macd_test$macd > macd_test$signal, 1, -1))
trading_signal_test[is.na(trading_signal_test)] <- 0

# Calculate daily returns and strategy returns for test data
daily_returns_test <- dailyReturn(sp500_test)
strategy_returns_test <- daily_returns_test * trading_signal_test
end_equity_test <- cumprod(1 + strategy_returns_test)

# Calculate the portfolio value for test data using the starting amount
portfolio_value_test <- starting_amount * end_equity_test

# Calculate the performance metrics
total_return <- last(portfolio_value_test) - starting_amount
annualized_return <- (total_return / starting_amount) / length(sp500_test) * 252