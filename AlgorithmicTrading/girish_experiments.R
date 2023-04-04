# install.packages("quantmod")
# install.packages("dplyr")
# install.packages("hydroPSO")

library(quantmod)
library(hydroPSO)

backtest_signal <- function(trading_signal, prices, initial_capital = 10000) {
  cash_value <- rep(NA, length(trading_signal)+1)
  asset_value <- rep(NA, length(trading_signal)+1)
  number_of_shares <- rep(NA, length(trading_signal)+1)

  trading_signal <- as.vector(trading_signal)
  prices <- as.vector(prices)

  cash_value[1] <- initial_capital
  asset_value[1] <- 0
  number_of_shares[1] <- 0

  for (i in 1:(length(trading_signal))) {
    if (trading_signal[i] == 1 && cash_value[i] >= prices[i]) {
      # buy
      number_of_shares[i+1] <- number_of_shares[i] + 1
      cash_value[i+1] <- cash_value[i] - prices[i]
    } else if (trading_signal[i] == -1 && number_of_shares[i] > 0) {
      # sell
      number_of_shares[i+1] <- number_of_shares[i] - 1
      cash_value[i+1] <- cash_value[i] + prices[i]
    } else {
      # hold
      number_of_shares[i+1] <- number_of_shares[i]
      cash_value[i+1] <- cash_value[i]
    }
    asset_value[i+1] <- number_of_shares[i+1] * prices[i]
  }
  return(data.frame(
    cash_value = cash_value, asset_value = asset_value,
    number_of_shares = number_of_shares, total_value = cash_value + asset_value
  ))
}

get_profit <- function(trading_signal,index) {
  initial_capital <- 10000
  backtest <- backtest_signal(trading_signal,index,initial_capital)
  profit <- backtest$total_value[length(backtest$total_value)] - backtest$total_value[1]
  return(profit)
}

get_signal <- function(params,index) {
  fast = as.integer(params[1])
  slow = as.integer(params[2])
  signal = as.integer(params[3])
  
  macd <- MACD(index, nFast = fast, nSlow = slow, nSig = signal, maType = "EMA")
  
  trading_signal <- ifelse(macd$macd - macd$signal > 0.1, 1, ifelse(macd$macd - macd$signal < -0.1, -1, 0))
  trading_signal[is.na(trading_signal)] <- 0

  return(trading_signal)
}

fitness_function <- function(params,index) {
  trading_signal <- get_signal(params,index)
  return(-get_profit(trading_signal,index))
}

get_optimal_parameter <- function(lower_bounds,upper_bounds,index) {
  this_fitness <- function(params) {return(fitness_function(params,index))}
  opt_results <- hydroPSO(fn = this_fitness, lower = lower_bounds, upper = upper_bounds, control = list(c1 = 2, c2 = 2, maxit = 100))
  optimal_params <- as.integer(opt_results$par)
  return(optimal_params)
}

get_majority_signal <- function(pool) {
  # sum the pool signals across the 4 columns
  signal_sum <- rowSums(pool)
  new_signal <- ifelse(signal_sum >= 2, 1, ifelse(signal_sum <= -2, -1, 0))
  return(new_signal)
}

# Training data: SPY index from 2018 to 2021
getSymbols("SPY", src = "yahoo", from = "2018-01-01", to = "2021-12-31")
training_index <- Cl(SPY)

set.seed(42)
parameter_small <- get_optimal_parameter(c(1,5,5),c(5,20,50),training_index)
parameter_med <- get_optimal_parameter(c(1,15,5),c(15,50,50),training_index)
parameter_big <- get_optimal_parameter(c(1,50,5),c(50,200,50),training_index)
parameter_common <- c(12,26,9)

signal_small <- get_signal(parameter_small, training_index)
signal_med <- get_signal(parameter_med, training_index)
signal_big <- get_signal(parameter_big, training_index)
signal_common <- get_signal(parameter_common, training_index)

pool_signals <- data.frame(signal_small, signal_med, signal_big, signal_common)
colnames(pool_signals) <- c("signal_small", "signal_med", "signal_big", "signal_common")
# take the cumulative sum of the signals and plot it, use date (index) as x-axis
cumsum_signals <- cumsum(pool_signals)

library(ggplot2)
ggplot(cumsum_signals, aes(x = index(cumsum_signals))) +
  geom_line(aes(y = signal_small, color = "signal_small")) +
  geom_line(aes(y = signal_med, color = "signal_med")) +
  geom_line(aes(y = signal_big, color = "signal_big")) +
  geom_line(aes(y = signal_common, color = "signal_common")) +
  labs(title = "Cumulative sum of signals", x = "Date", y = "Cumulative sum of signals") +
  theme_minimal()
     
get_investment_values <- function(parameter,index) {
  signal <- get_signal(parameter, index)
  backtest <- backtest_signal(signal,index,10000)
  return(backtest$total_value)
}
pool_values <- data.frame(
  get_investment_values(parameter_small, training_index), get_investment_values(parameter_med, training_index),
  get_investment_values(parameter_big, training_index), get_investment_values(parameter_common, training_index)
)
colnames(pool_values) <- c("value_small", "value_med", "value_big", "value_common")

ggplot(pool_values, aes(x = index(pool_values))) +
  geom_line(aes(y = value_small, color = "value_small")) +
  geom_line(aes(y = value_med, color = "value_med")) +
  geom_line(aes(y = value_big, color = "value_big")) +
  geom_line(aes(y = value_common, color = "value_common")) +
  labs(title = "Investment value", x = "Date", y = "Investment value") +
  theme_minimal()

# eval here

majority_signal <- get_majority_signal(pool_signals)
print(majority_signal)

# eval here

all_signals <- data.frame(pool_signals, majority_signal)
print(all_signals)

# Grammars

library(gramEvol)

# Expects small, med, big, common
get_grammar_expr <- function(signals) {
  # Define our grammar
  rules <- list(expr = grule(op(expr, expr),var),
              op = grule('+', '-', '*'),
              var = grule(signals$small, signals$med, signals$big, signals$common))

  # Create grammar from rules
  grammar <- CreateGrammar(rules)

  grammar_fitness <- function(expr) {
    signal <- eval(expr)
    return(-get_profit(signal,training_index))
  }

  ge <- GrammaticalEvolution(grammar, grammar_fitness, iterations = 500, max.depth = 5)
  expr <- ge$best$expressions
  return(expr)
}

get_grammar_signal <- function(signals,expr) {
  return(eval(expr))
}

all_signals <- data.frame(small=signal_small, med=signal_med, big=signal_big, common=signal_common)
names(all_signals) <- c('small','med','big','common')
grammar_expr <- get_grammar_expr(all_signals)
grammar_signal <- get_grammar_signal(all_signals,grammar_expr)

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