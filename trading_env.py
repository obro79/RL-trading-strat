import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from enum import Enum
import logging

# Setup logging
logger = logging.getLogger('TradingEnv')

class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gymnasium interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001, max_holding_period=20):
        super(TradingEnv, self).__init__()
        
        # Data for the environment
        self.df = df.copy()
        
        # Check for NaN values in the input dataframe
        if self.df.isna().any().any():
            logger.warning("NaN values detected in input dataframe. Filling with forward fill and backward fill.")
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
            
            # If there are still NaNs, replace with zeros as a last resort
            if self.df.isna().any().any():
                logger.warning("Still have NaN values after ffill/bfill. Replacing with zeros.")
                self.df = self.df.fillna(0)
        
        # Ensure Date column is the index or removed from numerical calculations
        if 'Date' in self.df.columns:
            # Store dates separately if needed for reference
            self.dates = self.df['Date'].values
            # Remove Date from the dataframe used for calculations
            self.df_numeric = self.df.drop(columns=['Date'])
        else:
            self.df_numeric = self.df.copy()
            
        self.reward_range = (-np.inf, np.inf)
        
        # Trading params
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.max_holding_period = max_holding_period  # Maximum number of steps to hold before penalty
        
        # Actions: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(len(Actions))
        
        # Observation space: includes price data (OHLCV) and account information
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(16,), dtype=np.float32  # Added one more for holding period
        )
        
        # Episode variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_cost = 0
        self.total_sales = 0
        self.net_worth_history = [self.initial_balance]
        self.trades = []
        self.holding_period = 0  # How long shares have been held
        self.buy_price = 0       # Price at which shares were bought
        
        return self._next_observation(), {}  # Return observation and empty info dict
        
    def _next_observation(self):
        # Get data points for the last 5 days and scale to between 0-1
        frame = np.zeros(16, dtype=np.float32)  # Ensure dtype matches observation_space
        
        # If we have enough data points, include the last 5 days of prices
        if self.current_step >= 5:
            try:
                # Include OHLCV data for last 5 days - use only numeric columns
                np_data = self.df_numeric.iloc[self.current_step-5:self.current_step].to_numpy()
                
                # Check for NaN or infinity values in np_data
                if np.isnan(np_data).any() or np.isinf(np_data).any():
                    logger.warning(f"NaN or infinity values found in observation data at step {self.current_step}. Replacing with safe values.")
                    # Replace NaN and inf values with 0
                    np_data = np.nan_to_num(np_data, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Fix the normalization - normalize each column separately then flatten
                # This avoids broadcasting errors between different shape arrays
                if np_data.size > 0:  # Make sure we have data
                    # Get the max value for each column (avoiding division by zero)
                    max_values = np.maximum(np.max(np_data, axis=0), np.finfo(float).eps)
                    
                    # Normalize each column by its max value
                    normalized_data = np_data / max_values[np.newaxis, :]
                    
                    # Check for NaN values after normalization
                    if np.isnan(normalized_data).any():
                        logger.warning(f"NaN values introduced during normalization at step {self.current_step}. Using safe normalization.")
                        # Try a safer normalization approach
                        for col in range(normalized_data.shape[1]):
                            col_max = np.max(np.abs(np_data[:, col]))
                            if col_max > 0:  # Avoid division by zero
                                normalized_data[:, col] = np_data[:, col] / col_max
                            else:
                                normalized_data[:, col] = 0.0
                    
                    # Double-check and fix any remaining NaNs or infs
                    normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Now flatten the normalized data
                    flat_data = normalized_data.flatten()
                    
                    # Make sure we don't exceed the frame size and all values are finite
                    n_elements = min(len(flat_data), 10)
                    frame[:n_elements] = flat_data[:n_elements]
            except Exception as e:
                logger.error(f"Error creating observation at step {self.current_step}: {str(e)}")
                # Ensure we return a valid observation even if there's an error
                frame[:10] = 0.0  # Set price data to zeros
        
        try:
            # Include account info
            frame[10] = float(self.balance / max(self.initial_balance, 1e-8))  # Avoid division by zero
            frame[11] = float(self.shares_held)
            frame[12] = float(self.total_shares_bought)
            frame[13] = float(self.total_shares_sold)
            frame[14] = float(self.current_step / max(len(self.df), 1))  # Avoid division by zero
            frame[15] = float(min(self.holding_period / max(self.max_holding_period, 1), 1.0))  # Normalized holding period
        except Exception as e:
            logger.error(f"Error adding account info to observation: {str(e)}")
            # Keep the default zeros in the frame
        
        # Final sanity check - ensure no NaN or infinity values
        frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
        
        return frame
            
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False  # Add truncated flag for gymnasium compatibility
        
        # If holding shares, increment holding period
        if self.shares_held > 0:
            self.holding_period += 1
        else:
            self.holding_period = 0
            
        obs = self._next_observation()
        reward = self._calculate_reward(action)
        info = self._get_info()
        
        # Update net worth history
        self.net_worth_history.append(self.balance + self.shares_held * self.current_price)
        
        return obs, reward, done, truncated, info  # Add truncated flag
    
    def _take_action(self, action):
        # Get current price and set it as a variable
        try:
            self.current_price = float(self.df.iloc[self.current_step]['Close'])
            # Ensure price is valid
            if np.isnan(self.current_price) or np.isinf(self.current_price) or self.current_price <= 0:
                logger.warning(f"Invalid price at step {self.current_step}: {self.current_price}. Using previous or default price.")
                # Use previous price if available, otherwise use a small positive default
                if hasattr(self, 'current_price') and self.current_step > 0:
                    self.current_price = self.current_price  # Keep previous price
                else:
                    self.current_price = 1.0  # Default price
        except Exception as e:
            logger.error(f"Error getting price at step {self.current_step}: {str(e)}")
            self.current_price = 1.0  # Default price
            
        # Take an action
        action_type = Actions(action)
        
        if action_type == Actions.BUY and self.balance > 0:
            # Calculate maximum shares we can buy
            max_shares = self.balance / (self.current_price * (1 + self.transaction_fee_percent))
            shares_bought = int(max_shares)
            
            # Calculate cost with transaction fee
            cost = shares_bought * self.current_price * (1 + self.transaction_fee_percent)
            
            # Update variables
            self.balance -= cost
            self.shares_held += shares_bought
            self.total_shares_bought += shares_bought
            self.total_cost += cost
            
            # Record trade and buy price
            if shares_bought > 0:
                self.buy_price = self.current_price  # Record purchase price
                self.holding_period = 0  # Reset holding period
                self.trades.append({
                    'step': self.current_step,
                    'price': self.current_price,
                    'type': 'buy',
                    'shares': shares_bought,
                    'cost': cost
                })
            
        elif action_type == Actions.SELL and self.shares_held > 0:
            # Sell all shares
            shares_sold = self.shares_held
            
            # Calculate sales with transaction fee
            sales = shares_sold * self.current_price * (1 - self.transaction_fee_percent)
            
            # Calculate profit/loss
            profit = sales - (shares_sold * self.buy_price)
            
            # Update variables
            self.balance += sales
            self.shares_held = 0
            self.total_shares_sold += shares_sold
            self.total_sales += sales
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'price': self.current_price,
                'type': 'sell',
                'shares': shares_sold,
                'sales': sales,
                'profit': profit
            })
    
    def _calculate_reward(self, action=None):
        # Calculate reward based on net worth change
        try:
            current_net_worth = self.balance + self.shares_held * self.current_price
            previous_net_worth = self.net_worth_history[-1] if self.net_worth_history else self.initial_balance
            
            # Ensure previous_net_worth is not zero to avoid division by zero
            if previous_net_worth <= 0:
                previous_net_worth = max(previous_net_worth, 1e-8)
            
            # Base reward is the percentage change in net worth
            reward = ((current_net_worth - previous_net_worth) / previous_net_worth) * 100
            
            # Additional reward factors
            
            # Add penalty for holding too long
            if self.shares_held > 0 and self.holding_period > self.max_holding_period:
                # Increasing penalty the longer we hold beyond max period
                holding_penalty = -0.1 * (self.holding_period - self.max_holding_period)
                reward += holding_penalty
                
            # Add reward for profitable sells
            if action == Actions.SELL.value and self.current_price > self.buy_price and self.buy_price > 0:
                # Proportional to profit percentage
                profit_percent = ((self.current_price - self.buy_price) / max(self.buy_price, 1e-8)) * 100
                sell_reward = profit_percent * 0.5  # Scale the reward
                reward += sell_reward
                
            # Add small penalty for doing nothing (no shares and no trades)
            if self.shares_held == 0 and len(self.trades) == 0 and self.current_step > 20:
                reward -= 0.1  # Small penalty to encourage taking some action
                
            # Ensure reward is finite
            if np.isnan(reward) or np.isinf(reward):
                logger.warning(f"Invalid reward at step {self.current_step}: {reward}. Using zero reward.")
                reward = 0.0
                
            return float(reward)
            
        except Exception as e:
            logger.error(f"Error calculating reward: {str(e)}")
            return 0.0  # Default reward
    
    def _get_info(self):
        # Return additional information
        return {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'net_worth': self.balance + self.shares_held * self.current_price,
            'holding_period': self.holding_period
        }
        
    def render(self, mode='human'):
        # Render the environment to the screen
        profit = self.net_worth_history[-1] - self.initial_balance
        return {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'net_worth': self.net_worth_history[-1],
            'profit': profit,
            'trades': self.trades,
            'holding_period': self.holding_period
        } 