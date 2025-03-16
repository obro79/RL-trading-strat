import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import datetime
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataHandler')

class DataHandler:
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler()
        self.is_synthetic = False
        self.ticker_info = {}
        
    def download_data(self, ticker, start_date, end_date, interval='1d', max_retries=3):
        """
        Download stock data from Yahoo Finance.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d', '1h', etc.)
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            pd.DataFrame: DataFrame with stock data or synthetic data if download failed
        """
        self.ticker_info = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'interval': interval
        }
        
        # For shorter intervals, verify the date range
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_range = (end_dt - start_dt).days
            
            # yfinance only provides intraday data for a limited time period
            if interval == '1m' and date_range > 7:
                logger.warning(f"yfinance can only provide 1m data for the last 7 days. Limiting request.")
                start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                
            elif interval in ['2m', '5m', '15m', '30m', '60m', '90m', '1h'] and date_range > 60:
                logger.warning(f"yfinance can only provide {interval} data for the last 60 days. Limiting request.")
                start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        
        # Using exponential backoff for retries
        backoff_factor = 1.5
        wait_time = 1  # starting wait time in seconds
        
        for attempt in range(max_retries):
            try:
                # Try to download data
                logger.info(f"Attempt {attempt+1}/{max_retries}: Downloading data for {ticker} from {start_date} to {end_date} with interval {interval}")
                
                # Simplified data downloading using just yf.download
                self.data = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False)
                
                # Check if data was downloaded successfully
                if self.data is None or self.data.empty:
                    logger.warning(f"No data returned for {ticker} on attempt {attempt+1}")
                    if attempt < max_retries - 1:
                        logger.info(f"Waiting {wait_time:.1f} seconds before retrying...")
                        time.sleep(wait_time)
                        wait_time *= backoff_factor  # Increase wait time exponentially
                        continue
                    else:
                        logger.warning(f"Failed to download data for {ticker} after {max_retries} attempts")
                        logger.warning("USING SYNTHETIC DATA for demonstration purposes")
                        self.data = self.generate_synthetic_data(ticker, start_date, end_date)
                        self.is_synthetic = True
                        self.add_technical_indicators()
                        return self.data
                
                logger.info(f"Successfully downloaded {len(self.data)} data points for {ticker}")
                
                # Check if 'Adj Close' exists in the data
                if 'Adj Close' in self.data.columns:
                    # Make a copy of 'Adj Close' to a new column 'Close_Original' and replace 'Close' with 'Adj Close'
                    self.data['Close_Original'] = self.data['Close'].copy()
                    self.data['Close'] = self.data['Adj Close'].copy()
                    logger.info("Using Adjusted Close prices instead of regular Close prices")
                else:
                    logger.warning("Adjusted Close prices not found in the data. Using regular Close prices.")
                
                # Reset index to make Date a column
                self.data = self.data.reset_index()
                self.is_synthetic = False
                
                # Check for NaN values before adding technical indicators
                if self.data.isna().any().any():
                    logger.warning("NaN values found in the downloaded data. Filling with forward and backward fill.")
                    # Forward fill first, then backward fill any remaining NaNs
                    self.data = self.data.fillna(method='ffill').fillna(method='bfill')
                
                # Add technical indicators
                self.add_technical_indicators()
                
                # Final check for NaN values after adding technical indicators
                if self.data.isna().any().any():
                    logger.warning("NaN values still present after adding technical indicators. Filling remaining NaNs.")
                    # First identify any problematic columns with a high percentage of NaNs
                    nan_percent = self.data.isna().mean() * 100
                    problematic_cols = nan_percent[nan_percent > 20].index.tolist()
                    
                    if problematic_cols:
                        logger.warning(f"Columns with >20% NaN values: {problematic_cols}")
                        # For columns with too many NaNs, we might want to drop them
                        # But here we'll try to fill them first
                    
                    # Final attempt to fill any remaining NaNs with column means
                    for col in self.data.select_dtypes(include=[np.number]).columns:
                        if self.data[col].isna().any():
                            col_mean = self.data[col].mean()
                            self.data[col] = self.data[col].fillna(col_mean)
                    
                    # If there are still NaNs, replace with zeros as a last resort
                    self.data = self.data.fillna(0)
                    
                    logger.info("All NaN values have been handled.")
                
                return self.data
                
            except Exception as e:
                logger.error(f"Error downloading {ticker}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {wait_time:.1f} seconds before retrying...")
                    time.sleep(wait_time)
                    wait_time *= backoff_factor  # Increase wait time exponentially
                    continue
                else:
                    logger.warning(f"Failed to download data for {ticker} after {max_retries} attempts")
                    logger.warning("USING SYNTHETIC DATA for demonstration purposes")
                    self.data = self.generate_synthetic_data(ticker, start_date, end_date)
                    self.is_synthetic = True
                    self.add_technical_indicators()
                    return self.data
                    
    def generate_synthetic_data(self, ticker, start_date, end_date):
        """
        Generate synthetic stock data for demonstration purposes when API fails.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date: Start date for data generation
            end_date: End date for data generation
            
        Returns:
            pd.DataFrame: DataFrame with synthetic stock data
        """
        logger.info(f"Generating synthetic data for {ticker}")
        
        # Convert string dates to datetime if necessary
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to only include business days (weekdays)
        date_range = date_range[date_range.weekday < 5]
        
        # Get realistic initial price based on ticker
        initial_price = self._get_realistic_initial_price(ticker)
            
        # Number of data points
        n = len(date_range)
        
        # Generate realistic price movement with some randomness
        # Using geometric brownian motion (simplified)
        random.seed(42)  # For reproducible results but different from other random generators
        np.random.seed(42)  # For reproducible results
        
        # Use more realistic volatility based on the ticker
        volatility = self._get_volatility_for_ticker(ticker)
        mean_return = 0.0005  # Slight positive drift
        
        returns = np.random.normal(mean_return, volatility, n)  # mean daily return and volatility
        price_series = initial_price * (1 + returns).cumprod()
        
        # Create more realistic OHLC data
        high = price_series * (1 + np.random.uniform(0, 0.02, n))
        low = price_series * (1 - np.random.uniform(0, 0.02, n))
        open_prices = price_series * (1 + np.random.normal(0, 0.01, n))
        close_prices = price_series
        
        # Generate volume based on price levels and occasional spikes
        base_volume = initial_price * 10000  # Higher price stocks often have lower volume
        volume_factor = np.random.exponential(1, n)  # Create some spikes
        volume = (base_volume * volume_factor).astype(int)
        
        # Create DataFrame
        synthetic_data = pd.DataFrame({
            'Date': date_range,
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': close_prices,
            'Adj Close': close_prices,  # For synthetic data, Adj Close = Close
            'Volume': volume
        })
        
        # Add some price gaps between days (weekend effect)
        for i in range(1, len(synthetic_data)):
            if (synthetic_data['Date'].iloc[i] - synthetic_data['Date'].iloc[i-1]).days > 1:
                # Add a small gap effect after weekends (up or down with equal probability)
                gap_effect = np.random.choice([-1, 1]) * np.random.uniform(0.005, 0.015)
                synthetic_data.iloc[i:, 1:6] *= (1 + gap_effect)  # Apply to OHLC and Adj Close
        
        logger.info(f"Generated {len(synthetic_data)} days of synthetic data for {ticker}")
        logger.warning("NOTE: Using synthetic data since real data could not be downloaded")
        return synthetic_data
    
    def _get_realistic_initial_price(self, ticker):
        """Get realistic price estimate for the ticker."""
        ticker_prices = {
            'AAPL': 150.0,   # Apple
            'MSFT': 300.0,   # Microsoft
            'GOOGL': 2800.0, # Google
            'AMZN': 3500.0,  # Amazon
            'META': 350.0,   # Meta (Facebook)
            'TSLA': 800.0,   # Tesla
            'NVDA': 400.0,   # NVIDIA
            'JPM': 150.0,    # JPMorgan
            'V': 200.0,      # Visa
            'JNJ': 170.0,    # Johnson & Johnson
            'WMT': 140.0,    # Walmart
            'PG': 140.0,     # Procter & Gamble
            'BAC': 40.0,     # Bank of America
            'DIS': 150.0,    # Disney
            'NFLX': 500.0,   # Netflix
        }
        
        # If ticker is in our dictionary, use that price
        if ticker in ticker_prices:
            return ticker_prices[ticker]
        
        # Otherwise, generate a random but plausible price
        # S&P 500 stocks typically range from $10 to $4000
        price_tiers = [
            (0.7, (10, 50)),    # 70% chance of low-priced stock ($10-$50)
            (0.2, (51, 200)),   # 20% chance of mid-priced stock ($51-$200)
            (0.08, (201, 500)), # 8% chance of high-priced stock ($201-$500)
            (0.02, (501, 4000)) # 2% chance of very high-priced stock ($501-$4000)
        ]
        
        # Choose a tier based on probability
        rand = random.random()
        cumulative_prob = 0
        for prob, (min_price, max_price) in price_tiers:
            cumulative_prob += prob
            if rand <= cumulative_prob:
                return round(random.uniform(min_price, max_price), 2)
        
        # Fallback
        return 100.0
    
    def _get_volatility_for_ticker(self, ticker):
        """Get a realistic volatility estimate for the ticker."""
        # Different sectors have different volatilities
        high_volatility = ['TSLA', 'NVDA', 'COIN', 'MSTR', 'GME', 'AMC', 'RIVN', 'NFLX']
        medium_volatility = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'INTC', 'QCOM']
        low_volatility = ['JNJ', 'PG', 'KO', 'PEP', 'WMT', 'VZ', 'T', 'PFE', 'MRK', 'UNH']
        
        if ticker in high_volatility:
            return random.uniform(0.025, 0.04)
        elif ticker in medium_volatility:
            return random.uniform(0.015, 0.025)
        elif ticker in low_volatility:
            return random.uniform(0.008, 0.015)
        else:
            # Default volatility for unknown tickers
            return random.uniform(0.012, 0.03)
    
    def add_technical_indicators(self):
        """Add technical indicators to the data."""
        try:
            # Make sure we have data to work with
            if self.data is None or len(self.data) == 0:
                logger.warning("No data available to calculate technical indicators")
                return
            
            # Ensure we have enough data points for calculation
            if len(self.data) < 26:  # Minimum required for MACD (26-day EMA)
                logger.warning("Warning: Not enough data points for some indicators")
                # Generate more synthetic data if needed
                if hasattr(self, 'is_synthetic') and self.is_synthetic:
                    logger.info("Extending synthetic data to ensure indicator calculations")
                    # Add more synthetic data points at the beginning
                    n_extra = max(0, 30 - len(self.data))
                    if n_extra > 0:
                        current_first_date = self.data['Date'].iloc[0]
                        extra_dates = pd.date_range(end=current_first_date - pd.Timedelta(days=1), periods=n_extra, freq='B')
                        
                        # Create extra data with similar patterns
                        first_price = self.data['Close'].iloc[0]
                        np.random.seed(41)  # Different seed for variety
                        returns = np.random.normal(-0.0005, 0.015, n_extra)  # slight downtrend for historical data
                        extra_prices = first_price / (1 + returns[::-1]).cumprod()
                        
                        extra_data = pd.DataFrame({
                            'Date': extra_dates,
                            'Open': extra_prices * (1 + np.random.normal(0, 0.01, n_extra)),
                            'High': extra_prices * (1 + np.random.uniform(0, 0.02, n_extra)),
                            'Low': extra_prices * (1 - np.random.uniform(0, 0.02, n_extra)),
                            'Close': extra_prices,
                            'Adj Close': extra_prices,  # Adding Adj Close for consistency
                            'Volume': np.random.randint(1000000, 10000000, n_extra)
                        })
                        
                        # Concatenate and sort
                        self.data = pd.concat([extra_data, self.data]).reset_index(drop=True)
                        self.data = self.data.sort_values('Date').reset_index(drop=True)
                        logger.info(f"Extended data to {len(self.data)} points for technical indicators")
            
            logger.info(f"Calculating technical indicators for {len(self.data)} data points")
            
            # Ensure all numeric data is float to avoid integer division issues
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.data[col] = self.data[col].astype(float)
            
            # Add Moving Averages
            self.data['MA5'] = self.data['Close'].rolling(window=5).mean()
            self.data['MA10'] = self.data['Close'].rolling(window=10).mean()
            self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
            
            # Add Relative Strength Index (RSI)
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            
            # Avoid division by zero
            epsilon = np.finfo(float).eps
            rs = gain / (loss + epsilon)  # Add epsilon to prevent division by zero
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # Add MACD
            exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
            self.data['MACD'] = exp1 - exp2
            self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Add Bollinger Bands
            self.data['20d_std'] = self.data['Close'].rolling(window=20).std()
            self.data['Upper_Band'] = self.data['MA20'] + (self.data['20d_std'] * 2)
            self.data['Lower_Band'] = self.data['MA20'] - (self.data['20d_std'] * 2)
            
            # Add daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            
            # Add Average True Range (ATR) for volatility
            high_low = self.data['High'] - self.data['Low']
            high_close = abs(self.data['High'] - self.data['Close'].shift())
            low_close = abs(self.data['Low'] - self.data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            self.data['ATR'] = true_range.rolling(14).mean()
            
            # Check for infinities and NaNs that might have been introduced
            # and replace them with more appropriate values
            inf_mask = np.isinf(self.data.select_dtypes(include=[np.number]))
            if inf_mask.any().any():
                logger.warning("Infinity values detected in technical indicators. Replacing with NaN.")
                self.data = self.data.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with forward fill then backward fill
            # This ensures we have complete data even at the beginning
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            
            # For any remaining NaNs, fill with column means
            for col in self.data.select_dtypes(include=[np.number]).columns:
                if self.data[col].isna().any():
                    col_mean = self.data[col].mean()
                    if np.isnan(col_mean):  # If mean is also NaN, use 0
                        col_mean = 0
                    self.data[col] = self.data[col].fillna(col_mean)
            
            logger.info("Successfully calculated technical indicators")
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            # Try to fill any remaining NaNs to ensure data usability
            if self.data is not None:
                self.data = self.data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
    def split_train_test(self, train_ratio=0.8):
        """
        Split data into training and testing sets.
        
        Args:
            train_ratio (float): Proportion of data to use for training (0-1)
            
        Returns:
            tuple: (train_data, test_data)
        """
        if self.data is None or len(self.data) == 0:
            logger.warning("No data available to split into train/test sets")
            return pd.DataFrame(), pd.DataFrame()
            
        train_size = int(len(self.data) * train_ratio)
        
        if train_size <= 0 or train_size >= len(self.data):
            logger.warning(f"Invalid train_ratio {train_ratio} resulting in train_size={train_size}")
            if len(self.data) > 1:
                train_size = max(1, min(len(self.data) - 1, int(len(self.data) * 0.8)))
                logger.info(f"Using default train_size={train_size}")
            else:
                logger.error("Not enough data to split")
                return pd.DataFrame(), pd.DataFrame()
        
        self.train_data = self.data[:train_size].copy()
        self.test_data = self.data[train_size:].copy()
        
        logger.info(f"Data split: {len(self.train_data)} training samples, {len(self.test_data)} testing samples")
        return self.train_data, self.test_data
    
    def get_scaled_data(self, data=None):
        """
        Scale the data using MinMaxScaler.
        
        Args:
            data (pd.DataFrame, optional): Data to scale. If None, uses self.data.
            
        Returns:
            pd.DataFrame: Scaled data
        """
        if data is None:
            data = self.data
            
        if data is None or data.empty:
            logger.warning("No data available to scale")
            return pd.DataFrame()
            
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Check if there's data to scale
        if numeric_data.empty:
            logger.warning("No numeric columns found in data for scaling")
            return pd.DataFrame()
            
        try:
            # Fit and transform data
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # Convert back to DataFrame
            scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns)
            
            # Add back non-numeric columns
            for col in data.columns:
                if col not in numeric_data.columns:
                    scaled_df[col] = data[col]
                    
            logger.info(f"Successfully scaled {len(scaled_df)} data points")
            return scaled_df
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            return data  # Return original data if scaling fails
            
    def get_data_info(self):
        """
        Get information about the current data.
        
        Returns:
            dict: Information about the data
        """
        if self.data is None:
            return {"status": "No data loaded"}
            
        info = {
            "ticker": self.ticker_info.get('ticker', 'Unknown'),
            "data_points": len(self.data),
            "start_date": self.data['Date'].min() if 'Date' in self.data.columns else None,
            "end_date": self.data['Date'].max() if 'Date' in self.data.columns else None,
            "is_synthetic": self.is_synthetic,
            "columns": list(self.data.columns),
            "has_missing_values": self.data.isna().any().any(),
            "technical_indicators": [col for col in self.data.columns 
                                    if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        }
        
        if self.train_data is not None and self.test_data is not None:
            info["train_size"] = len(self.train_data)
            info["test_size"] = len(self.test_data)
            
        return info 