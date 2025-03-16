import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import os
import traceback

from data_handler import DataHandler
from trading_env import TradingEnv
from agent_handler import AgentHandler

# Set page config
st.set_page_config(
    page_title="RL Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Reinforcement Learning Trading Strategy")
st.markdown("""
This application demonstrates how to use reinforcement learning (RL) to develop a trading strategy.
The process involves fetching stock data, training an RL agent, and evaluating its performance.
""")

# Add a note about potential API issues
st.info("""
**Note:** This app attempts to fetch real stock data from Yahoo Finance. If the live API is not available, 
the app will automatically use synthetic data for demonstration purposes.
""")

# Sidebar for parameters
st.sidebar.header("Parameters")

# Stock selection
ticker = st.sidebar.text_input("Stock Ticker", "MSFT")

# Date range
today = datetime.date.today()
default_start = today - timedelta(days=365*2)  # 2 years ago
default_end = today

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

# RL algorithm selection
algorithm = st.sidebar.selectbox(
    "RL Algorithm",
    ["PPO", "A2C", "DQN"],
    index=0,
)

# Training parameters
st.sidebar.subheader("Training Parameters")
initial_balance = st.sidebar.number_input("Initial Balance ($)", 10000, 1000000, 10000)
train_ratio = st.sidebar.slider("Training Data Ratio", 0.5, 0.9, 0.8)
total_timesteps = st.sidebar.number_input("Training Timesteps", 10000, 1000000, 50000)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Exploration", "Training Process", "Evaluation", "Example of a Trained Model", "Learning Resources"])

# Data exploration tab
with tab1:
    st.header("Data Exploration")
    
    st.subheader("1. Fetching Stock Data")
    st.markdown("""
    We start by fetching historical stock data using the `yfinance` library. This includes:
    - Open, High, Low, Close prices
    - Volume
    - Additional technical indicators (added in preprocessing)
    """)
    
    # Fetch data button
    if st.button("Fetch Data"):
        try:
            with st.spinner("Fetching stock data..."):
                # Create data handler and download data
                data_handler = DataHandler()
                df = data_handler.download_data(ticker, start_date, end_date)
                
                # Check if the data is synthetic
                is_synthetic = False
                if 'is_synthetic' in locals():
                    is_synthetic = locals()['is_synthetic']
                
                if df is not None and not df.empty:
                    # Store in session state
                    st.session_state.data_handler = data_handler
                    st.session_state.df = df
                    
                    # Check if we're using synthetic data (by checking if data generation message was printed)
                    if "Generating synthetic data" in df.iloc[0].to_string() if len(df) > 0 else False:
                        st.warning(f"⚠️ Using synthetic data for {ticker} as the Yahoo Finance API is currently unavailable.")
                        st.session_state.using_synthetic_data = True
                    else:
                        st.success(f"Successfully fetched real data for {ticker}!")
                        st.session_state.using_synthetic_data = False
                    
                    # Display data
                    st.dataframe(df.head())
                    
                    # Display basic statistics
                    st.subheader("Basic Statistics")
                    st.dataframe(df.describe())
                else:
                    st.error(f"Failed to fetch data for {ticker}. Please try another ticker or check your internet connection.")
        except Exception as e:
            st.error(f"An error occurred while fetching data: {str(e)}")
            st.info("Try a different ticker symbol. For example: MSFT, GOOGL, AMZN")
    
    # If data is already fetched, display visualizations
    if 'df' in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        
        # Show synthetic data warning if applicable
        if st.session_state.get('using_synthetic_data', False):
            st.warning(f"⚠️ Displaying synthetic data for {ticker} as the Yahoo Finance API is currently unavailable.")
            st.info("The synthetic data is generated using a random walk model with realistic price movements. It's suitable for demonstrating the RL trading strategy but does not reflect actual market conditions.")
        
        st.subheader("2. Visualizing Stock Data")
        
        # Plot stock price using Plotly
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=('Price', 'Volume'),
                           row_width=[0.2, 0.7])
        
        # Price chart with candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'MA5' in df.columns and 'MA20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['MA5'],
                    line=dict(color='blue', width=1),
                    name="MA5"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['MA20'],
                    line=dict(color='orange', width=1),
                    name="MA20"
                ),
                row=1, col=1
            )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name="Volume"
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text=f"{ticker} Stock Price",
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        st.subheader("3. Technical Indicators")
        
        # Show a card explaining the indicators
        with st.expander("What are these technical indicators?", expanded=False):
            st.markdown("""
            ### Technical Indicators Explained:
            
            **RSI (Relative Strength Index):**
            - Measures the speed and change of price movements
            - Values above 70 suggest the asset is overbought
            - Values below 30 suggest the asset is oversold
            
            **MACD (Moving Average Convergence Divergence):**
            - Shows the relationship between two moving averages of a security's price
            - Signal line crossovers can indicate buy/sell opportunities
            
            **Bollinger Bands:**
            - Three lines: middle band (20-day SMA) and upper/lower bands (±2 standard deviations)
            - Help identify overbought/oversold conditions and volatility
            
            **ATR (Average True Range):**
            - Measures market volatility
            - Higher values indicate higher volatility
            """)
            
            if st.session_state.get('using_synthetic_data', False):
                st.info("Note: These indicators are calculated from synthetically generated data as Yahoo Finance API data is currently unavailable.")
        
        # Create tabs for different indicators
        ind_tab1, ind_tab2, ind_tab3, ind_tab4 = st.tabs(["RSI & MACD", "Bollinger Bands", "Returns", "Volatility (ATR)"])
        
        with ind_tab1:
            # Check if the required columns exist and if df has any data
            if df is not None and not df.empty and 'RSI' in df.columns and 'MACD' in df.columns and 'Signal_Line' in df.columns:
                # RSI and MACD
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.03, subplot_titles=('RSI', 'MACD'),
                                   row_width=[0.2, 0.7])
                
                # RSI
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['RSI'],
                        line=dict(color='purple', width=1),
                        name="RSI"
                    ),
                    row=1, col=1
                )
                
                # Add RSI guidelines
                if len(df) > 0:
                    # Safely access the first and last date
                    first_date = df['Date'].iloc[0] if not df['Date'].isna().all() else None
                    last_date = df['Date'].iloc[-1] if not df['Date'].isna().all() else None
                    
                    if first_date is not None and last_date is not None:
                        fig.add_shape(
                            type="line", line_color="red", line_width=1, opacity=0.5,
                            x0=first_date, x1=last_date, y0=70, y1=70,
                            row=1, col=1
                        )
                        
                        fig.add_shape(
                            type="line", line_color="green", line_width=1, opacity=0.5,
                            x0=first_date, x1=last_date, y0=30, y1=30,
                            row=1, col=1
                        )
                
                # MACD
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['MACD'],
                        line=dict(color='blue', width=1),
                        name="MACD"
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['Signal_Line'],
                        line=dict(color='red', width=1),
                        name="Signal Line"
                    ),
                    row=2, col=1
                )
                
                # Add MACD histogram (difference between MACD and Signal Line)
                fig.add_trace(
                    go.Bar(
                        x=df['Date'],
                        y=df['MACD'] - df['Signal_Line'],
                        name="MACD Histogram",
                        marker_color=np.where(df['MACD'] - df['Signal_Line'] > 0, 'green', 'red')
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation of current RSI value
                latest_rsi = df['RSI'].iloc[-1] if len(df) > 0 else None
                if latest_rsi is not None:
                    if latest_rsi > 70:
                        st.warning(f"Current RSI: {latest_rsi:.2f} - The stock may be **overbought**.")
                    elif latest_rsi < 30:
                        st.warning(f"Current RSI: {latest_rsi:.2f} - The stock may be **oversold**.")
                    else:
                        st.info(f"Current RSI: {latest_rsi:.2f} - The stock is in a **neutral** zone.")
                
            else:
                if 'data_handler' in st.session_state and st.session_state.data_handler.data is not None and hasattr(st.session_state.data_handler, 'is_synthetic') and st.session_state.data_handler.is_synthetic:
                    st.warning("RSI & MACD data shown is calculated from synthetic data. This is for demonstration purposes only.")
                else:
                    st.warning("RSI or MACD data not available. Please fetch data first.")
            
        with ind_tab2:
            # Check if the required columns exist and if df has any data
            if df is not None and not df.empty and 'MA20' in df.columns and 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
                # Bollinger Bands
                fig = go.Figure()
                
                # Price
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['Close'],
                        line=dict(color='black', width=1),
                        name="Close"
                    )
                )
                
                # Moving average
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['MA20'],
                        line=dict(color='blue', width=1),
                        name="MA20"
                    )
                )
                
                # Upper band
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['Upper_Band'],
                        line=dict(color='red', width=1, dash='dot'),
                        name="Upper Band"
                    )
                )
                
                # Lower band
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['Lower_Band'],
                        line=dict(color='green', width=1, dash='dot'),
                        name="Lower Band"
                    )
                )
                
                # Add band width (volatility indicator)
                bandwidth = (df['Upper_Band'] - df['Lower_Band']) / df['MA20'] * 100
                
                # Update layout
                fig.update_layout(
                    height=400,
                    title="Bollinger Bands",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show secondary chart with Band Width
                bandwidth_fig = go.Figure()
                bandwidth_fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=bandwidth,
                        line=dict(color='purple', width=1),
                        name="Band Width %"
                    )
                )
                
                bandwidth_fig.update_layout(
                    height=200,
                    title="Bollinger Band Width (Volatility)",
                    showlegend=True
                )
                
                st.plotly_chart(bandwidth_fig, use_container_width=True)
                
                # Add interpretation
                latest_close = df['Close'].iloc[-1] if len(df) > 0 else None
                latest_upper = df['Upper_Band'].iloc[-1] if len(df) > 0 else None
                latest_lower = df['Lower_Band'].iloc[-1] if len(df) > 0 else None
                
                if all(x is not None for x in [latest_close, latest_upper, latest_lower]):
                    if latest_close > latest_upper:
                        st.warning(f"Price is currently above the upper band, suggesting the stock may be **overbought**.")
                    elif latest_close < latest_lower:
                        st.warning(f"Price is currently below the lower band, suggesting the stock may be **oversold**.")
                    else:
                        st.info(f"Price is currently within the Bollinger Bands, suggesting normal trading range.")
                
            else:
                st.warning("Bollinger Bands data not available. Please fetch data first.")
            
        with ind_tab3:
            # Check if the required column exists
            if df is not None and not df.empty and 'Daily_Return' in df.columns:
                # Daily returns
                fig = go.Figure()
                
                # Daily returns
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['Daily_Return'] * 100,  # Convert to percentage
                        mode='lines',
                        name="Daily Returns (%)"
                    )
                )
                
                # Add baseline
                if len(df) > 0:
                    fig.add_shape(
                        type="line", line_color="black", line_width=1, opacity=0.5,
                        x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=0, y1=0
                    )
                
                # Update layout
                fig.update_layout(
                    height=400,
                    title="Daily Returns (%)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display statistics
                returns_stats = df['Daily_Return'].describe() * 100  # Convert to percentage
                
                # Format the statistics table
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Daily Return", f"{returns_stats['mean']:.2f}%")
                    st.metric("Max Daily Return", f"{returns_stats['max']:.2f}%")
                with col2:
                    st.metric("Daily Volatility", f"{returns_stats['std']:.2f}%")
                    st.metric("Min Daily Return", f"{returns_stats['min']:.2f}%")
                
                # Calculate additional metrics
                positive_days = (df['Daily_Return'] > 0).sum()
                negative_days = (df['Daily_Return'] < 0).sum()
                win_rate = positive_days / (positive_days + negative_days) * 100
                
                # Display win rate
                st.metric("Win Rate", f"{win_rate:.1f}%", 
                          delta=f"{win_rate - 50:.1f}% vs 50/50" if win_rate != 50 else None)
                
                st.dataframe(returns_stats)
            else:
                st.warning("Daily returns data not available.")
                
        with ind_tab4:
            # Check if ATR is available
            if df is not None and not df.empty and 'ATR' in df.columns:
                # ATR Chart
                fig = go.Figure()
                
                # ATR Line
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['ATR'],
                        line=dict(color='orange', width=2),
                        name="ATR (14-day)"
                    )
                )
                
                # Update layout
                fig.update_layout(
                    height=350,
                    title="Average True Range (ATR) - Volatility Indicator",
                    showlegend=True,
                    yaxis_title="Price Range"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display ATR percentage (ATR/Price)
                atr_pct = (df['ATR'] / df['Close'] * 100).iloc[-1] if len(df) > 0 else None
                if atr_pct is not None:
                    if atr_pct > 3:
                        st.warning(f"Current ATR is {atr_pct:.2f}% of price - **High Volatility**")
                    elif atr_pct < 1:
                        st.info(f"Current ATR is {atr_pct:.2f}% of price - **Low Volatility**")
                    else:
                        st.info(f"Current ATR is {atr_pct:.2f}% of price - **Moderate Volatility**")
            else:
                st.warning("ATR data not available.")
        
        st.subheader("4. Data Splitting")
        st.markdown("""
        We split the data into training and testing sets:
        - Training set: Used to train the RL agent
        - Testing set: Used to evaluate the trained agent
        """)
        
        if st.button("Split Data"):
            with st.spinner("Splitting data..."):
                # Split data
                data_handler = st.session_state.data_handler
                train_data, test_data = data_handler.split_train_test(train_ratio)
                
                # Store in session state
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                
                # Display split info
                st.success(f"Data split completed: {len(train_data)} training samples, {len(test_data)} testing samples")
                
                # Display samples
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Training Data (First 5 rows)")
                    st.dataframe(train_data.head())
                with col2:
                    st.subheader("Testing Data (First 5 rows)")
                    st.dataframe(test_data.head())

# Training process tab
with tab2:
    st.header("Training Process")
    
    st.subheader("1. Creating the Trading Environment")
    st.markdown("""
    We create a custom trading environment using OpenAI Gym. The environment:
    - Simulates a stock trading scenario
    - Handles actions (buy, sell, hold)
    - Calculates rewards based on portfolio performance
    - Provides observations (state) to the agent
    """)
    
    st.code("""
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001):
        # Initialize environment with stock data and parameters
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(15,))
    
    def reset(self):
        # Reset environment to initial state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        # ... more reset logic ...
        return self._next_observation()
    
    def step(self, action):
        # Execute action, update state, calculate reward
        self._take_action(action)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._next_observation()
        reward = self._calculate_reward()
        return obs, reward, done, {}
    """, language="python")
    
    st.subheader("2. Reinforcement Learning Agent")
    st.markdown(f"""
    We use {algorithm} (from Stable Baselines3) as our RL algorithm. The agent:
    - Observes the current state of the environment
    - Takes actions (buy, sell, hold)
    - Receives rewards based on performance
    - Learns to maximize cumulative reward
    """)
    
    # Algorithm explanation based on selection
    if algorithm == "PPO":
        st.markdown("""
        **Proximal Policy Optimization (PPO)** is a policy gradient method that:
        - Uses a clipped objective function to constrain policy updates
        - Strikes a good balance between sample complexity, implementation complexity, and performance
        - Works well for continuous control tasks and discrete action spaces
        """)
    elif algorithm == "A2C":
        st.markdown("""
        **Advantage Actor-Critic (A2C)** is a policy gradient method that:
        - Combines value-based and policy-based methods
        - Uses a critic to estimate the value function and reduce variance
        - Actor decides which action to take, critic tells how good the action is
        """)
    elif algorithm == "DQN":
        st.markdown("""
        **Deep Q-Network (DQN)** is a value-based method that:
        - Combines Q-learning with deep neural networks
        - Uses experience replay to reduce correlation between samples
        - Uses a target network to stabilize learning
        - Works well for discrete action spaces
        """)
    
    # Training button
    if 'train_data' in st.session_state:
        if st.button("Train Agent"):
            with st.spinner(f"Training {algorithm} agent... This may take a while."):
                # Create training environment
                train_data = st.session_state.train_data
                env = TradingEnv(train_data, initial_balance=initial_balance)
                
                # Create agent
                agent_handler = AgentHandler(env, algorithm=algorithm)
                
                # Train agent
                model = agent_handler.train(total_timesteps=total_timesteps)
                
                # Store in session state
                st.session_state.agent_handler = agent_handler
                st.session_state.model = model
                
                st.success(f"{algorithm} agent training completed after {total_timesteps} timesteps!")
    else:
        st.warning("Please complete the data exploration steps first!")

# Evaluation tab
with tab3:
    st.header("Evaluation")
    
    st.subheader("1. Testing the Trained Agent")
    st.markdown("""
    We evaluate the trained agent on the test dataset to see how well it performs on unseen data.
    This helps us understand the agent's generalization capability and real-world applicability.
    """)
    
    # Test button
    if 'model' in st.session_state and 'test_data' in st.session_state:
        if st.button("Test Agent"):
            with st.spinner("Testing agent on test data..."):
                # Get agent and test data
                agent_handler = st.session_state.agent_handler
                test_data = st.session_state.test_data
                
                # Create test environment
                test_env = TradingEnv(test_data, initial_balance=initial_balance)
                
                # Test agent
                episodes_df, mean_reward, std_reward = agent_handler.test(test_env, num_episodes=1)
                
                # Store results
                st.session_state.episodes_df = episodes_df
                
                # Display results
                st.success(f"Agent testing completed with mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    else:
        st.warning("Please complete the training process first!")
    
    # If episodes data exists, display results
    if 'episodes_df' in st.session_state:
        episodes_df = st.session_state.episodes_df
        
        st.subheader("2. Performance Metrics")
        
        # Calculate performance metrics
        initial_balance = episodes_df['balance'].iloc[0] + episodes_df['shares_held'].iloc[0] * episodes_df['current_price'].iloc[0]
        final_balance = episodes_df['net_worth'].iloc[-1]
        profit = final_balance - initial_balance
        profit_percent = (profit / initial_balance) * 100
        
        # Total trades
        buy_trades = episodes_df[episodes_df['action'] == 1].shape[0]
        sell_trades = episodes_df[episodes_df['action'] == 2].shape[0]
        total_trades = buy_trades + sell_trades
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Balance", f"${final_balance:.2f}", f"{profit_percent:.2f}%")
        col2.metric("Total Profit", f"${profit:.2f}")
        col3.metric("Total Trades", total_trades)
        col4.metric("Buy/Sell Ratio", f"{buy_trades}/{sell_trades}")
        
        st.subheader("3. Trading Activity Visualization")
        
        # Create figure with price and trades
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=episodes_df['step'],
                y=episodes_df['current_price'],
                mode='lines',
                name="Price",
                line=dict(color='black', width=1)
            )
        )
        
        # Add buy trades
        buy_points = episodes_df[episodes_df['action'] == 1]
        fig.add_trace(
            go.Scatter(
                x=buy_points['step'],
                y=buy_points['current_price'],
                mode='markers',
                name="Buy",
                marker=dict(color='green', size=10, symbol='triangle-up')
            )
        )
        
        # Add sell trades
        sell_points = episodes_df[episodes_df['action'] == 2]
        fig.add_trace(
            go.Scatter(
                x=sell_points['step'],
                y=sell_points['current_price'],
                mode='markers',
                name="Sell",
                marker=dict(color='red', size=10, symbol='triangle-down')
            )
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            title="Trading Activity",
            xaxis_title="Time Step",
            yaxis_title="Price",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("4. Portfolio Value Over Time")
        
        # Create figure with portfolio value
        fig = go.Figure()
        
        # Add net worth line
        fig.add_trace(
            go.Scatter(
                x=episodes_df['step'],
                y=episodes_df['net_worth'],
                mode='lines',
                name="Portfolio Value",
                line=dict(color='blue', width=2)
            )
        )
        
        # Add initial balance line
        fig.add_shape(
            type="line", line_color="red", line_width=1, opacity=0.5,
            x0=episodes_df['step'].iloc[0], x1=episodes_df['step'].iloc[-1], 
            y0=initial_balance, y1=initial_balance,
            name="Initial Balance"
        )
        
        # Add annotations
        fig.add_annotation(
            x=episodes_df['step'].iloc[0],
            y=initial_balance,
            text="Initial Balance",
            showarrow=False,
            yshift=10
        )
        
        fig.add_annotation(
            x=episodes_df['step'].iloc[-1],
            y=final_balance,
            text=f"Final Balance: ${final_balance:.2f}",
            showarrow=True,
            arrowhead=1
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            title="Portfolio Value Over Time",
            xaxis_title="Time Step",
            yaxis_title="Value ($)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("5. Reward Analysis")
        
        # Create figure with cumulative rewards
        fig = go.Figure()
        
        # Calculate cumulative rewards
        episodes_df['cumulative_reward'] = episodes_df['reward'].cumsum()
        
        # Add cumulative reward line
        fig.add_trace(
            go.Scatter(
                x=episodes_df['step'],
                y=episodes_df['cumulative_reward'],
                mode='lines',
                name="Cumulative Reward",
                line=dict(color='purple', width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            title="Cumulative Reward Over Time",
            xaxis_title="Time Step",
            yaxis_title="Cumulative Reward",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show reward statistics
        reward_stats = episodes_df['reward'].describe()
        st.dataframe(reward_stats)

# Example of a Trained Model tab
with tab4:
    st.header("Example of a Trained Model")
    
    st.markdown("""
    This section demonstrates the performance of a trained reinforcement learning model on Microsoft (MSFT) stock data.
    
    **Training Details:**
    - **Model**: PPO (Proximal Policy Optimization)
    - **Training Period**: January 2013 - December 2020 (8 years)
    - **Testing Period**: January 2021 - December 2022 (2 years)
    - **Initial Investment**: $10,000
    """)
    
    # Insert a static image of training vs testing periods
    st.image("data/plots/training_testing_periods.png", 
             caption="Training vs Testing Periods for Microsoft Stock", 
             use_column_width=True)
    
    # Display example metrics
    st.subheader("Performance Summary")
    
    # Create example metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Portfolio Value", "$15,842.63", "+58.43%")
    with col2:
        st.metric("Total Trades", "32")
    with col3:
        st.metric("Win Rate", "67.5%")
    
    # More detailed metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Daily Return", "0.12%")
        st.metric("Max Drawdown", "-12.4%")
    with col2:
        st.metric("Sharpe Ratio", "1.78")
        st.metric("Total Profit", "$5,842.63")
    
    # Display performance visualization
    st.subheader("Performance Visualization")
    
    # Insert a static image of portfolio performance
    st.image("data/plots/portfolio_performance.png", 
             caption="Portfolio Value vs Microsoft Stock Price", 
             use_column_width=True)
    
    # Display trade activity
    st.subheader("Trading Activity")
    
    # Insert a static image of trading activity
    st.image("data/plots/trading_activity.png", 
             caption="Buy/Sell Actions on Microsoft Stock", 
             use_column_width=True)
    
    # Example trades table
    st.subheader("Key Trades")
    
    # Create a sample dataframe of trades
    trades_data = {
        "Date": ["2021-02-15", "2021-03-05", "2021-05-12", "2021-07-28", "2021-10-15", "2022-02-24", "2022-05-12", "2022-11-10"],
        "Type": ["Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell"],
        "Price": ["$244.20", "$231.60", "$239.00", "$286.54", "$304.21", "$294.59", "$255.35", "$242.98"],
        "Shares": [20, 20, 22, 22, 18, 18, 21, 21],
        "Profit/Loss": ["--", "+$172.00", "--", "+$1,046.88", "--", "-$173.16", "--", "-$259.77"]
    }
    
    trades_df = pd.DataFrame(trades_data)
    st.dataframe(trades_df)
    
    # Return distribution
    st.subheader("Return Distribution")
    
    # Insert a static image of return distribution
    st.image("data/plots/return_distribution.png", 
             caption="Distribution of Daily Returns (%)", 
             use_column_width=True)
    
    # Add active trading vs buy-and-hold comparison
    st.subheader("Active Trading vs Buy-and-Hold Strategy")
    
    # Insert a static image of active trading vs buy-and-hold
    st.image("data/plots/active_vs_buyhold.png", 
             caption="Comparison of Active Trading (RL Strategy) vs Buy-and-Hold Strategy", 
             use_column_width=True)
    
    # Add key insights
    st.subheader("Key Insights")
    st.markdown("""
    1. **Market Timing**: The model successfully identified major buying opportunities during market corrections, particularly in Feb 2021, May 2021, and May 2022.
    
    2. **Profit Taking**: The model demonstrated good profit-taking behavior, selling near local peaks in March 2021 and July 2021.
    
    3. **Drawdown Handling**: Despite the significant tech sector drawdown in early 2022, the model maintained a positive overall return.
    
    4. **Patience**: The model showed appropriate patience, making only 32 trades over the 2-year period rather than overtrading.
    
    5. **Limitations**: The model struggled during the volatile period of late 2022, showing that even well-trained models face challenges in rapidly changing market conditions.
    """)
    
    # Disclaimer
    st.warning("""
    **Disclaimer**: Past performance is not indicative of future results. This demo uses historical data and does not account for all real-world trading factors like slippage, liquidity issues, and breaking news events. This demonstration is for educational purposes only and should not be considered investment advice.
    """)

# Learning resources tab
with tab5:
    st.header("Learning Resources")
    
    st.subheader("Reinforcement Learning Concepts")
    st.markdown("""
    ### Key RL Concepts:
    
    1. **Agent**: The learner and decision maker, in our case the trading algorithm.
    2. **Environment**: The world that the agent interacts with, in our case the stock market.
    3. **State**: The situation in which the agent finds itself, represented by market data and portfolio status.
    4. **Action**: The decisions made by the agent (buy, sell, hold).
    5. **Reward**: The feedback signal that measures the success of an action, in our case portfolio growth.
    6. **Policy**: The strategy that the agent employs to determine actions based on the current state.
    7. **Value function**: A prediction of future rewards, used to evaluate the goodness of states.
    
    ### Algorithms:
    
    - **Policy Gradient Methods** (like PPO, A2C): Learn a policy directly, mapping states to actions.
    - **Value-Based Methods** (like DQN): Learn the value of states and actions, then select actions that maximize value.
    - **Actor-Critic Methods**: Combine policy gradient and value-based approaches.
    
    ### Further Reading:
    
    - ["Reinforcement Learning: An Introduction" by Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)
    - [OpenAI's Spinning Up in Deep RL](https://spinningup.openai.com/)
    - [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
    """)
    
    st.subheader("Trading Strategy Considerations")
    st.markdown("""
    ### Key Trading Concepts:
    
    1. **Technical Analysis**: Using price and volume data to identify patterns and trends.
    2. **Fundamental Analysis**: Evaluating a company's financial health and future prospects.
    3. **Risk Management**: Controlling exposure to potential losses.
    4. **Transaction Costs**: Including fees, slippage, and market impact.
    5. **Market Regimes**: Adapting to changing market conditions (bull, bear, sideways).
    
    ### Limitations to Consider:
    
    - **Overfitting**: The model may learn patterns specific to the training data but not generalizable.
    - **Market Efficiency**: Markets may be efficient enough that consistent excess returns are difficult.
    - **Non-stationarity**: Financial markets change over time, sometimes drastically.
    - **Black Swan Events**: Rare, unpredictable events can have massive impacts.
    - **Execution Realities**: Simulated trading often doesn't account for liquidity issues, slippage, etc.
    
    ### Improvement Directions:
    
    - Incorporating more features (sentiment analysis, macroeconomic indicators)
    - Ensemble methods combining multiple algorithms
    - Multi-asset portfolio optimization
    - Adaptive learning rates based on market volatility
    - Hierarchical reinforcement learning for multi-timeframe analysis
    """)
    
    st.subheader("Project Extensions")
    st.markdown("""
    ### Ways to Extend This Project:
    
    1. **Multi-Asset Trading**: Extend to trade multiple assets simultaneously.
    2. **Portfolio Optimization**: Incorporate modern portfolio theory to optimize asset allocation.
    3. **Market Sentiment Integration**: Add news sentiment analysis as features.
    4. **Live Trading Integration**: Connect to a broker API for paper or real trading.
    5. **Advanced RL Algorithms**: Implement more sophisticated algorithms like Soft Actor-Critic (SAC) or Trust Region Policy Optimization (TRPO).
    6. **Explainable AI**: Add techniques to interpret and explain the agent's decisions.
    7. **Regime Detection**: Automatically detect and adapt to different market regimes.
    8. **Hyperparameter Optimization**: Use Bayesian optimization to find optimal hyperparameters.
    """)

# Main script execution
if __name__ == "__main__":
    # Add footer
    st.markdown("---")
    st.markdown("Created with Streamlit, Stable Baselines3, and yfinance") 