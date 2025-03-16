"""
Reinforcement Learning Trading Strategy

This script serves as the entry point for the RL Trading Strategy application.
It allows users to run the Streamlit app directly or import components for custom use.

Usage:
    To run the Streamlit app:
    $ streamlit run main.py

    To use components individually:
    >>> from data_handler import DataHandler
    >>> from trading_env import TradingEnv
    >>> from agent_handler import AgentHandler
"""

import os
import sys
import subprocess

def run_streamlit_app():
    """Run the Streamlit app."""
    print("Starting Reinforcement Learning Trading Strategy App...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    # Check if the required files exist
    required_files = ["app.py", "data_handler.py", "trading_env.py", "agent_handler.py"]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    
    if missing_files:
        print(f"Error: Missing required files: {', '.join(missing_files)}")
        print("Please make sure all required files are in the current directory.")
        sys.exit(1)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Run the Streamlit app
    run_streamlit_app()
