import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from trading_env import TradingEnv
import gymnasium as gym
import logging
import torch

# Setup logging
logger = logging.getLogger('AgentHandler')

# Custom callback to check for NaN values during training
class NanCheckCallback(BaseCallback):
    """
    Callback for checking and handling NaN values during training.
    """
    def __init__(self, verbose=0):
        super(NanCheckCallback, self).__init__(verbose)
        self.nan_detected = False
        self.inf_detected = False
        
    def _on_step(self):
        # Check for NaNs in observations
        if torch.is_tensor(self.locals['obs_tensor']):
            if torch.isnan(self.locals['obs_tensor']).any():
                self.nan_detected = True
                logger.warning(f"NaN detected in observation tensor at step {self.num_timesteps}")
                
                # Replace NaNs with zeros
                self.locals['obs_tensor'] = torch.nan_to_num(self.locals['obs_tensor'], nan=0.0)
                
            if torch.isinf(self.locals['obs_tensor']).any():
                self.inf_detected = True
                logger.warning(f"Infinity detected in observation tensor at step {self.num_timesteps}")
                
                # Replace infs with large values
                self.locals['obs_tensor'] = torch.nan_to_num(self.locals['obs_tensor'], posinf=1.0, neginf=0.0)
                
        # Check for NaNs in rewards
        if 'rewards' in self.locals and torch.is_tensor(self.locals['rewards']):
            if torch.isnan(self.locals['rewards']).any():
                logger.warning(f"NaN detected in rewards at step {self.num_timesteps}")
                self.locals['rewards'] = torch.nan_to_num(self.locals['rewards'], nan=0.0)
                
            if torch.isinf(self.locals['rewards']).any():
                logger.warning(f"Infinity detected in rewards at step {self.num_timesteps}")
                self.locals['rewards'] = torch.nan_to_num(self.locals['rewards'], posinf=1.0, neginf=-1.0)
                
        return True

class AgentHandler:
    def __init__(self, env, algorithm='PPO', model_path='./models'):
        """
        Initialize the RL agent handler.
        
        Args:
            env (TradingEnv): Trading environment
            algorithm (str): RL algorithm to use ('PPO', 'A2C', 'DQN')
            model_path (str): Path to save models
        """
        # Check that the environment is valid
        if not isinstance(env, gym.Env):
            raise ValueError("Environment must be a valid Gym environment")
            
        # Wrap the environment in a Monitor for better logging
        self.env = Monitor(env)
        self.algorithm = algorithm
        self.model_path = model_path
        self.model = None
        
        # Validate the environment observation space
        try:
            test_obs = self.env.reset()[0]  # Get initial observation
            
            # Check for NaN values in the initial observation
            if np.isnan(test_obs).any() or np.isinf(test_obs).any():
                logger.warning("Initial observation contains NaN or infinity values! Fixing the environment.")
                
                # Test a single step to ensure it works
                action = self.env.action_space.sample()
                obs, reward, done, truncated, info = self.env.step(action)
                
                if np.isnan(obs).any() or np.isinf(obs).any():
                    logger.warning("Observation after a step contains NaN or infinity values! Environment may have issues.")
                
        except Exception as e:
            logger.error(f"Error validating environment: {str(e)}")
            raise ValueError(f"Environment validation failed: {str(e)}")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize the agent based on the specified algorithm
        self._initialize_agent()
        
    def _initialize_agent(self):
        """Initialize the RL agent based on the specified algorithm."""
        try:
            # Common parameters with better defaults to avoid NaN issues
            policy_kwargs = dict(
                net_arch=[64, 64],  # Smaller network can be more stable
                activation_fn=torch.nn.ReLU  # ReLU can be more stable than tanh
            )
            
            if self.algorithm == 'PPO':
                self.model = PPO('MlpPolicy', self.env, verbose=1, 
                                learning_rate=0.0003,
                                n_steps=2048,
                                batch_size=64,
                                n_epochs=10,
                                gamma=0.99,
                                gae_lambda=0.95,
                                clip_range=0.2,  # Standard PPO clipping
                                normalize_advantage=True,  # Help with training stability
                                ent_coef=0.01,  # Slight entropy for exploration
                                vf_coef=0.5,
                                max_grad_norm=0.5,  # Gradient clipping helps avoid NaNs
                                policy_kwargs=policy_kwargs)
                
            elif self.algorithm == 'A2C':
                self.model = A2C('MlpPolicy', self.env, verbose=1,
                                learning_rate=0.0007,
                                n_steps=5,
                                gamma=0.99,
                                ent_coef=0.01,
                                vf_coef=0.5,
                                max_grad_norm=0.5,
                                policy_kwargs=policy_kwargs)
                
            elif self.algorithm == 'DQN':
                self.model = DQN('MlpPolicy', self.env, verbose=1,
                                learning_rate=0.0001,
                                buffer_size=10000,
                                exploration_fraction=0.1,
                                exploration_final_eps=0.02,
                                batch_size=32,
                                gamma=0.99,
                                target_update_interval=1000,
                                policy_kwargs=policy_kwargs)
                
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
                
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise ValueError(f"Failed to initialize {self.algorithm} agent: {str(e)}")
    
    def train(self, total_timesteps=100000, eval_freq=10000, save_path=None):
        """
        Train the RL agent.
        
        Args:
            total_timesteps (int): Total number of timesteps to train for
            eval_freq (int): Frequency of evaluation during training
            save_path (str, optional): Path to save the final model
            
        Returns:
            model: Trained RL model
        """
        try:
            # Create callback for evaluation
            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
            eval_callback = EvalCallback(self.env, callback_on_new_best=stop_callback, 
                                        eval_freq=eval_freq, verbose=1, 
                                        best_model_save_path=self.model_path)
            
            # Create NaN checking callback
            nan_callback = NanCheckCallback(verbose=1)
            
            # Combine callbacks
            callbacks = [eval_callback, nan_callback]
            
            # Validate the model before training
            logger.info("Validating environment for NaN values before training...")
            
            # Set torch to detect anomalies during training
            torch.autograd.set_detect_anomaly(True)
            
            # Train the agent with error checking
            try:
                self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
            except ValueError as ve:
                if "NaN" in str(ve) or "nan" in str(ve) or "invalid values" in str(ve):
                    logger.error(f"NaN values detected during training: {str(ve)}")
                    logger.info("Attempting to fix the issue and continue training...")
                    
                    # Reload the model with more conservative hyperparameters
                    if self.algorithm == 'PPO':
                        logger.info("Reinitializing PPO with more conservative hyperparameters")
                        self.model = PPO('MlpPolicy', self.env, verbose=1,
                                        learning_rate=0.0001,  # Lower learning rate
                                        n_steps=1024,
                                        batch_size=32,
                                        n_epochs=5,
                                        gamma=0.99,
                                        gae_lambda=0.9,
                                        clip_range=0.1,  # More conservative clipping
                                        ent_coef=0.005,
                                        vf_coef=0.5,
                                        max_grad_norm=0.3,  # More aggressive gradient clipping
                                        use_sde=False)  # Disable stochastic features
                        
                        # Try training again with fewer steps
                        logger.info("Resuming training with updated model...")
                        self.model.learn(total_timesteps=max(10000, total_timesteps // 2), callback=callbacks)
                    else:
                        # For other algorithms, just reduce learning rate and retry
                        logger.warning("Unable to automatically fix NaN issues. Please check your environment and data.")
                        raise
                else:
                    raise
            
            # Save the model if a path is provided
            if save_path:
                self.model.save(save_path)
                
            return self.model
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            if "NaN" in str(e) or "nan" in str(e) or "invalid values" in str(e):
                logger.error("NaN values detected during training. This often happens with unstable environments or extreme observation values.")
                logger.error("Please check your environment implementation and ensure observations are properly normalized and bounded.")
            raise
    
    def test(self, env, num_episodes=1):
        """
        Test the trained agent on a test environment.
        
        Args:
            env (TradingEnv): Test environment
            num_episodes (int): Number of episodes to test
            
        Returns:
            tuple: (episodes_data, mean_reward, std_reward)
        """
        if not self.model:
            raise ValueError("No trained model available. Please train a model first.")
        
        episodes_data = []
        
        for i in range(num_episodes):
            # Reset the environment
            obs, _ = env.reset()  # Updated for gym v0.26+ API
            done = False
            truncated = False
            episode_reward = 0
            step_data = []
            
            while not (done or truncated):  # Handle both termination conditions
                try:
                    # Check for NaN values in observations
                    if np.isnan(obs).any() or np.isinf(obs).any():
                        logger.warning(f"NaN/Inf in test observation at episode {i}. Fixing.")
                        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Predict action
                    action, _states = self.model.predict(obs, deterministic=True)
                    
                    # Take action - updated for gym v0.26+ API
                    obs, reward, done, truncated, info = env.step(action)
                    
                    # Collect data
                    step_info = env.render()
                    step_info['action'] = action
                    step_info['reward'] = reward
                    step_data.append(step_info)
                    
                    episode_reward += reward
                    
                except Exception as e:
                    logger.error(f"Error during testing at episode {i}: {str(e)}")
                    break
            
            # Collect episode data
            if step_data:  # Only if we have data
                episode_df = pd.DataFrame(step_data)
                episode_df['episode'] = i
                episodes_data.append(episode_df)
            
        # Combine all episodes if we have data
        if episodes_data:
            all_episodes_df = pd.concat(episodes_data, ignore_index=True)
            
            # Calculate mean and std reward
            mean_reward = all_episodes_df.groupby('episode')['reward'].sum().mean()
            std_reward = all_episodes_df.groupby('episode')['reward'].sum().std()
            
            return all_episodes_df, mean_reward, std_reward
        else:
            logger.warning("No valid episodes were completed during testing")
            return pd.DataFrame(), 0, 0
    
    def evaluate(self, env, n_eval_episodes=10):
        """
        Evaluate the trained agent.
        
        Args:
            env (TradingEnv): Environment to evaluate on
            n_eval_episodes (int): Number of episodes to evaluate
            
        Returns:
            tuple: (mean_reward, std_reward)
        """
        if not self.model:
            raise ValueError("No trained model available. Please train a model first.")
        
        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=n_eval_episodes)
        
        return mean_reward, std_reward
    
    def plot_results(self, episodes_df):
        """
        Plot the results of the trained agent.
        
        Args:
            episodes_df (pd.DataFrame): DataFrame with episode data
            
        Returns:
            dict: Dictionary with plot figures
        """
        figures = {}
        
        # Plot net worth over time
        fig, ax = plt.subplots(figsize=(14, 7))
        for episode in episodes_df['episode'].unique():
            episode_data = episodes_df[episodes_df['episode'] == episode]
            ax.plot(episode_data['step'], episode_data['net_worth'], label=f'Episode {episode}')
        
        ax.set_title('Net Worth Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Net Worth ($)')
        ax.legend()
        ax.grid(True)
        figures['net_worth'] = fig
        
        # Plot cumulative rewards
        fig, ax = plt.subplots(figsize=(14, 7))
        for episode in episodes_df['episode'].unique():
            episode_data = episodes_df[episodes_df['episode'] == episode]
            cumulative_rewards = episode_data['reward'].cumsum()
            ax.plot(episode_data['step'], cumulative_rewards, label=f'Episode {episode}')
        
        ax.set_title('Cumulative Rewards Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Reward')
        ax.legend()
        ax.grid(True)
        figures['rewards'] = fig
        
        return figures
    
    def load_model(self, model_path):
        """
        Load a pre-trained model.
        
        Args:
            model_path (str): Path to the pre-trained model
            
        Returns:
            model: Loaded RL model
        """
        if self.algorithm == 'PPO':
            self.model = PPO.load(model_path, env=self.env)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(model_path, env=self.env)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(model_path, env=self.env)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        return self.model 