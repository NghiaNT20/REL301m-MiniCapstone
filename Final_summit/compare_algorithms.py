import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from env.cutting_stock import CuttingStockEnv
from policy.FirstFit_Policy import FirstFitPolicy
from policy.BestFit_Policy import BestFitPolicy
from policy.Combination_Policy import CombinationPolicy
from policy.QLearning_Policy import QLearningPolicy

def evaluate_policy(policy_class, policy_name, env, runs=5, **policy_kwargs):
    """Evaluate a policy on the environment for multiple runs"""
    results = []
    
    for i in range(runs):
        env.reset()
        start_time = time.time()
        
        if policy_name == "Q-Learning":
            # Q-Learning needs different initialization
            policy = policy_class(env=env, **policy_kwargs)
            # Try to load existing policy
            policy.load_policy()
            reward, steps, info = policy.execute_policy(render=(i == 0))
        else:
            # Heuristic policies
            policy = policy_class(**policy_kwargs)
            state = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                if i == 0:  # Render only first run
                    env.render()
                
                action = policy.select_action(env)
                if action is None:
                    break
                    
                state, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
            
            reward = total_reward
            steps = step
            
        elapsed_time = time.time() - start_time
        
        results.append({
            'Algorithm': policy_name,
            'Run': i + 1,
            'Reward': reward,
            'Steps': steps,
            'Time': elapsed_time,
            'Used Stocks': info.get('used_stocks', 0),
            'Remaining Products': info.get('remaining_products', 0),
            'Trim Loss': info.get('trim_loss', 0)
        })
    
    return results

def plot_comparison(results_df, save_path="results/algorithm_comparison.png"):
    """Plot comparison of different algorithms"""
    # Group by algorithm and calculate mean
    grouped = results_df.groupby('Algorithm').mean()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Reward comparison
    grouped['Reward'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_ylabel('Reward')
    
    # Used Stocks comparison
    grouped['Used Stocks'].plot(kind='bar', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title('Average Used Stocks')
    axes[0, 1].set_ylabel('Count')
    
    # Trim Loss comparison
    grouped['Trim Loss'].plot(kind='bar', ax=axes[1, 0], color='salmon')
    axes[1, 0].set_title('Average Trim Loss (%)')
    axes[1, 0].set_ylabel('Percentage')
    
    # Execution Time comparison
    grouped['Time'].plot(kind='bar', ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Average Execution Time')
    axes[1, 1].set_ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Initialize environment
    env = CuttingStockEnv(
        data_path="data/data.csv",
        max_steps=1000,
        normalize_reward=True
    )
    
    # Define algorithms to compare
    algorithms = [
        {
            'policy_class': FirstFitPolicy,
            'policy_name': 'First Fit',
            'policy_kwargs': {}
        },
        {
            'policy_class': BestFitPolicy,
            'policy_name': 'Best Fit',
            'policy_kwargs': {}
        },
        {
            'policy_class': CombinationPolicy,
            'policy_name': 'Combination',
            'policy_kwargs': {}
        },
        {
            'policy_class': QLearningPolicy,
            'policy_name': 'Q-Learning',
            'policy_kwargs': {
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'exploration_rate': 0.05,  # Low for evaluation
                'save_path': "results/q_table.pkl"
            }
        }
    ]
    
    # Evaluate all algorithms
    all_results = []
    for algo in algorithms:
        print(f"Evaluating {algo['policy_name']}...")
        results = evaluate_policy(
            policy_class=algo['policy_class'],
            policy_name=algo['policy_name'],
            env=env,
            runs=5,
            **algo['policy_kwargs']
        )
        all_results.extend(results)
        
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv("results/algorithm_comparison.csv", index=False)
    print("Results saved to results/algorithm_comparison.csv")
    
    # Plot comparison
    plot_comparison(results_df)
    print("Comparison plot saved to results/algorithm_comparison.png")
    
    # Display summary
    print("\nAlgorithm Comparison Summary:")
    summary = results_df.groupby('Algorithm').mean()
    print(summary[['Reward', 'Used Stocks', 'Remaining Products', 'Trim Loss', 'Time']])

if __name__ == "__main__":
    main()