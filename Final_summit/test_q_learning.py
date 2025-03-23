import pickle
import numpy as np
from env.cutting_stock import CuttingStockEnv

# Load the trained Q-table
q_table_path = "results/q_table.pkl"
with open(q_table_path, "rb") as f:
    Q_table = pickle.load(f)

# Initialize the environment
env = CuttingStockEnv(
    render_mode="human",
    max_w=120,
    max_h=120,
    seed=42,
    stock_list=[
        (80, 70), (90, 80), (70, 50), (80, 100), (90, 70),
        (100, 80), (110, 60), (100, 80), (100, 90), (100, 100),
        (70, 120), (90, 120), (110, 100), (60, 120), (120, 120)
    ],
    product_list=[
        (30, 25), (15, 10), (20, 10), (25, 25), (30, 20),
        (35, 20), (35, 30), (45, 25), (50, 30), (55, 35),
        (20, 25), (25, 10), (30, 15), (35, 20), (40, 55),
        (45, 30), (50, 35), (35, 40), (60, 45), (65, 50),
        (50, 30), (35, 40), (60, 50), (55, 40), (90, 60),
        (15, 10), (20, 15), (25, 20), (70, 25), (35, 30)
    ]
)

# Define a function to get the state index
def get_state(observation):
    # Simplified state representation for testing
    return hash(str(observation)) % Q_table.shape[0]

# Test the trained policy
observation = env.reset()
# Check if reset returns a tuple (newer Gym/Gymnasium API)
if isinstance(observation, tuple):
    observation = observation[0]  # In new API, it returns (obs, info)

done = False
total_reward = 0

while not done:
    state = get_state(observation)
    action = np.argmax(Q_table[state])  # Select the best action from Q-table
    env_action = {
        "stock_idx": action // len(env.product_list),
        "size": env.product_list[action % len(env.product_list)],
        "position": (0, 0)  # Simplified for testing
    }
    
    # Handle both old and new Gym/Gymnasium API
    step_result = env.step(env_action)
    
    # Check the return type and unpack accordingly
    if len(step_result) == 4:  # Old API: obs, reward, done, info
        observation, reward, done, info = step_result
    elif len(step_result) == 5:  # New API: obs, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    
    total_reward += reward
    env.render()

print(f"Total reward achieved: {total_reward}")
env.close()