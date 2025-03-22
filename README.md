# Q-Learning Algorithm for 2D Cutting Stock Problem

This project implements a Q-Learning reinforcement learning approach to solve the 2D Cutting Stock Problem. The algorithm learns to efficiently cut smaller rectangular pieces (products) from larger rectangular sheets (stocks) while minimizing waste.

## Project Structure

```
Final_summit/
├── QLearning.py               # Main Q-Learning implementation
├── test.py                    # Enhanced Q-Learning with improved exploration strategies
├── visualize_metrics.py       # Visualization tools for training process and results
├── data/                      # Data directory for stock and product specifications
├── env/                       # Environment implementation
│   ├── cutting_stock.py       # Cutting stock environment 
│   └── ...
├── policy/                    # Policy implementations
│   └── QLearning_Policy.py    # Q-Learning policy class
├── results/                   # Output directory for training results
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Pygame (for visualization)
- Seaborn
- OpenCV (for animation generation)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Final_summit

# Install required packages
pip install numpy pandas matplotlib pygame seaborn opencv-python
```

## Running the Q-Learning Algorithm

### Basic Q-Learning

To run the basic Q-Learning algorithm:

```bash
python QLearning.py
```

This will:
1. Initialize the cutting stock environment
2. Train the Q-Learning agent for 500 episodes
3. Save the trained Q-table to `results/q_table.pkl`
4. Generate training logs in `results/q_learning_log.txt` and detailed metrics in `results/detailed_cutting_log.txt`
5. Display the best solution found during training

### Enhanced Q-Learning

For the improved Q-Learning implementation with advanced exploration strategies:

```bash
python test.py
```

The enhanced version includes:
- Improved state representation
- Boltzmann exploration strategy
- Guided exploration using domain knowledge
- Dynamic reward scaling
- Curriculum learning

### Configuration Options

You can modify the following parameters in the scripts:

- `num_episodes`: Number of training episodes (default: 500)
- `alpha`: Learning rate (default: 0.3)
- `gamma`: Discount factor (default: 0.9)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_decay`: Rate at which exploration decreases (default: 0.995)
- `min_epsilon`: Minimum exploration rate (default: 0.01)

## Visualizing Results

To generate visualizations of the training process and results:

```bash
python visualize_metrics.py
```

This will create:
1. Training progress plots (rewards, epsilon decay, steps per episode)
2. Q-table heatmap visualization
3. Material usage and waste percentages charts
4. A comprehensive HTML report at `results/metrics_visualization_report.html`

## Understanding the Output

After training, the following files will be available:

- `results/q_table.pkl`: The trained Q-table (state-action values)
- `results/q_learning_log.txt`: Training log with rewards and steps
- `results/detailed_cutting_log.txt`: Detailed metrics for each episode
- `results/plots/`: Directory containing all visualization plots

## Algorithm Details

The Q-Learning implementation features:

1. **State Representation**: Encoding of the cutting environment state (stock utilization, remaining products)
2. **Action Selection**: Epsilon-greedy policy with decay
3. **Reward Function**: Rewards based on material usage, trim loss, and stock efficiency
4. **Learning Process**: Update Q-values based on immediate rewards and future expected returns
5. **Exploration vs Exploitation**: Balance between exploring new actions and exploiting known good actions

## Visualization Features

The visualization script generates several types of plots:

- **Rewards over Episodes**: Shows how rewards change during training
- **Steps per Episode**: Number of actions taken in each episode
- **Trim Loss**: Percentage of wasted material over time
- **Material Usage**: Efficiency of material utilization
- **Stocks Used**: Number of stock sheets used in each episode
- **Remaining Products**: Products left uncut after each episode
- **Correlation Matrix**: Relationships between different metrics
- **Combined Dashboard**: Overview of all key metrics
- **Q-table Analysis**: Distribution and sparsity of learned Q-values

## Tips for Best Results

- For larger problems, increase the number of training episodes
- Adjust exploration parameters based on problem complexity
- Use visualization tools to monitor training progress
- Try different reward function configurations by modifying `get_reward()`

## Detailed Metrics Explanation

- **Trim Loss**: Percentage of unused space in utilized stock sheets
- **Material Usage**: Percentage of stock sheet area that's productively used
- **Stocks Used**: Number of stock sheets utilized in a solution
- **Remaining Products**: Number of products that couldn't be cut in the solution

## How to Read Visualization Results

- **Upward trends** in rewards and material usage indicate improvement
- **Downward trends** in trim loss, stocks used, and remaining products indicate improvement
- **Correlation plots** help understand relationships between different aspects of performance
- **Highlight points** show the best performance episodes for each metric

## How to Extend the Algorithm

1. **State Representation**: Modify the `get_state()` function to capture additional features
2. **Reward Function**: Customize `get_reward()` to prioritize different aspects
3. **Action Selection**: Change the exploration strategy in the main training loop
4. **Environment**: Adapt the environment to handle different cutting constraints
5. **Hyperparameters**: Tune learning rate, discount factor, and exploration parameters

## Troubleshooting

- If you encounter memory issues, reduce the state space representation
- If learning is slow, try adjusting the learning rate or reward scaling
- If visualization fails, check if all output files were properly generated
- For poor solutions, try increasing the number of episodes or adjusting the reward function

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
- Bennell, J. A., & Oliveira, J. F. (2008). The geometry of nesting problems: A tutorial.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine learning.