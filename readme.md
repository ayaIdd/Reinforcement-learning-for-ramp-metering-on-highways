# Reinforcement Learning for Ramp Metering Optimization

This project applies **Reinforcement Learning (RL)** algorithms—**Q-learning** and **Deep Q-learning (DQL)**—to optimize ramp metering on highways, improving traffic flow by controlling traffic lights at the ramp. The goal is to compare the performance of RL models against a **No Traffic Light** baseline using the **SUMO** traffic simulator.

## Project Summary

### Three Scenarios:
1. **No Traffic Lights**: A baseline where no traffic control is applied, and traffic flows freely.
2. **Q-learning**: A model-free RL algorithm to optimize traffic flow by adjusting traffic lights.
3. **Deep Q-learning (DQL)**: A neural network-based Q-learning approach to better handle more complex traffic scenarios.

### Key Files:
- `q_network_model.keras`: The trained DQN model.
- `q_table.pkl`: Saved Q-table for Q-learning.
- `DQ_learning.ipynb`: Jupyter notebook for DQL training.
- `Q_learning.ipynb`: Jupyter notebook for Q-learning training.
- `trained_traffic_light_model_ql.keras`: Trained model for Q-learning ramp metering.
- `testing_NoTrafficLight.ipynb`: Testing the baseline scenario.
- `testing_Q-learning.ipynb`: Testing Q-learning traffic control.
- `testing_DQL.ipynb`: Testing Deep Q-learning traffic control.
## How to Run the Model
1. **Run the Jupyter Notebooks**:
   - Run `Q_learning.ipynb` to train and test the Q-learning model.
   - Run `DQ_learning.ipynb` to train and test the Deep Q-learning model.
   - Test the baseline scenario using `testing_NoTrafficLight.ipynb`.

2. You can test all models by running the testing notebooks: `testing_Q-learning.ipynb` and `testing_DQL.ipynb`.

## Future Work

- Improve RL model training and reward structures.
- Experiment with other RL algorithms like Double Q-learning or Actor-Critic.
- Expand the simulation to handle more complex traffic environments.

