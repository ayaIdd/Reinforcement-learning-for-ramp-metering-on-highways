import traci
import numpy as np
import random

# Constants
NUM_PHASES = 2  # Number of traffic light phases (0: Green on ramp, 1: Green on highway)
STATE_DIM = 6  # Number of state variables: [highway_queue_length, ramp_queue_length, highway_speed, ramp_speed, highway_occupancy, ramp_occupancy]
ACTION_DIM = NUM_PHASES  # Number of possible traffic light phases
GAMMA = 0.9  # Discount factor for future rewards
EPSILON = 0.1  # Epsilon for epsilon-greedy action selection
ALPHA = 0.1  # Learning rate
MEMORY_CAPACITY = 10000  # Size of experience replay memory
BATCH_SIZE = 32  # Batch size for training (not used directly in Q-learning without experience replay)
NUM_EPISODES = 100  # Number of episodes to train

# Initialize Q-table: rows = state space, columns = action space
Q_table = np.zeros((STATE_DIM, ACTION_DIM))

def get_state():
    """Retrieve the state from the SUMO simulation."""
    highway_queue = traci.edge.getLastStepHaltingNumber("highway")
    ramp_queue = traci.edge.getLastStepHaltingNumber("ramp")
    highway_speed = traci.edge.getLastStepMeanSpeed("highway")
    ramp_speed = traci.edge.getLastStepMeanSpeed("ramp")
    highway_occupancy = traci.edge.getLastStepOccupancy("highway")
    ramp_occupancy = traci.edge.getLastStepOccupancy("ramp")

    state = np.array([highway_queue, ramp_queue, highway_speed, ramp_speed, highway_occupancy, ramp_occupancy])
    return state

def choose_action(state):
    """Choose action using epsilon-greedy policy."""
    if random.random() < EPSILON:
        # Choose a random action
        return random.randint(0, ACTION_DIM - 1)
    else:
        # Choose the action with the highest Q-value
        state_idx = np.array(state).astype(int)  # Convert state to integer for indexing (this can be refined)
        return np.argmax(Q_table[state_idx])

def calculate_reward(state, action, next_state):
    """Reward function to optimize traffic flow."""
    highway_queue = state[0]
    ramp_queue = state[1]
    highway_speed = state[2]
    ramp_speed = state[3]

    # Example reward: minimize ramp waiting time and maximize highway speed
    reward = -(ramp_queue) + highway_speed - highway_queue + ramp_speed
    return reward

def update_q_table(state, action, reward, next_state):
    """Update the Q-table using the Q-learning update rule."""
    state_idx = np.array(state).astype(int)  # Convert state to integer for indexing
    next_state_idx = np.array(next_state).astype(int)

    # Q-learning update rule
    max_next_q = np.max(Q_table[next_state_idx])
    Q_table[state_idx, action] = Q_table[state_idx, action] + ALPHA * (reward + GAMMA * max_next_q - Q_table[state_idx, action])

def check_if_done():
    """Check if the episode is done based on simulation time."""
    return traci.simulation.getTime() > 1000  # Example condition, change to your own simulation condition

# Connect to SUMO
traci.start(["sumo", "-c", "sumo_config/simulation.sumocfg"])

# Training loop
for episode in range(NUM_EPISODES):
    state = get_state()
    total_reward = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        action = choose_action(state)

        # Apply the chosen action to the traffic light in SUMO (example: set the phase)
        traci.trafficlight.setPhase("ramp_light", action)

        # Step the simulation
        traci.simulationStep()

        # Get the next state after the simulation step
        next_state = get_state()

        # Calculate reward for the current state-action transition
        reward = calculate_reward(state, action, next_state)

        # Update the Q-table
        update_q_table(state, action, reward, next_state)

        # Update the current state
        state = next_state
        total_reward += reward

        if check_if_done():
            break

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Save Q-table for future use
np.save("q_table.npy", Q_table)

# Close connection to SUMO
traci.close()
