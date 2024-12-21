import random
from env import TrafficEnv

sumo_config_path = "sumo_config/simulation.sumocfg"  # Path to your SUMO configuration file
env = TrafficEnv(sumo_config_path)

state = env.reset()
done = False

while not done:
    action = random.choice([0, 1])  # Randomly select action
    state, reward, done, _ = env.step(action)
    print(f"State: {state}, Reward: {reward}")

env.close()
