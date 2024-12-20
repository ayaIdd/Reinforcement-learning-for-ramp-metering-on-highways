import traci
import gym 
import random


class TrafficEnv(gym.Env):
    def __init__(self, sumo_config_path):
        super(TrafficEnv, self).__init__()
        self.sumo_config_path = sumo_config_path
        self.sumo_cmd = ["sumo", "-c", self.sumo_config_path]
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0 (green), 1 (red)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.state = np.zeros(4)

    def reset(self):
        traci.start(self.sumo_cmd)
        self.state = self.get_state()
        return self.state

    def get_state(self):
        ramp_vehicles = len(traci.edge.getLastStepVehicleIDs("ramp"))
        highway_vehicles = len(traci.edge.getLastStepVehicleIDs("highway"))
        ramp_waiting_time = traci.edge.getWaitingTime("ramp")
        highway_flow = traci.edge.getLastStepVehicleNumber("highway")
        return np.array([ramp_vehicles, highway_vehicles, ramp_waiting_time, highway_flow])

    def step(self, action):
        traci.trafficlight.setRedYellowGreenState("traffic_light_0", "Gr" if action == 0 else "rG")
        traci.simulationStep()
        self.state = self.get_state()
        reward = -self.state[2] + self.state[3]  # Reward: reduce waiting time, increase flow
        done = traci.simulation.getMinExpectedNumber() == 0  # End of simulation
        return self.state, reward, done, {}

    def close(self):
        traci.close()


sumo_config_path = "sumo_config\simulation.sumocfg"
env = TrafficEnv(sumo_config_path)


state = env.reset()
done = False

while not done:
    action = random.choice([0, 1])  # Random action: 0 or 1
    state, reward, done, _ = env.step(action)
    print(f"State: {state}, Reward: {reward}")

env.close()

