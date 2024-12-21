import traci
import gym
import numpy as np


class TrafficEnv(gym.Env):
    def __init__(self, sumo_config_path):
        """
        Initializes the traffic environment for SUMO simulations.
        :param sumo_config_path: Path to the SUMO configuration file.
        """
        super(TrafficEnv, self).__init__()
        self.sumo_config_path = sumo_config_path
        self.sumo_cmd = ["sumo", "-c", self.sumo_config_path]
        self.action_space = gym.spaces.Discrete(2)  # Actions: 0 (green), 1 (red)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.state = np.zeros(4)

    def reset(self):
        """
        Resets the SUMO simulation and returns the initial state.
        """
        traci.start(self.sumo_cmd)
        self.state = self.get_state()
        return self.state

    def get_state(self):
        """
        Retrieves the current state of the environment.
        :return: An array containing ramp vehicles, highway vehicles, ramp waiting time, and highway flow.
        """
        ramp_vehicles = len(traci.edge.getLastStepVehicleIDs("E5"))
        highway_vehicles = len(traci.edge.getLastStepVehicleIDs("E1"))
        ramp_waiting_time = traci.edge.getWaitingTime("E5")
        highway_flow = traci.edge.getLastStepVehicleNumber("E1")
        return np.array([ramp_vehicles, highway_vehicles, ramp_waiting_time, highway_flow])

    def step(self, action):
        """
        Executes the given action and advances the simulation by one step.
        :param action: 0 for green on ramp, 1 for green on highway.
        :return: Tuple containing the next state, reward, done flag, and additional info.
        """
        traci.trafficlight.setRedYellowGreenState("J7", "G" if action == 0 else "r")
        traci.simulationStep()
        self.state = self.get_state()
        reward = -self.state[2] + self.state[3]  # Reduce waiting time, increase flow
        done = traci.simulation.getMinExpectedNumber() == 0  # End of simulation if no vehicles remain
        return self.state, reward, done, {}

    def close(self):
        """
        Closes the SUMO simulation.
        """
        traci.close()
