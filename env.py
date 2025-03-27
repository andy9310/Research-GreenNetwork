import numpy as np
class ActionSpace():
    def __init__(self):
        len(StateSpace.topology)
class NetworkEnv():
    """
    A simple custom Gym environment example.
    - Observation space: a single float number in range [-10.0, 10.0]
    - Action space: discrete (2 actions: 0 or 1)
    """

    def __init__(self):
        super(NetworkEnv, self).__init__()

        # Define action and observation space
        self.action_space = ActionSpace()
        self.observation_space = StateSpace()

        self.state = None
        self.step_count = 0
        # self.max_steps = 20

    def reset(self):
        """
        Reset the environment to an initial state
        """
        self.state = np.array([0.0])  # starting from 0
        self.step_count = 0
        return self.state

    def step(self, action):
        """
        Execute one time step within the environment
        """
        self.step_count += 1

        # Apply action effect
        if action == 0:
            self.state[0] -= 1.0
        elif action == 1:
            self.state[0] += 1.0
        else:
            raise ValueError("Invalid action")

        # Clip the state to the observation bounds
        self.state = np.clip(self.state, -10.0, 10.0)

        # Define a reward: encourage moving towards +10
        reward = 1.0 if self.state[0] > 0 else -1.0

        # Check if done (maximum steps or reaching limits)
        done = bool(
            self.step_count >= self.max_steps
            or self.state[0] <= -10.0
            or self.state[0] >= 10.0
        )

        info = {}

        return self.state, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment (optional)
        """
        print(f"Current state: {self.state[0]}")

    def close(self):
        pass
