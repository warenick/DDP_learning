import numpy as np


class Agent():
    def __init__(self, initial_state=np.array([0., 0., 0., 0., 0.]),  # [x, y, yaw, dv, dyaw]
                 goal=np.array([10, 10]),
                 type="agent",
                 kinematic_type="differencial",
                 dt=0.1,
                 umax=np.array([5., 3.])):
        self.state_initial = initial_state
        self.goal = goal
        self.type = type
        self.dt = dt
        self.kinematic_type = kinematic_type
        self.umax = umax
        self.history = {}
        self.update_history()
    
    def update_history(self):
        self.history["state"] = [self.state_initial]
        self.history["cost"] = [0.]
        self.state = np.copy(self.state_initial)


    def step(self, controll):
        if "differencial" in self.kinematic_type:
            # u = [v,vyaw]
            self.state[3:] = np.clip(self.dt*(controll+self.state[3:]), -self.umax, self.umax)
            self.state[:3] = self.state[:3] + np.matmul(np.array([[np.cos(self.state[2]), 0],
                                                                  [np.sin(self.state[2]), 0],
                                                                  [0,                    1]]), self.state[3:])
            self.history["state"].append(np.copy(self.state))
        return self.state 
        # action = u[v,yaw]

    def final_cost(self):
        return np.linalg.norm(self.state[:2]-self.goal)