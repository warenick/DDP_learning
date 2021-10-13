# import numpy as np
import autograd.numpy as np


class Agent():
    def __init__(self, initial_state=np.array([0., 0., 0., 0., 0.]),  # [x, y, yaw, dv, dyaw]
                 goal=np.array([10, 10, 0, 0, 0]),
                 type="agent",
                 kinematic_type="differencial",
                 dt=0.1,
                 umax=np.array([0, 0, 0, 5.0, 5.0])):
        self.state_initial = initial_state
        self.goal = goal
        self.type = type
        self.dt = dt
        self.kinematic_type = kinematic_type
        self.umax = umax
        self.history = {}
        self.update_history()
    
    def update_history(self, controll_arr = None):
        self.history["state"] = [np.copy(self.state_initial)]
        self.history["controll"] = []
        self.state = np.copy(self.state_initial)
        if controll_arr is not None:
            # generate nominal trajectory
            for u in controll_arr:
                new_state = self.step(u)
                self.history["state"].append(np.copy(new_state))
                self.history["controll"].append(np.copy(u))

    def step_func(self, x, u):
        if "differencial" in self.kinematic_type:
            # u = [0,0,0,v,vyaw]
            pose = x[:3]
            velocity = x[3:]
            velocity = self.dt*np.clip(u[3:]+velocity, -self.umax[3:], self.umax[3:])
            # velocity = self.dt*np.clip(u[3:], -self.umax[3:], self.umax[3:])
            pose = pose + np.matmul(np.array([[np.cos(x[2]), 0],
                                              [np.sin(x[2]), 0],
                                              [0,                    1]]), velocity)
        return np.concatenate((pose,velocity)) 
            # x[3:] = self.dt*np.clip(u+x[3:], -self.umax, self.umax)
            # x[:3] = x[:3] + np.matmul(np.array([[np.cos(x[2]), 0],
            #                                                       [np.sin(x[2]), 0],
            #                                                       [0,                    1]]), x[3:])
        # return x

    def step(self, controll):
        self.state = self.step_func(self.state, controll)
        return self.state
        # action = u[v,yaw]

    def final_cost(self,state):
        # state[x,y,yaw]
        return 10*np.linalg.norm(state[:3]-self.goal[:3])

    def running_cost(self, state, controll):
        return self.final_cost(state)+0.5*np.sum(np.square(controll))
