from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
import numpy as np
from time import sleep
import random


dt = 0.1
horizon = 20
epochs = 10

if __name__=="__main__":
    viz = Visualizer_ros()
    agent = Agent()
    u = np.zeros((horizon,2)) # [[V,Vyaw],[V,Vyaw],...]
    u[:,0]=1.  
    state_dim = agent.state.shape[-1]
    # calc nominal trajectory(->agent.history)
    for controll in u:
        agent.step(controll)
        viz.pub_agent_state([agent]) # just vis in rviz
    agent.update_history()
    for epoch in range(epochs):
        # BACKWARD pass
        V = [0.0 for _ in range(horizon + 1)]
        V_x = [np.zeros(state_dim) for _ in range(horizon + 1)]
        V_xx = [np.zeros((state_dim, state_dim)) for _ in range(horizon + 1)]
        for i in reversed(range(horizon)):
            # Qx = 
            # Qu = 
            # Qxx = 
            # Quu = 
            # Qux =
            # k = -Quu(-1)@Qu
            # K =  -Quu(-1)@Qux
            pass
        # FORWARD pass
        # calc nominal trajectory(->agent.history)
        for controll in u:
            agent.step(controll)
            viz.pub_agent_state([agent]) # just vis in rviz
        agent.update_history()


    exit()