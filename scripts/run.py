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
    u = np.ones((horizon,2))
    # rng = np.random.default_rng()
    # rng.standard_normal(2)
    for epoch in range(epochs):
        # forward pass
        for controll in u:
            agent.step(controll)
        
        viz.pub_agent_state([agent])
        


    exit()