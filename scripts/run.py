#! /usr/bin/python3
from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
import torch
from DDP import DDP 

if __name__=="__main__":
    dt = torch.tensor(0.4)
    horizon = 12
    epochs = 100
    gradient_rate = 1.0
    regularisation = 0.95
    viz = Visualizer_ros()
    agent = Agent(goal=torch.tensor([-2, 5, 0]), dt=dt) # [x,y,yaw,V,Vyaw]
    agent.update_history(torch.ones((horizon,2))) # calc nominal trajectory(->agent.prediction)    # u =  # [[V,Vyaw],[V,Vyaw],...]
    viz.pub_agent_state([agent]) # just vis in rviz
    ddp = DDP(agent, gradient_rate, regularisation)
    for epoch in range(epochs):
        k_seq, kk_seq                                     = ddp.backward(agent.prediction["state"],agent.prediction["controll"])
        agent.prediction["state"], agent.prediction["controll"] = ddp.forward(agent.prediction["state"],agent.prediction["controll"], k_seq, kk_seq)
        viz.pub_agent_state([agent]) # just vis in rviz
    exit()