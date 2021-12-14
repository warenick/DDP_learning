#! /usr/bin/python3
from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
import torch
from CDDP import CDDP 
from DDP import DDP 

if __name__=="__main__":
    dt = torch.tensor(0.4)
    horizon = 12
    epochs = 100
    gradient_rate = 1.0
    regularisation = 0.95
    viz = Visualizer_ros()
    agent = Agent(goal=torch.tensor([-2, 5, 0]), dt=dt) # [x,y,yaw,V,Vyaw]
    agent.update_history(torch.ones((horizon,2))) # calc nominal trajectory(->agent.history)    # u =  # [[V,Vyaw],[V,Vyaw],...]
    viz.pub_agent_state([agent]) # just vis in rviz
    ddp = CDDP(agent, gradient_rate, regularisation)
    for epoch in range(epochs):
        k_seq, kk_seq                                     = ddp.backward(agent.history["state"],agent.history["controll"])
        agent.history["state"], agent.history["controll"] = ddp.forward(agent.history["state"],agent.history["controll"], k_seq, kk_seq)
        agent.state = agent.history["state"][-1].clone().detach()
        viz.pub_agent_state([agent]) # just vis in rviz
    exit()