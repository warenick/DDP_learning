#! /usr/bin/python3
from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
import torch
from CDDP import CDDP 
from DDP import DDP 

if __name__=="__main__":
    dt = torch.tensor(0.1)
    horizon = 10
    epochs = 10
    gradient_rate = 0.1

    viz = Visualizer_ros()
    agent = Agent(goal=torch.tensor([2, 1, 0, 0., 0.]), dt=dt) # [x,y,yaw,V,Vyaw]
    u = torch.zeros((horizon,agent.state.shape[-1])) # [[0,0,0,V,Vyaw],[0,0,0,V,Vyaw],...]
    u[:,-2]=2. # set initial V
    u[:,-1]=-1. # set initial Vyaw
    state_dim = agent.state.shape[-1]
    agent.update_history(u) # calc nominal trajectory(->agent.history)    
    viz.pub_agent_state([agent]) # just vis in rviz
    ddp = CDDP(agent, gradient_rate) # init differencial functions
    for epoch in range(epochs):
        k_seq, kk_seq                                     = ddp.backward(agent.history["state"],agent.history["controll"])
        agent.history["state"], agent.history["controll"] = ddp.forward(agent.history["state"],agent.history["controll"], k_seq, kk_seq)
        agent.state = agent.history["state"][-1].clone().detach()
        viz.pub_agent_state([agent]) # just vis in rviz


    exit()