#! /usr/bin/python3
from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
from DDP import DDP 

if __name__=="__main__":
    viz = Visualizer_ros()
    dt = 0.4
    horizon = 12
    epochs = 100
    gradient_rate = 1.0
    regularisation = 0.95
    ddp = DDP(gradient_rate, regularisation)
    agent = Agent(goal=[-2, 5, 0], dt=dt) # state = [x,y,yaw]
    agent.prediction["state"], agent.prediction["controll"] = ddp.optimize(agent,epochs,viz) # optimize trajectory
    # viz.pub_agent_state([agent])
    exit()