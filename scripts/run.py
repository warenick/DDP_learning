#! /usr/bin/python3
from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
from env.Crowd import Crowd
from DDP import DDP 

if __name__=="__main__":
    viz = Visualizer_ros()
    dt = 0.4
    horizon = 12
    epochs = 30
    gradient_rate = 1.0
    regularisation = 0.95

    ddp = DDP(gradient_rate, regularisation)
    agent1 = Agent(initial_state=[-0.5, 0.5, 0], goal=[-2, 3, 0], dt=dt) # state = [x,y,yaw]
    agent2 = Agent(initial_state=[0.5, 0.5, 0], goal=[2, 3, 0], dt=dt) # state = [x,y,yaw]
    agent3 = Agent(initial_state=[0.5, -0.5, 0], goal=[2, -3, 0], dt=dt) # state = [x,y,yaw]
    agent4 = Agent(initial_state=[-0.5, -0.5, 0], goal=[-2, -3, 0], dt=dt) # state = [x,y,yaw]
    crowd = Crowd(visualizer=viz)
    crowd.add_agent(agent1,ddp)
    crowd.add_agent(agent2,ddp)
    crowd.add_agent(agent3,ddp)
    crowd.add_agent(agent4,ddp)
    crowd.optimize(epochs,visualize=True)
    crowd.visualaze()
    # agent.prediction["state"], agent.prediction["controll"] = ddp.optimize(agent, epochs, viz) # optimize trajectory
    # viz.pub_agent_state([agent])
    exit()