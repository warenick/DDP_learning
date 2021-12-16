#! /usr/bin/python3
from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
from env.Crowd import Crowd
from DDP import DDP 

if __name__=="__main__":
    viz = Visualizer_ros()
    dt = 0.4
    # horizon = 10 # 10 by default
    epochs = 30
    gradient_rate = 1.0
    regularisation = 0.95
    # TODO: generate from config file
    ddp = DDP(gradient_rate, regularisation)
    agent1 = Agent(initial_state=[-0.5, 0.5, 0],  goal=[-2, 3, 0],  dt=dt) 
    agent2 = Agent(initial_state=[0.5, 0.5, 0],   goal=[2, 3, 0],   dt=dt) 
    agent3 = Agent(initial_state=[0.5, -0.5, 0],  goal=[2, -3, 0],  dt=dt) 
    agent4 = Agent(initial_state=[-0.5, -0.5, 0], goal=[-2, -3, 0], dt=dt) 
    crowd = Crowd(visualizer=viz)
    crowd.add_agent(agent1, ddp)
    crowd.add_agent(agent2, ddp)
    crowd.add_agent(agent3, ddp)
    crowd.add_agent(agent4, ddp)

    import time
    t1 = time.time()
    crowd.optimize(epochs, visualize=True)
    print(f"calculation time: {(time.time()-t1):.3}s",)
    
    crowd.visualaze()
    exit()