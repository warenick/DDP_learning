#! /usr/bin/python3
from env.Agent import Agent
from scripts.utilites.Visualizer import Visualizer
from DDP import DDP 
import time

if __name__=="__main__":
    viz = Visualizer()
    dt = 0.4
    # horizon = 10 # 10 by default
    gradient_rate = 1.0
    regularisation = 0.95

    ddp = DDP(gradient_rate, regularisation)
    agent = Agent(initial_state=[-0.5, 0.5, 0],  goal=[-2, 3, 0],  dt=dt, name="1") 
    
    t1 = time.time()
    agent.prediction["state"], agent.prediction["controll"] = ddp.optimize(agent, 50, viz)
    print(f"calculation time: {(time.time()-t1):.3}s",)
    ddp.initial_gradient_rate = 0.2
    for _ in range(10):
        agent.step()
        agent.prediction["state"], agent.prediction["controll"] = ddp.optimize(agent, 5, viz)
        viz.pub_agent_state([agent])
    exit()