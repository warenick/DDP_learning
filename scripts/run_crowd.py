#! /usr/bin/python3
from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
from env.Crowd import Crowd
from DDP import DDP 

if __name__=="__main__":
    viz = Visualizer_ros()
    crowd = Crowd(visualizer=viz)
    crowd.read_from_conf("configs.X4")
    import time
    t1 = time.time()
    crowd.optimize(epochs=10, visualize=True)
    print(f"calculation time: {(time.time()-t1):.3}s",)

    for _ in range(10):
        crowd.step()
        crowd.optimize(epochs=1, visualize=False)
        crowd.visualaze()
    exit()