#! /usr/bin/python3
from utilites.Visualizer import Visualizer
from utilites.CostmapReader import CostmapReader
from env.Crowd import Crowd
import time

if __name__=="__main__":
    viz = Visualizer()
    cr = CostmapReader()
    crowd = Crowd(visualizer=viz)
    # crowd.read_from_conf("configs.X4")
    
    # crowd.read_from_conf("configs.H4_social_costmap", costmap=cr)
    # crowd.read_from_conf("configs.H4_costmap", costmap=cr)
    # crowd.read_from_conf("configs.H4_social")
    # crowd.read_from_conf("configs.H4_linear")
    crowd.read_from_conf("configs.H4_mix")
    # crowd.read_from_conf("configs.costs_issue1")
    # crowd.read_from_conf("configs.costs_issue2")
    # crowd.read_from_conf("configs.costs_issue3")
    t1 = time.time()
    crowd.optimize(epochs=5, visualize=True)
    print(f"calculation time: {(time.time()-t1):.3}s",)
    for _ in range(60):
        crowd.step()
        crowd.optimize(epochs=2, visualize=True)
        crowd.visualaze()
    exit()