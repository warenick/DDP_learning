#! /usr/bin/python3
from env.Visualizer_ros import Visualizer_ros
from env.Crowd import Crowd

if __name__=="__main__":
    viz = Visualizer_ros()
    crowd = Crowd(visualizer=viz)
    # crowd.read_from_conf("configs.X4")
    crowd.read_from_conf("configs.H4")
    import time
    t1 = time.time()
    crowd.optimize(epochs=5, gradient_rate=0.2, regularisation=0.90, visualize=True)
    print(f"calculation time: {(time.time()-t1):.3}s",)
    for _ in range(30):
        crowd.step()
        crowd.optimize(epochs=2, gradient_rate=0.15, visualize=True)
        crowd.visualaze()
    exit()