from env.Agent import Agent
from DDP import DDP
from importlib import import_module
from pprint import pprint
class Crowd:
    
    def __init__(self, visualizer = None) -> None:
        self.viz = visualizer
        self.agents = []
        self.optimizers = []
    
    def read_from_conf(self, config = "configs.X4"):
        module = import_module(config)
        print("==========================")
        print(" read config: ", config)
        print("==========================")
        print("\n========default==========")
        pprint(module.default)
        print("\n=========agents==========")
        pprint(module.agents)
        
        for agent in module.agents:
            # setup default params
            for field in module.default:
                if field not in agent:
                    agent[field]=module.default[field]
            # create optimizer
            opt = agent["optimizer"]
            if "ddp" in opt["type"].lower(): # "ddp" or "social_ddp"
                new_optimizer = DDP(opt["gradient_rate"], opt["regularisation"])
            new_agent = Agent(
                agent["initial_state"],
                agent["goal"],
                agent["type"],
                agent["kinematic_type"],
                agent["dt"],
                agent["umax"],
                agent["horizon"],
                agent["name"],
            )
            self.add_agent(new_agent, new_optimizer)
        



    def add_agent(self, agent, optimizer):
        self.agents.append(agent)
        self.optimizers.append(optimizer)

    def optimize(self, epochs, visualize = False, gradient_rate=None, regularisation=0.91):
        # TODO: parallel it
        # if "social_ddp" in self.optimizers.lower():
        #     def optimize_
        for optimizer in self.optimizers:
            if gradient_rate is not None:
                optimizer.initial_gradient_rate = gradient_rate
            if regularisation is not None:
                optimizer.regularisation = regularisation
        viz = self.viz if visualize else None
        for _ in range(epochs):
            for (agent, optimizer) in zip(self.agents, self.optimizers):
                agent.prediction["state"], agent.prediction["controll"] = optimizer.optimize(agent, 1, viz) # optimize trajectory

    def step(self):
        # TODO: parallel it
        for agent in self.agents:
            agent.step()

    def visualaze(self):
        self.viz.pub_agent_state(self.agents)
