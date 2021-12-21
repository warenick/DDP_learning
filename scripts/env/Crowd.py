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
            if "ddp" in opt["type"] or "DDP" in opt["type"]: 
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
        for (agent, optimizer) in zip(self.agents, self.optimizers):
            if gradient_rate is not None:
                optimizer.initial_gradient_rate = gradient_rate
            if regularisation is not None:
                optimizer.regularisation = regularisation
            if visualize:
                agent.prediction["state"], agent.prediction["controll"] = optimizer.optimize(agent, epochs, self.viz) # optimize trajectory
            else:
                agent.prediction["state"], agent.prediction["controll"] = optimizer.optimize(agent, epochs) # optimize trajectory

    def step(self):
        # TODO: parallel it
        for agent in self.agents:
            agent.step()

    def visualaze(self):
        self.viz.pub_agent_state(self.agents)
