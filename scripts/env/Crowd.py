import torch
from env.Agent import Agent
from DDP import DDP
from Linear import Linear
from importlib import import_module
from pprint import pprint
class Crowd:
    
    def __init__(self, visualizer = None) -> None:
        self.viz = visualizer
        self.agents = []
        self.optimizers = []
        self.with_social = False
    
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
            if "social" in opt["type"]:
                self.with_social = True

            if "ddp" in opt["type"].lower(): # "ddp" or "social_ddp"
                new_optimizer = DDP(
                    gradient_rate = opt["gradient_rate"], 
                    regularisation=opt["regularisation"], 
                    type=opt["type"].lower()
                    )
            elif "linear" in opt["type"].lower(): # "linear"
                new_optimizer = Linear(type=opt["type"].lower())
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
        print("optimisation with socials: ", self.with_social)
        



    def add_agent(self, agent, optimizer):
        self.agents.append(agent)
        self.optimizers.append(optimizer)

    def optimize(self, epochs=1, visualize=False, gradient_rate=None, regularisation=0.91):
        viz = self.viz if visualize else None
        # update optimisation params
        for optimizer in self.optimizers:
            optimizer.initial_gradient_rate = gradient_rate if gradient_rate is not None else optimizer.initial_gradient_rate
            optimizer.regularisation = regularisation if regularisation is not None else optimizer.regularisation                


        # TODO: parallel it
        stacked_agents = None
        for _ in range(epochs):
            if self.with_social:
                stacked_agents = self.stack_agents_poses(self.agents)

            # for (agent, optimizer) in zip(self.agents, self.optimizers):
            for num in range(len(self.agents)):
                stacked_agents_exclude_one = stacked_agents[num] if self.with_social else None
                self.agents[num].prediction["state"], self.agents[num].prediction["controll"] = self.optimizers[num].optimize(agent = self.agents[num], agents=stacked_agents_exclude_one, num_epochs=1, visualizer=viz) # optimize trajectory
    
    def stack_agents_poses(self, agents=None): # without one to avoid self influance
        agents = agents if agents is not None else self.agents 
        num_agents = len(agents)
        num_steps = len(agents[0].prediction["state"]) # prediction horizon # TODO: warn about different prediction horizon for agents 
        state_dim = len(agents[0].prediction["state"][0])
        out = []
        
        for num in range(num_agents):
            stacked_agents_without_one = torch.zeros((num_agents-1, num_steps, state_dim))
            stupid_fix = 0
            for agent in range(num_agents):
                if agent is num:
                    stupid_fix=1
                    continue
                stacked_agents_without_one[agent-stupid_fix] = agents[agent].prediction["state"]
            changed_axes = torch.moveaxis(stacked_agents_without_one,1,0) # convert to (num_steps, num_agents, state_dim)
            out.append(changed_axes)
        return out

    def step(self):
        # TODO: parallel it
        for agent in self.agents:
            agent.step()

    def visualaze(self):
        self.viz.pub_agent_state(self.agents)
