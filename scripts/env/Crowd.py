

class Crowd:
    
    def __init__(self, visualizer = None) -> None:
        self.viz = visualizer
        self.agents = []
        self.optimizers = []
    
    def add_agent(self, agent, optimizer):
        self.agents.append(agent)
        self.optimizers.append(optimizer)

    def optimize(self, epochs, visualize = False):
        for (agent, optimizer) in zip(self.agents, self.optimizers):
            if visualize:
                agent.prediction["state"], agent.prediction["controll"] = optimizer.optimize(agent, epochs, self.viz) # optimize trajectory
            else:
                agent.prediction["state"], agent.prediction["controll"] = optimizer.optimize(agent, epochs) # optimize trajectory

    def visualaze(self):
        self.viz.pub_agent_state(self.agents)
