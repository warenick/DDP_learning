class Linear:
    def __init__(self, type="linear") -> None:
        self.type = type
        self.initial_gradient_rate = 1.0
        self.gradient_rate = self.initial_gradient_rate
        self.regularisation = 0.95

    def optimize(self, agent, agents = None, num_epochs=None, visualizer=None):
        horizon = len(agent.prediction["controll"])
        controll = agent.generate_linear_controll(horizon, agent.state, agent.goal)
        agent.calc_trajectory(controll) 
        if visualizer is not None:
            visualizer.pub_agent_state([agent])
        return agent.prediction["state"], agent.prediction["controll"]