class Linear:
    def __init__(self, type="linear") -> None:
        self.type = type

    def optimize(self, agent, agents = None, num_epochs=None, visualizer=None):
        horizon = len(agent.prediction["controll"])
        controll = agent.generate_linear_controll(horizon, agent.state, agent.goal)
        agent.calc_trajectory(controll) 
        if visualizer is not None:
            visualizer.pub_agent_state([agent])
        return agent.prediction["state"], agent.prediction["controll"]