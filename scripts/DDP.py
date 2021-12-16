import torch

class DDP:
    # ddp with small diferences from cddp article http://ieeexplore.ieee.org/document/7989086/
    def __init__(self, gradient_rate = 1., regularisation=0.95) -> None:
        self.gradient_rate = gradient_rate
        self.regularisation = regularisation

    def optimize(self, agent, num_epochs=1,visualizer=None):
        states = agent.prediction["state"]      #.clone().detach() # not sure that clone().detach() is necessary
        controlls = agent.prediction["controll"]#.clone().detach() # not sure that clone().detach() is necessary
        for _ in range(num_epochs):
            k_seq, kk_seq     = self.backward(states, controlls, agent)
            states, controlls = self.forward(states, controlls, k_seq, kk_seq, agent.step_func)
            if visualizer is not None:
                visualizer.pub_agent_state([agent]) # just vis in rviz
        return states, controlls
    
    def backward(self, x_seq, u_seq, agent):
        pred_time = len(u_seq)
        state_dim = x_seq.shape[-1]
        # that definition here just for readability
        b = [torch.zeros(state_dim)+1e-10 for _ in range(pred_time + 1)]
        A = [torch.zeros((state_dim, state_dim))+1e-10 for _ in range(pred_time + 1)]
        lf_x  = agent.lf_x
        lf_xx = agent.lf_xx
        l_x   = agent.l_x
        l_u   = agent.l_u
        l_xx  = agent.l_xx
        l_uu  = agent.l_uu
        l_ux  = agent.l_ux
        f_x   = agent.f_x
        f_u   = agent.f_u
        f_xx  = agent.f_xx
        # f_uu  = agent.f_uu
        # f_ux  = agent.f_ux
        # that definition here just for readability
        

        b[-1] = lf_x(x_seq[-1])
        A[-1] = lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(pred_time - 1, -1, -1):
            x,u = (x_seq[t], u_seq[t]) # state and controll at the current time step
            f_x_t = f_x(x,u)
            f_u_t = f_u(x,u)
            Q_x = l_x(x,u) + f_x_t.T @ b[t+1]
            Q_u = l_u(x,u) + f_u_t.T @ b[t+1]
            Q_xx = l_xx(x,u) + f_x_t.T @ A[t+1] @ f_x_t + b[t+1] @ f_xx(x,u)
            Q_ux = l_ux(x,u) + f_u_t.T @ A[t+1] @ f_x_t# + b[t+1] @ f_ux(x,u)
            Q_uu = l_uu(x,u) + f_u_t.T @ A[t+1] @ f_u_t# + b[t+1] @ f_uu(x,u)
            try:
                inv_Q_uu = torch.linalg.inv(Q_uu)
            except:
                inv_Q_uu = torch.zeros_like(Q_uu)
            k = -inv_Q_uu @ Q_u
            kk = -inv_Q_uu @ Q_ux
            A[t] = Q_xx + kk.T@Q_uu@kk + Q_ux.T@kk + kk.T@Q_ux
            b[t] = Q_x + kk.T@Q_uu@k + Q_ux.T@k + kk.T@Q_u
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq, step_func, umax = [1.,1.]):
        x_seq_hat = x_seq#.clone().detach() # not sure that clone().detach() is necessary
        u_seq_hat = u_seq#.clone().detach() # not sure that clone().detach() is necessary
        for t in range(len(u_seq)):
            control = u_seq[t] + (kk_seq[t] @ (x_seq_hat[t] - x_seq[t]) + k_seq[t])*self.gradient_rate
            u_seq_hat[t] =torch.clamp(control, -umax[-1], umax[-1]) # it is necessary to clamp here
            x_seq_hat[t + 1] = step_func(x_seq_hat[t], u_seq_hat[t])
        # regularisation gradient rate
        self.gradient_rate = self.gradient_rate*self.regularisation
        return x_seq_hat, u_seq_hat
