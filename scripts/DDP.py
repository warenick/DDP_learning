import torch

class DDP:
    # ddp with small diferences from cddp article http://ieeexplore.ieee.org/document/7989086/
    def __init__(self, agent, gradient_rate = 1., regularisation=0.95) -> None:
        self.gradient_rate = gradient_rate
        self.regularisation = regularisation
        self.agent = agent

    def backward(self, x_seq, u_seq):
        pred_time = len(u_seq)
        state_dim = x_seq.shape[-1]
        # that definition here just for readability
        b = [torch.zeros(state_dim)+1e-10 for _ in range(pred_time + 1)]
        A = [torch.zeros((state_dim, state_dim))+1e-10 for _ in range(pred_time + 1)]
        lf_x  = self.agent.lf_x
        lf_xx = self.agent.lf_xx
        l_x   = self.agent.l_x
        l_u   = self.agent.l_u
        l_xx  = self.agent.l_xx
        l_uu  = self.agent.l_uu
        l_ux  = self.agent.l_ux
        f_x   = self.agent.f_x
        f_u   = self.agent.f_u
        f_xx  = self.agent.f_xx
        # f_uu  = self.agent.f_uu
        # f_ux  = self.agent.f_ux
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

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = x_seq#.clone().detach()
        u_seq_hat = u_seq#.clone().detach()
        for t in range(len(u_seq)):
            control = u_seq[t] + (kk_seq[t] @ (x_seq_hat[t] - x_seq[t]) + k_seq[t])*self.gradient_rate
            u_seq_hat[t] =torch.clamp(control, -self.agent.umax[-1], self.agent.umax[-1])
            x_seq_hat[t + 1] = self.agent.step_func(x_seq_hat[t], u_seq_hat[t])
        # regularisation gradient rate decrising
        self.gradient_rate = self.gradient_rate*self.regularisation
        return x_seq_hat, u_seq_hat
