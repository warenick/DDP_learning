import torch

class DDP:
    def __init__(self, agent, gradient_rate = 1) -> None:
        self.state_dim = agent.state.shape[-1]
        self.gradient_rate = gradient_rate
        pred_time = len(agent.history["state"]) - 1
        self.umax = agent.umax
        self.dt = agent.dt
        # self.v = [1e-10 for _ in range(pred_time + 1)]
        self.v_x = [torch.zeros(self.state_dim)+1e-10 for _ in range(pred_time + 1)]
        self.v_xx = [torch.zeros((self.state_dim, self.state_dim))+1e-10 for _ in range(pred_time + 1)]
        self.f = agent.step_func
        # self.lf = agent.final_cost
        self.lf_x = agent.lf_x
        self.lf_xx =agent.lf_xx
        self.l_x =  agent.l_x
        self.l_u =  agent.l_u
        self.l_xx = agent.l_xx
        self.l_uu = agent.l_uu
        self.l_ux = agent.l_ux
        self.f_x =  agent.f_x
        self.f_u =  agent.f_u
        self.f_xx = agent.f_xx
        self.f_uu = agent.f_uu
        self.f_ux = agent.f_ux

    def backward(self, x_seq, u_seq):
        pred_time = len(u_seq)
        # that definition here just for readability
        # v = [1e-10 for _ in range(pred_time + 1)]
        v_x = [torch.zeros(self.state_dim)+1e-10 for _ in range(pred_time + 1)]
        v_xx = [torch.zeros((self.state_dim, self.state_dim))+1e-10 for _ in range(pred_time + 1)]
        # lf = self.lf
        lf_x = self.lf_x
        lf_xx = self.lf_xx
        l_x = self.l_x
        l_u = self.l_u
        l_xx = self.l_xx
        l_uu = self.l_uu
        l_ux = self.l_ux
        f_x = self.f_x
        f_u = self.f_u
        f_xx = self.f_xx
        f_uu = self.f_uu
        f_ux = self.f_ux
        # that definition here just for readability
        

        # v[-1] = lf(x_seq[-1])
        v_x[-1] = lf_x(x_seq[-1])
        v_xx[-1] = lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(pred_time - 1, -1, -1):
            x,u = (x_seq[t], u_seq[t]) # state and controll at the current time step
            f_x_t = f_x(x,u)
            f_u_t = f_u(x,u)
            Q_x = l_x(x,u) + f_x_t.T @ v_x[t+1]
            Q_u = l_u(x,u) + f_u_t.T @ v_x[t+1]
            Q_xx = l_xx(x,u) + f_x_t.T @ v_xx[t+1] @ f_x_t + v_x[t+1] @ f_xx(x,u)
            Q_uu = l_uu(x,u) + f_u_t.T @ v_xx[t+1] @ f_u_t + v_x[t+1] @ f_uu(x,u)
            Q_ux = l_ux(x,u) + f_u_t.T @ v_xx[t+1] @ f_x_t + v_x[t+1] @ f_ux(x,u)
            inv_Q_uu = torch.linalg.inv(Q_uu)
            k = -inv_Q_uu @ Q_u
            kk = -inv_Q_uu @ Q_ux
            # dv = 0.5 * Q_u @ k
            # v[t] += dv
            # v_x[t] = Q_x - Q_u @ inv_Q_uu @ Q_ux
            v_x[t] = Q_x - Q_u @ inv_Q_uu @ Q_ux
            v_xx[t] = Q_xx + Q_ux.T @ kk
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = x_seq.clone().detach()
        u_seq_hat = u_seq.clone().detach()
        for t in range(len(u_seq)):
            control = u_seq[t] + (kk_seq[t] @ (x_seq_hat[t] - x_seq[t]) + k_seq[t])*self.gradient_rate
            u_seq_hat[t] =torch.clamp(control, -self.umax[-1], self.umax[-1])
            # u_seq_hat[t] = u_seq[t] + control
            # u_seq_hat[t] = control
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat
