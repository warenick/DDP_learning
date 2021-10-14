from env.Agent import Agent
from env.Visualizer_ros import Visualizer_ros
import numpy as np
from autograd import grad, jacobian
import autograd.numpy as np

class DDP:
    def __init__(self, agent) -> None:
        pred_time = len(agent.history["state"]) - 1
        self.umax = agent.umax
        self.dt = agent.dt
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.f = agent.step_func
        self.lf = agent.final_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)
        self.l_x = grad(agent.running_cost, 0)
        self.l_u = grad(agent.running_cost, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_uu = jacobian(self.f_u, 1)
        self.f_ux = jacobian(self.f_u, 0)

    def backward(self, x_seq, u_seq):
        pred_time = len(agent.history["state"]) - 1
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t]) + np.matmul(f_x_t.T, self.v_x[t + 1])
            q_u = self.l_u(x_seq[t], u_seq[t]) + np.matmul(f_u_t.T, self.v_x[t + 1])
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
              np.matmul(np.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_xx(x_seq[t], u_seq[t])))
            tmp = np.matmul(f_u_t.T, self.v_xx[t + 1])
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(tmp, f_u_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_uu(x_seq[t], u_seq[t])))
            q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.matmul(tmp, f_x_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_ux(x_seq[t], u_seq[t])))
            inv_q_uu = np.linalg.inv(q_uu)
            k = -np.matmul(inv_q_uu, q_u)
            kk = -np.matmul(inv_q_uu, q_ux)
            dv = 0.5 * np.matmul(q_u, k)
            self.v[t] += dv
            self.v_x[t] = q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[t] = q_xx + np.matmul(q_ux.T, kk)
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = self.dt*k_seq[t] + np.matmul(self.dt*kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            # control = 0.03*k_seq[t] + np.matmul(0.03*kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat

if __name__=="__main__":
    dt = 0.05
    horizon = 15
    epochs = 20

    viz = Visualizer_ros()
    agent = Agent(goal=np.array([3, 3, 0, 5., 0.])) # [x,y,yaw,V,Vyaw]
    u = np.zeros((horizon,agent.state.shape[-1])) # [[0,0,0,V,Vyaw],[0,0,0,V,Vyaw],...]
    u[:,-2]=2. # set initial V=2.
    state_dim = agent.state.shape[-1]
    agent.update_history(u) # calc nominal trajectory(->agent.history)
    viz.pub_agent_state([agent]) # just vis in rviz
    ddp = DDP(agent) # init differencial functions
    for epoch in range(epochs):
        k_seq, kk_seq = ddp.backward(agent.history["state"],agent.history["controll"])
        agent.history["state"], agent.history["controll"] = ddp.forward(agent.history["state"],agent.history["controll"],k_seq, kk_seq)
        agent.state = np.copy(agent.history["state"][-1])
        viz.pub_agent_state([agent]) # just vis in rviz


    exit()