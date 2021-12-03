# import numpy as np
# import autograd.numpy as np
from os import stat
import torch

class Agent():
    def __init__(self, initial_state=torch.tensor([0., 0., 0., 0., 0.]),  # [x, y, yaw, dv, dyaw]
                 goal=torch.tensor([10, 10, 0, 0, 0]),
                 type="agent",
                 kinematic_type="differencial",
                 dt=torch.tensor(0.1),
                 umax=torch.tensor([0, 0, 0, 5.0, 5.0])):
        self.state_initial = initial_state
        self.goal = goal
        self.type = type
        self.dt = dt
        self.kinematic_type = kinematic_type
        self.umax = umax
        self.history = {}
        self.state = self.state_initial.clone().detach()
        # self.update_history()
    
    def update_history(self, controll_arr = None):
        self.history["state"] = torch.zeros((controll_arr.shape[0]+1,controll_arr.shape[1]))
        self.history["state"][0] = self.state_initial.clone().detach()
        self.history["controll"] = torch.zeros_like(controll_arr)
        if controll_arr is not None:
            # generate nominal trajectory
            for t in range(controll_arr.shape[0]):
                new_state = self.step_func(self.history["state"][t], controll_arr[t])
                self.history["state"][t][3:] = new_state.clone().detach()[3:]
                self.history["state"][t+1][:3] = new_state.clone().detach()[:3]
                self.history["controll"][t] = controll_arr[t].clone().detach()

    def step_func(self, x, u):
        if "differencial" in self.kinematic_type:
            # u = [0,0,0,v,vyaw]
            pose = x[:3]
            # controll = self.dt*torch.clamp((u[3:]+x[3:]), -self.umax[-1], self.umax[-1])
            controll = self.dt*torch.clamp((u[3:]+x[3:]), -self.umax[-1], self.umax[-1])
            matrix = torch.tensor([[torch.cos(x[2]), 0],
                                   [torch.sin(x[2]), 0],
                                   [0,               1]])
            pose = pose + torch.matmul(matrix, controll)
        return torch.cat((pose,controll)) 
        # return (pose,controll) 

    # def step(self, controll):
    #     self.state = self.step_func(self.state, controll)
    #     # self.history["state"].append(np.copy(self.state))
    #     return self.state
    #     # action = u[0,0,0,v,yaw]

    def final_cost(self,state):
        # state[x,y,yaw]
        # dist = torch.linalg.norm(state[:3]-self.goal[:3])
        dist = torch.linalg.norm(state[:2]-self.goal[:2])
        dist_yaw = torch.linalg.norm(state[2]-self.goal[2])
        # return dist**3 + dist**2+dist
        return dist**3+dist**2+dist+dist_yaw

    def running_cost(self, state, controll):
        state_cost = self.final_cost(state)
        controll_cost = torch.sum(torch.square(controll))**3/10.
        # print("state_cost",state_cost)
        # print("controll_cost",controll_cost)
        return state_cost+controll_cost
    
    def l_x(self, state, controll):
        # [5] Gradient over state
        # x = state.clone().detach().requires_grad_(True)
        x = state.clone().detach().requires_grad_(True)
        y = self.running_cost(x,controll)
        y.backward()
        return x.grad

    def l_u(self, state, controll):
        # [5] Gradient over controll
        u = controll.clone().detach().requires_grad_(True)
        y = self.running_cost(state,u)
        y.backward()
        return u.grad

    def lf_x(self,state):
        # [5] Gradient over state
        x = state.clone().detach().requires_grad_(True)
        y = self.final_cost(x)
        y.backward()
        return x.grad

    def f_x(self,state,controll):
        # [5,5] Jacobian over state
        out_jacobian = torch.autograd.functional.jacobian(self.step_func,(state,controll)) # [0] - over state, [1] - over controll
        return out_jacobian[0] # [0] -> over state

    def f_u(self,state,controll):
        # [5,5] Jacobian over constroll
        out_jacobian = torch.autograd.functional.jacobian(self.step_func,(state,controll)) # [0] - over state, [1] - over controll
        return out_jacobian[1] # [1] -> over controll

    def lf_xx(self,state):
        # [5,5] Hessian over state
        return torch.autograd.functional.hessian(self.final_cost,state)

    def l_xx(self, state, controll):
        # [5,5] Hessian over state
        hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xy],[yx,yy]]
        return hessian_out[0][0] # [0][0] -> x,x

    def l_uu(self, state, controll):
        # [5,5] Hessian over controll
        hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xu],[ux,uu]]
        return hessian_out[1][1] # [1][1] -> u,u
    def l_ux(self, state, controll):
        # [5,5] Hessian over controll and state
        hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xy],[yx,yy]]
        return hessian_out[1][0]# [1][0] -> u,x

    def f_xx(self, state, controll):
        # [5,5,5]
        inputs = (state.clone().detach().requires_grad_(True),controll.clone().detach().requires_grad_(True))
        return self.second_derivative_of_vector_func(self.step_func, inputs, wrt=[0,0])        

    def f_uu(self, state, controll):
        # [5,5,5]
        inputs = (state.clone().detach().requires_grad_(True),controll.clone().detach().requires_grad_(True))
        return self.second_derivative_of_vector_func(self.step_func, inputs, wrt=[1,1])
    
    def func2(self,x,y):
        return (x**3)*y**3

    def f_ux(self, state, controll):        
        # [5,5,5]
        inputs = (state.clone().detach().requires_grad_(True),controll.clone().detach().requires_grad_(True))
        return self.second_derivative_of_vector_func(self.step_func, inputs, wrt=[1,0])
        # return self.second_derivative_of_vector_func(self.func2, inputs, wrt=[1,0])

    def second_derivative_of_vector_func(self, func, inputs, wrt=None):
        # func - func that takes vector input and return vector
        # inputs - vector input or cartage of vectors for func
        # wrt - if inputs are cartage, that wrt needs to choose the derivative option [[xx,xy],[yx,yy]]
        jacobian = torch.autograd.functional.jacobian(func, inputs, create_graph = True) # []
        # print(jacobian[0])
        if wrt is not None:
            jacobian = jacobian[wrt[0]]
            last_shape = inputs[wrt[0]].shape[0]
        else:
            last_shape = inputs.shape[0]
        second = torch.zeros((jacobian.shape[0], jacobian.shape[1], last_shape))
        if jacobian.requires_grad:
            for x in range(jacobian.shape[0]):
                for y in range(jacobian.shape[1]):
                    second[x,y] = torch.autograd.grad(jacobian[x,y], inputs, create_graph=True, allow_unused=True)[wrt[1]]
        return second


if __name__=="__main__":
    ag =Agent()
    state = torch.ones(5)
    controll = torch.ones(5)
    print("l_x ",ag.l_x(state,controll))
    print("l_u ",ag.l_u(state,controll))
    print("l_xx ",ag.l_xx(state,controll))
    print("l_uu ",ag.l_uu(state,controll))
    print("l_ux ",ag.l_ux(state,controll))
    print("lf_x ",ag.lf_x(state))
    print("lf_xx ",ag.lf_xx(state))
    print("f_x ",ag.f_x(state,controll))
    print("f_u ",ag.f_u(state,controll))
    print("f_ux ",ag.f_ux(state,controll))
    print("f_uu ",ag.f_uu(state,controll))
    print("f_xx ",ag.f_xx(state,controll))
