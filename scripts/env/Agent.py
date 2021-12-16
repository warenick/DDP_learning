import torch
from torch.functional import Tensor

class Agent():
    def __init__(self, initial_state=torch.tensor([0., 0., 0.]),  # [x, y, yaw, dv, dyaw]
                 goal=torch.tensor([10, 10, 0]),
                 type="agent",
                 kinematic_type="differencial",
                 dt=1.0,
                 umax=[2.0, 1.0]):

        self.state = initial_state if isinstance(initial_state, torch.Tensor) else torch.tensor(initial_state)
        self.goal = goal if isinstance(goal, torch.Tensor) else torch.tensor(goal)
        self.type = type
        self.dt = dt
        self.kinematic_type = kinematic_type
        self.umax = umax
        self.prediction = {}
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.init_aux()
        # self.update_history()


    def init_aux(self):
        self.aux1 = torch.eye(3)
        self.aux1[2,2]=0
        # [[1., 0., 0.],
        #  [0., 1., 0.],
        #  [0., 0., 0.]]

        self.aux2 = torch.zeros((3,3))
        self.aux2[1,0] = 1
        self.aux2[0,1] = -1
        # [[ 0.,-1., 0.],
        #  [ 1., 0., 0.],
        #  [ 0., 0., 0.]]
        
        self.aux3 = torch.zeros((3,3))
        self.aux3[2,2] = 1
        # [[0., 0., 0.],
        #  [0., 0., 0.],
        #  [0., 0., 1.]]
        self.aux100 = torch.Tensor([1,0,0])
        self.aux010 = torch.Tensor([0,1,0])
        self.aux001 = torch.Tensor([0,0,1])
        # self.aux11100 = torch.Tensor([1,1,1,0,0])
        # self.aux00011 = torch.Tensor([0,0,0,1,1])
        pass
    
    def calc_trajectory(self, controll_arr):
        controll_arr = controll_arr if isinstance(controll_arr, torch.Tensor) else torch.tensor(controll_arr)
        self.prediction["state"] = torch.zeros((controll_arr.shape[0]+1, self.state.shape[0]))
        self.prediction["controll"] = controll_arr
        # generate nominal trajectory
        self.prediction["state"][0] = self.state#.clone().detach()
        for t in range(self.prediction["controll"].shape[0]):
            self.prediction["state"][t+1] = self.step_func(self.prediction["state"][t], self.prediction["controll"][t])

    def step_func(self, x, u):
        if "differencial" in self.kinematic_type:
            # u = [v,vyaw]
            V = torch.clamp((u[0]), -self.umax[0], self.umax[0]) # Linear velocity
            Vr = torch.clamp((u[1]), -self.umax[1], self.umax[1]) # Angular velocity of robot
            # Radius of rotation
            R = V/(Vr+1e-10)
            # Instantaneous Center of Curvature
            icc = x[:2]+R*(
                torch.sin(x[2])*torch.Tensor([-1,0])+
                torch.cos(x[2])*torch.Tensor([0,1]))
            matrix = torch.cos(Vr*self.dt)*self.aux1 + \
                     torch.sin(Vr*self.dt)*self.aux2 + \
                     self.aux3
            controll = x[:3] - \
                            (icc[0]*self.aux100 + \
                             icc[1]*self.aux010)# torch.cat((x[0]-iccx,x[1]-iccy,x[2]))
            pose = matrix @ controll + \
                (icc[0]*self.aux100 + \
                 icc[1]*self.aux010 + \
                 Vr*self.dt*self.aux001)
        return pose
        

    def step_func2(self, x, u):
        if "differencial" in self.kinematic_type:
            # u = [0,0,0,v,vyaw]
            # controll = self.dt*torch.clamp((u[3:]+x[3:]), -self.umax[-1], self.umax[-1])
            # controll = torch.tensor([torch.clamp((u[3]+x[3]), -vmax, vmax),
            #                         torch.clamp((u[4]+x[4]), -vyawmax, vyawmax)])
            controll = torch.tensor([1,0])* torch.clamp(u[3], -self.umax[3], self.umax[3]) + torch.tensor([0,1])*torch.clamp(u[4], -self.umax[4], self.umax[4])
            # controll = torch.clamp((u[3:]), -self.umax[-1], self.umax[-1])
            aux_sin = torch.zeros((3,2))
            aux_sin[1,0] = 1.
            aux_cos = torch.zeros((3,2))
            aux_cos[0,0] = 1.
            aux_aux = torch.zeros((3,2))
            aux_aux[-1,-1] = 1.
            matrix =  torch.cos(x[2]*self.dt)*aux_cos+torch.sin(x[2]*self.dt)*aux_sin+aux_aux
            # matrix = torch.tensor([[torch.cos(x[2]*self.dt), 0],
            #                        [torch.sin(x[2]*self.dt), 0],
            #                        [0,                       1]])
            pose = x[:3] + matrix@controll
        # return torch.cat((pose,controll)) 
        return torch.cat((pose,controll)) 
        # return (pose,controll) 

    # def step(self, controll):
    #     self.state = self.step_func(self.state, controll)
    #     # self.prediction["state"].append(np.copy(self.state))
    #     return self.state
    #     # action = u[0,0,0,v,yaw]

    def final_cost(self,state, k_yaw=torch.tensor(0.1)):
        # state[x,y,yaw]
        evclidian_dist = torch.linalg.norm(state[:2]-self.goal[:2])
        # angle_dist = (state[2]-self.goal[2])%self.pi*k_yaw
        return evclidian_dist
        # return evclidian_dist+angle_dist

    def running_cost(self, state, controll, k_state=1.):
        state_cost = self.final_cost(state)*k_state
        controll_cost = torch.sum(torch.pow(controll, 2))
        # controll_cost = torch.sum(torch.abs(controll))
        # pred = self.final_cost(state)
        # next = self.final_cost(self.step_func(state,controll))

        # controll_cost = (pred - next)*torch.sum(torch.pow(controll,2))*k_state
        # print("state_cost",state_cost)
        # print("controll_cost",controll_cost)
        return state_cost+controll_cost
    
    def l_x(self, state, controll):
        # [3] Gradient over state
        # x = state.clone().detach().requires_grad_(True)
        x = state.clone().detach().requires_grad_(True)
        u = controll.clone().detach()
        # fuckfuckfuckfuckfuckfuckfuck
        # torch.autograd.gradcheck(self.running_cost,(x,u))
        y = self.running_cost(x,u)
        y.backward()
        return x.grad

    def l_u(self, state, controll):
        # [2] Gradient over controll
        u = controll.clone().detach().requires_grad_(True)
        x = state.clone().detach()
        y = self.running_cost(x,u)
        y.backward()
        return u.grad

    def lf_x(self,state):
        # [3] Gradient over state
        x = state.clone().detach().requires_grad_(True)
        y = self.final_cost(x)
        y.backward()
        return x.grad

    def f_x(self,state,controll):
        # [3,3] Jacobian over state
        out_jacobian = torch.autograd.functional.jacobian(self.step_func,(state,controll)) # [0] - over state, [1] - over controll
        return out_jacobian[0] # [0] -> over state

    def f_u(self,state,controll):
        # [3,2] Jacobian over constroll
        out_jacobian = torch.autograd.functional.jacobian(self.step_func,(state,controll)) # [0] - over state, [1] - over controll
        return out_jacobian[1] # [1] -> over controll

    def lf_xx(self,state):
        # [3,3] Hessian over state
        return torch.autograd.functional.hessian(self.final_cost,state)

    def l_xx(self, state, controll):
        # [3,3] Hessian over state
        hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xy],[yx,yy]]
        return hessian_out[0][0] # [0][0] -> x,x

    def l_uu(self, state, controll):
        # [2,2] Hessian over controll
        hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xu],[ux,uu]]
        return hessian_out[1][1] # [1][1] -> u,u
    def l_ux(self, state, controll):
        # [2,3] Hessian over controll and state
        hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xy],[yx,yy]]
        return hessian_out[1][0]# [1][0] -> u,x

    def f_xx(self, state, controll):
        # [3,3,3]
        inputs = (state.clone().detach().requires_grad_(True),controll.clone().detach().requires_grad_(True))
        return self.second_derivative_of_vector_func(self.step_func, inputs, wrt=[0,0])        

    def f_uu(self, state, controll):
        # [3,2,2]
        inputs = (state.clone().detach().requires_grad_(True),controll.clone().detach().requires_grad_(True))
        return self.second_derivative_of_vector_func(self.step_func, inputs, wrt=[1,1])
    
    def func2(self,x,y):
        return (x**3)*y**3

    def f_ux(self, state, controll):        
        # [3,2,3]
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
            last_shape = inputs[wrt[1]].shape[0]
        else:
            last_shape = inputs.shape[0]
        second = torch.zeros((jacobian.shape[0], jacobian.shape[1], last_shape))
        if jacobian.requires_grad:
            for x in range(jacobian.shape[0]):
                for y in range(jacobian.shape[1]):
                    second[x,y] = torch.autograd.grad(jacobian[x,y], inputs, create_graph=True, allow_unused=True)[wrt[1]]
        return second.clone().detach()


# tests
if __name__=="__main__":
    goal = [10.,10.,0.]
    dt = 1.0
    ag =Agent(goal=goal,dt=dt)
    state = torch.tensor([0.,0.,0.])
    controll = torch.tensor([1.,1.])
    with torch.autograd.detect_anomaly():
        # print(torch.autograd.gradcheck(ag.l_x,(state,controll)))
        print("----------------initial---------------------------")
        print("state ",state)
        print("controll ",controll)
        print("goal ",goal)
        print("dt ",dt)
        print("----------------dynamics--------------------------")
        print("step_func ",ag.step_func(state,controll))
        print("running_cost ",ag.running_cost(state,controll))
        print("final_cost ",ag.final_cost(state))
        print("----------------differencials---------------------")
        print("l_x ",ag.l_x(state,controll))
        print("l_u ",ag.l_u(state,controll))
        print("l_xx ",ag.l_xx(state,controll))
        print("l_uu ",ag.l_uu(state,controll))
        print("l_ux ",ag.l_ux(state,controll))
        print("lf_x ",ag.lf_x(state))
        print("lf_xx ",ag.lf_xx(state))
        print("f_x ",ag.f_x(state,controll))
        print("f_u ",ag.f_u(state,controll))
        print("f_uu ",ag.f_uu(state,controll))
        print("f_xx ",ag.f_xx(state,controll))
        print("f_ux ",ag.f_ux(state,controll))
    # torch.autograd.gradcheck
    