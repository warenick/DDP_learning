from math import dist
import torch
from torch.functional import Tensor

class Agent():
    def __init__(self, 
                 initial_state=torch.tensor([0., 0., 0.]),  # [x, y, yaw, dv, dyaw]
                 goal=torch.tensor([10, 10, 0]),
                 type="agent",
                 kinematic_type="differencial",
                 dt=0.4,
                 umax=[2.0, 1.0],
                 horizon = 10,
                 name="Bert",
                 costmap=None):
        self.costmap = costmap
        self.name = name
        self.state = initial_state if isinstance(initial_state, torch.Tensor) else torch.tensor(initial_state)
        self.goal = goal if isinstance(goal, torch.Tensor) else torch.tensor(goal)
        self.type = type
        self.dt = dt
        self.kinematic_type = kinematic_type
        self.umax = umax
        self.prediction = {}
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.init_aux()
        self.horizon = horizon
        # calc nominal trajectory(->agent.prediction)  # u =  # [[V,Vyaw],[V,Vyaw],...]
        initial_controll = self.generate_linear_controll(self.horizon, self.state, self.goal)# torch.rand((horizon,2))
        self.calc_trajectory(initial_controll) 
        # self.update_history()

    def generate_linear_controll(self, steps=None, state=None, goal=None):
        steps = steps if steps is not None else self.horizon
        state = state if state is not None else self.state
        goal = goal if goal is not None else self.goal
        controll = torch.rand((steps, 2)) #torch.zeros((steps, 2))
        current_state = state.clone()
        
        # rotate to moving dir firstly
        angle_direction = torch.atan2(goal[1]-state[1],goal[0]-state[0])
        angle_diff = current_state[2]-angle_direction
        sign = angle_diff/torch.abs(angle_diff)
        curr_step = 0
        while  torch.linalg.norm(current_state[2] - angle_direction)>0.01 and curr_step<steps:
            angle_diff = current_state[2]-angle_direction
            move_angle = sign*angle_diff#%self.pi
            controll[curr_step] = torch.clip(torch.tensor([0., move_angle]),-self.umax[1],self.umax[1])
            current_state = self.step_func(current_state, controll[curr_step])
            curr_step+=1
        # move forvard next
        while curr_step<steps:
            dist = torch.linalg.norm(current_state[:2]-goal[:2])
            controll[curr_step] = torch.clip(torch.tensor([dist,0.]),-self.umax[0],self.umax[0])
            current_state = self.step_func(current_state, controll[curr_step])
            curr_step+=1
        return controll
        # angle =         

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
            V  = torch.clamp((u[0]), -self.umax[0], self.umax[0]) # Linear velocity
            Vr = torch.clamp((u[1]), -self.umax[1], self.umax[1])+1e-6 # Angular velocity of robot
            # Radius of trajectory
            R = V/(Vr)
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

    def step(self, controll = None):
        controll = controll if controll is not None else self.prediction["controll"][0]
        self.state = self.step_func(self.state, controll)
        # slide the window
        self.prediction["controll"] = torch.cat((
            self.prediction["controll"][1:],
            self.prediction["controll"][-1][None]
            ))
        self.prediction["state"] = torch.cat((
            self.state[None],
            self.prediction["state"][2:],
            self.step_func(self.prediction["state"][-1], self.prediction["controll"][-1])[None]
            ))
        
        return self.state
    #     # action = u[v,yaw]
#############################################################
########################### costs ###########################
#############################################################
    def costmap_cost(self, state):
        return torch.tensor(self.costmap.at_position(state))

    def social_cost(self, state, others):
        distances = torch.sum(1./torch.linalg.norm(state[:2]-others[:,:2],dim=1))**2
        return distances

    def final_cost(self,state, k_yaw=torch.tensor(0.1)):
        # state[x,y,yaw]
        evclidian_dist = torch.linalg.norm(state[:2]-self.goal[:2])**2 + torch.linalg.norm(state[:2]-self.goal[:2])
        # evclidian_dist = torch.linalg.norm(state[:2]-self.goal[:2])
        # evclidian_dist = evclidian_dist if evclidian_dist>0.3 else evclidian_dist*0.3 # dirty fix oscilation near the goal
        # angle_dist = (state[2]-self.goal[2])%self.pi*k_yaw
        return evclidian_dist
        # return evclidian_dist+angle_dist

    def running_cost(self, state, controll, others=None, k_state=1.):
        state_cost = self.final_cost(state)*k_state
        controll_cost = torch.sum(torch.pow(controll, 2))
        social_cost = self.social_cost(state, others) if others is not None else torch.tensor(0.)
        costmap_cost = self.costmap_cost(state) if self.costmap is not None else torch.tensor(0.)
        # if costmap_cost !=0:
        #     print("here")
        # print("state_cost",state_cost)
        # print("controll_cost",controll_cost)
        return state_cost+controll_cost+social_cost+controll_cost*costmap_cost
#############################################################
########################### costs ###########################
#############################################################

    def l_x_l_u(self, state, controll, others = None):
        # [3] Gradient over state, [3] Gradient over controll
        # x = state.clone().detach().requires_grad_(True)
        x = state.clone().detach().requires_grad_(True)
        u = controll.clone().detach().requires_grad_(True)
        # torch.autograd.gradcheck(self.running_cost,(x,u))
        y = self.running_cost(x,u,others)
        y.backward()
        return x.grad, u.grad

    def l_x(self, state, controll, others = None):
        # [3] Gradient over state
        # x = state.clone().detach().requires_grad_(True)
        x = state.clone().detach().requires_grad_(True)
        u = controll.clone().detach()
        # torch.autograd.gradcheck(self.running_cost,(x,u))
        y = self.running_cost(x,u,others)
        y.backward()
        return x.grad

    def l_u(self, state, controll, others = None):
        # [2] Gradient over controll
        u = controll.clone().detach().requires_grad_(True)
        x = state.clone().detach()
        y = self.running_cost(x,u,others)
        y.backward()
        return u.grad

    def lf_x(self,state):
        # [3] Gradient over state
        x = state.clone().detach().requires_grad_(True)
        y = self.final_cost(x)
        y.backward()
        return x.grad

    def f_x_f_u(self,state,controll):
        # [3,3] Jacobian over state, [3,3] Jacobian over controll
        out_jacobian = torch.autograd.functional.jacobian(self.step_func,(state,controll)) # [0] - over state, [1] - over controll
        return out_jacobian[0], out_jacobian[1] # [0] -> over state

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

    def l_xx_l_uu_l_ux(self, state, controll, others = None):
        # [3,3] Hessian over state, [2,2] Hessian over controll, [2,3] Hessian over controll and state
        if others is not None:
            hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll,others)) # [[xx,xy],[yx,yy]]
        else:
            hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xy],[yx,yy]]
        return hessian_out[0][0], hessian_out[1][1], hessian_out[1][0] # [0][0] -> x,x

    def l_xx(self, state, controll, others = None):
        # [3,3] Hessian over state
        if others is not None:
            hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll,others))
        else:
            hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xy],[yx,yy]]
        return hessian_out[0][0] # [0][0] -> x,x

    def l_uu(self, state, controll, others = None):
        # [2,2] Hessian over controll
        if others is not None:
            hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll,others))
        else:
            hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll)) # [[xx,xu],[ux,uu]]

        return hessian_out[1][1] # [1][1] -> u,u

    def l_ux(self, state, controll, others = None):
        # [2,3] Hessian over controll and state
        if others is not None:
            hessian_out = torch.autograd.functional.hessian(self.running_cost,(state,controll,others))
        else:
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


# some debug checkers
if __name__=="__main__":
    from pprint import pprint
    goal = [10.,10.,0.]
    dt = 1.0
    ag =Agent(goal=goal,dt=dt)

        # print(torch.autograd.gradcheck(ag.l_x,(state,controll)))
    print("----------------initial---------------------------")
    print("state ",ag.state)
    print("goal ",goal)
    print("dt ",dt)
    print("state prediction ")
    pprint(ag.prediction["state"])
    print("controll prediction ")
    pprint(ag.prediction["controll"])
    controll = ag.prediction["controll"][0]
    print("----------------dynamics--------------------------")
    print("step_func ",ag.step_func(ag.state,controll))
    print("running_cost ",ag.running_cost(ag.state,controll))
    print("final_cost ",ag.final_cost(ag.state))
    print("step ",ag.step())
    with torch.autograd.detect_anomaly():
        print("----------------gradients---------------------")
        print("l_x ")
        pprint(ag.l_x(ag.state,controll))
        print("l_u ")
        pprint(ag.l_u(ag.state,controll))
        print("l_x_l_u")
        pprint(ag.l_x_l_u(ag.state,controll))
        print("l_xx ")
        pprint(ag.l_xx(ag.state,controll))
        print("l_uu ")
        pprint(ag.l_uu(ag.state,controll))
        print("l_ux ")
        pprint(ag.l_ux(ag.state,controll))
        print("l_xx_l_uu_l_ux ")
        pprint(ag.l_xx_l_uu_l_ux(ag.state,controll))
        print("lf_x ")
        pprint(ag.lf_x(ag.state))
        print("lf_xx ")
        pprint(ag.lf_xx(ag.state))
        print("f_x ")
        pprint(ag.f_x(ag.state,controll))
        print("f_u ")
        pprint(ag.f_u(ag.state,controll))
        print("f_x_f_u ")
        pprint(ag.f_x_f_u(ag.state,controll))
        print("f_uu ")
        pprint(ag.f_uu(ag.state,controll))
        print("f_xx ")
        pprint(ag.f_xx(ag.state,controll))
        print("f_ux ")
        pprint(ag.f_ux(ag.state,controll))
        # print("f_xx_f_uu_f_ux ",ag.f_xx_f_uu_f_ux(state,controll)) # TODO
        
    # torch.autograd.gradcheck
    