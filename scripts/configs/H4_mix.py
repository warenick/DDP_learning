default = {
    "dt":0.20, # seconds for step
    "initial_state":[0., 0., 0.],  # [x,y,yaw]
    "umax":[1.0, 0.5], # [Vmax,Vyawmax]
    "kinematic_type":"differencial",
    "type":"agent",
    "horizon": 10, # steps to be predicted
    "optimizer":{
        "type":"social-ddp", # options: linear, ddp, social-ddp, costmap-ddp, social-costmap-ddp
        "gradient_rate":0.50,
        "regularisation":0.80,
    }
}
agents = [
    {
    "name":"1",
    "initial_state":[3.0, 1.5, 0.0],
    "goal":         [-3.0, 1.5, 0.0],
    "optimizer":{
        "type":"social-ddp",
    }
    },
    {
    "name":"2",
    "initial_state":[-3.0, 1.0, 0.0],
    "goal":         [3.0, 1.0, 0.0],
    "optimizer":{
        "type":"linear",
    }
    },
    {
    "name":"3",
    "initial_state":[3.0, -1.5, 0.0],
    "goal":         [-3.0, -1.5, -0.0],
    "optimizer":{
        "type":"social-ddp",
    }
    },
    {
    "name":"4",
    "initial_state":[-3.0, -1.0, 0.0],
    "goal":         [3.0, -1.0, 0.0],
    "optimizer":{
        "type":"linear", 
    }
    },
]