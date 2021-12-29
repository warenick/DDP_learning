default = {
    "dt":0.4, # seconds for step
    "initial_state":[0., 0., 0.],  # [x,y,yaw]
    "umax":[2.0, 1.0], # [Vmax,Vyawmax]
    "kinematic_type":"differencial",
    "type":"agent",
    "horizon": 6, # steps to be predicted
    "optimizer":{
        "type":"costmap-ddp", # options: linear, ddp, social-ddp, costmap-ddp, social-costmap-ddp
        "gradient_rate":0.25,
        "regularisation":0.91,
    }
}
agents = [
    {
    "name":"1",
    "initial_state":[3.0, 1.0, 0.0],
    "goal":         [-4.0, 3.0, 0.0],
    },
    {
    "name":"2",
    "initial_state":[-4.0, 3.0, 0.0],
    "goal":         [3.0, 1.0, 0.0],
    },
    {
    "name":"3",
    "initial_state":[3.0, -1.5, 0.0],
    "goal":         [-3.0, -1.5, -0.0],
    },
    {
    "name":"4",
    "initial_state":[-3.0, -1.0, 0.0],
    "goal":         [3.0, -1.0, 0.0],
    },
]