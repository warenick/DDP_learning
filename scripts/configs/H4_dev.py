default = {
    "dt":0.40, # seconds for step
    "initial_state":[0., 0., 0.],  # [x,y,yaw]
    "umax":[2.0, 1.0], # [Vmax,Vyawmax]
    "kinematic_type":"differencial",
    "type": "agent",
    "horizon": 10, # steps to be predicted
    "optimizer":{
        "type":"costmap-social-ddp", # options: linear, ddp, social-ddp, costmap-ddp, social-costmap-ddp
        "gradient_rate":0.4,
        "regularisation":0.90,
    }
}
agents = [
    # {
    # "name":"1",
    # "initial_state":[2.0, 5., 0.0],
    # "goal":         [-5.0, 2.5, 0.0],
    # },
    {
    "name":"2",
    "initial_state":[-10.0, 0.5, 0.0],
    "goal":         [5.0, 1.0, 0.0],
    }
]