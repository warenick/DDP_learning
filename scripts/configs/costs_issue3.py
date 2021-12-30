default = {
    "dt":0.4,
    "initial_state":[0., 0., 0.], 
    "umax":[2.0, 1.0],
    "kinematic_type":"differencial",
    "type":"agent",
    "horizon": 10,
    "optimizer":{
        "type":"ddp",
        "gradient_rate":0.6,
        "regularisation":0.95,
    }
}
agents = [
    {
    "name":"0",
    "dt":0.1,
    "initial_state":[-4., 0., 1.57],
    "goal":[-5, -3, 0],
    },
    {
    "name":"1",
    "dt":0.2,
    "initial_state":[-2., 0., 1.57],
    "goal":[-3, -3, 0],
    },
    {
    "name":"2",
    "dt":0.4,
    "initial_state":[0., 0., 1.57],
    "goal":[-1, -3, 0],
    },
    {
    "name":"3",
    "dt":0.6,
    "initial_state":[2, 0, 1.57],
    "goal":[1, -3, 0],
    },
    {
    "name":"4",
    "dt":0.8,
    "initial_state":[4, 0, 1.57],
    "goal":[3, -3, 0],
    },
]