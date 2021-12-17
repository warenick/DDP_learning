default = {
    "dt":0.4,
    "initial_state":[0., 0., 0.], 
    "umax":[2.0, 1.0],
    "kinematic_type":"differencial",
    "type":"agent",
    "horizon": 10,
    "optimizer":{
        "type":"ddp",
        "gradient_rate":1.0,
        "regularisation":1.0,
    }
}
agents = [
    {
    "name":"1",
    "initial_state":[-0.5, 0.5, 0],
    "goal":[-2, 3, 0],
    },
    {
    "name":"2",
    "initial_state":[0.5, 0.5, 0],
    "goal":[2, 3, 0],
    },
    {
    "name":"3",
    "initial_state":[0.5, -0.5, 0],
    "goal":[2, -3, 0],
    },
    {
    "name":"4",
    "initial_state":[-0.5, -0.5, 0],
    "goal":[-2, -3, 0],
    },
]