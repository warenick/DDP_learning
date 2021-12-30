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
        "regularisation":0.95,
    }
}
agents = [
    {
    "name":"0",
    "initial_state":[-4., 0., 1.57],
    "goal":[-5, -3, 0],
    "optimizer":{
        "type":"ddp",
        "gradient_rate":1.0,
        "regularisation":0.95,
    }
    },
    {
    "name":"1",
    "initial_state":[-2., 0., 1.57],
    "goal":[-3, -3, 0],
    "optimizer":{
        "type":"ddp",
        "gradient_rate":0.8,
        "regularisation":0.95,
    }
    },
    {
    "name":"2",
    "initial_state":[0., 0., 1.57],
    "goal":[-1, -3, 0],
    "optimizer":{
        "type":"ddp",
        "gradient_rate":0.6,
        "regularisation":0.95,
    }
    },
    {
    "name":"3",
    "initial_state":[2, 0, 1.57],
    "goal":[1, -3, 0],
    "optimizer":{
        "type":"ddp",
        "gradient_rate":0.4,
        "regularisation":0.95,
    }
    },
    {
    "name":"4",
    "initial_state":[4, 0, 1.57],
    "goal":[3, -3, 0],
    "optimizer":{
        "type":"ddp",
        "gradient_rate":0.2,
        "regularisation":0.95,
    }
    },
]