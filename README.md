# sumo-atclib
Framework for adaptive traffic control with SUMO inspired by SUMO-RL package and started as its fork.

As of now, it mostly supports LibSUMO API, but TraCI compatibility will be provided.

In the folder [nets/RESCO](https://github.com/zhelyazik/sumo-atclib/tree/master/nets/RESCO) you can find the network and route files from [RESCO](https://github.com/jault/RESCO) (Reinforcement Learning Benchmarks for Traffic Signal Control), which was built on top of SUMO-RL. See their [paper](https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf) for results.


 In the folder [experiments](https://github.com/zhelyazik/sumo-atclib/tree/master/experiments) you can find examples on how to instantiate an environment and train your RL agent, most of them from were borrowed from [sumo-rl](https://github.com/LucasAlegre/sumo-rl/tree/master/experiments).
