import argparse
import os
import sys

import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    alpha = 0.02
    gamma = 0.95

    decay = 1
    runs = 1

    episodes = 20
    epsilon = 0.1

    env = SumoEnvironment(
        net_file="nets/Moscow/osm.net.xml",
        route_file=(
            "nets/Moscow/routes_chumakova.rou.xml,"
            "nets/Moscow/routes_moskvitina.rou.xml,"
            "nets/Moscow/routes_valuevskoe.rou.xml,"
            "nets/Moscow/routes_habarova.rou.xml,"
            "nets/Moscow/routes_atlasova.rou.xml"
            ),
        begin_time=36000,
        sim_max_time=40000,
        min_green=20,
        max_green=100,
        delta_time=5,
        additional_sumo_cmd=f"-a nets/Moscow/tls_cycles.evening.add.xml",
        mode="choose_next_phase", # ["switch"|"choose_next_phase"|"phases_duration"]
        fixed_ts=False,
        use_gui=False,
    )

    for run in range(1, runs + 1):
        initial_states, _, done, _ = env.reset()
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states, ts),
                state_space=env.observation_spaces[ts],
                action_space=env.action_spaces(ts),
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=epsilon, min_epsilon=0.005, decay=decay),
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            print(f"episode: {episode}")
            count = 0

            if episode != 1:
                initial_states, _, done, _ = env.reset()

                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states, ts)

            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s, agent_id), reward=r[agent_id])

                if count % 100 == 0:
                    print("count: ", count, ", time:", env.sim_step)
                count += 1

            env.save_csv(f"outputs/msc_q_choose/ql-msc_mode_choose", episode)
    env.close()
