import argparse
import os
import sys

import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl.agents import QLAgent
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
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
        additional_sumo_cmd=f"-a nets/Moscow/tls_cycles.day.add.xml",
        mode="phases_duration",  # ["switch"|"choose_next_phase"|"phases_duration"]
        fixed_ts=True,
        use_gui=False,
    )
    _, _, done, _ = env.reset()

    while not done["__all__"]:
        actions = {}
        _, _, done, _ = env.step(action=actions)

    env.save_csv(f"outputs/msc_schedule/ql-msc_default_schedule", 1)
    env.close()
