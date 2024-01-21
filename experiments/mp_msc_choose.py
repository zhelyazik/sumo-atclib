import os
import sys

import numpy as np


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_atclib.environment.env import SumoEnvironment


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
        max_green=90,
        delta_time=5,
        additional_sumo_cmd=f"-a nets/Moscow/tls_cycles.evening.add.xml",
        mode="choose_next_phase",  # ["switch"|"choose_next_phase"|"phases_duration"]
        fixed_ts=False,
        use_gui=False,
    )

    initial_states, _, done, _ = env.reset()
    actions = {}  # {tls:{} for tls in env.tls_controllers.keys()}
    count = 0

    while not done["__all__"]:
        # print("--")
        obs, _, done, _ = env.step(action=actions)
        actions = {tls: np.argmax(obs[tls]["pressure"]) for tls in obs.keys()}

        if count % 50 == 0:
            print("count: ", count, ", time:", env.sim_step)
        count += 1

    env.save_csv(f"outputs/mp_msc_choose/mp_choose", 1)
    env.close()
