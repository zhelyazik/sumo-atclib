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
        mode="phases_duration",  # ["switch"|"choose_next_phase"|"phases_duration"]
        fixed_ts=False,
        use_gui=False,
        observation_time=50,
    )

    initial_states, _, done, _ = env.reset()
    low = {tls: tc.action_space.low for tls, tc in env.tls_controllers.items()}
    high = {tls: tc.action_space.high for tls, tc in env.tls_controllers.items()}

    tls_availabe_times = {tls: tc.action_space.high - tc.action_space.low for tls, tc in env.tls_controllers.items()}
    available = {tls_id: len(slots) * 10 for tls_id, slots in tls_availabe_times.items()}
    actions = {}

    count = 0
    while not done["__all__"]:
        obs, _, done, _ = env.step(action=actions)

        addition = {}
        for tls in obs.keys():
            pressure = np.array(obs[tls]["pressure"])

            pressure[pressure < 0] = 0
            addition[tls] = pressure / sum(pressure) * available[tls]

        actions = {tls: low[tls] + addition[tls] for tls in obs.keys()}
        actions = {tls: [int(min(h, a)) for h, a in zip(high[tls], act)] for tls, act in actions.items()}

        if count % 1 == 0:
            print("count: ", count, ", time:", env.sim_step)

        count += 1

    env.save_csv(f"outputs/mp_msc_schedule/mp-msc", 1)
    env.close()
