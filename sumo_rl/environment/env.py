"""SUMO Environment for Traffic Signal Control."""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

import gymnasium as gym
import libsumo as traci
import numpy as np
import pandas as pd
import sumolib
from sumolib.net import readNet as read_net

from .observer import MIN_SPEED, StateObserver
from .traffic_controller import (
    TrafficLightsScheduleController,
    TrafficRealTimeController,
)


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class SumoEnvironment(gym.Env):
    """SUMO Environment for Traffic Signal Control.

    Class that implements a gym.Env interface for traffic signal control using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        sim_max_time (int): Number of simulated seconds on SUMO. The time in seconds the simulation must end. Default: 3600
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict): String with the name of the reward function used by the agents, a reward function, or dictionary with reward functions assigned to individual traffic lights by their keys.
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        fixed_ts (bool): If true, it will follow the phase configura`tion in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
        mode (str): ["switch"|"choose_next"|"phases_duration"|"nonadaptive"]
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        sim_max_time: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
        mode: str = None,
        observation_time: int = 1,
    ) -> None:
        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self.mode = mode
        self._net = net_file
        self.net = read_net(self._net, withInternal=True)

        self._route = route_file
        self.use_gui = use_gui

        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = sim_max_time
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        self.sumo_seed = sumo_seed
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0

        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observation_time = observation_time
        self.constants = traci.constants

        # self.observations = {ts: None for ts in self.ts_ids}
        # self.rewards = {ts: None for ts in self.ts_ids}

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if True:  # LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci

        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        # if self.use_gui or self.render_mode is not None:
        #     self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)

        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed

        self._start_simulation()
        self.ts_ids = list(self.sumo.trafficlight.getIDList())

        if self.mode == "phases_duration":
            self.tls_controllers = {
                ts: TrafficLightsScheduleController(
                    trafficlight=self.sumo.trafficlight,
                    tls_id=ts,
                    min_green=self.min_green,
                    max_green=self.max_green,
                )
                for ts in self.ts_ids
            }

        elif self.mode == "choose_next_phase":
            self.tls_controllers = {
                ts: TrafficRealTimeController(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.sumo,
                    cyclic_mode=False,
                )
                for ts in self.ts_ids
            }

        elif self.mode == "switch":
            self.tls_controllers = {
                ts: TrafficRealTimeController(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.sumo,
                    cyclic_mode=True,
                )
                for ts in self.ts_ids
            }
        else:
            raise Exception("unknown mode")

        self.state_observer = StateObserver(
            self.net,
            {tc_id: tc.green_phases for tc_id, tc in self.tls_controllers.items()},
        )
        self.departures = {}
        self.arrived = set()

        self._update_tc()
        self.observations = {tc_id: [] for tc_id in self.tls_controllers}

        self.prev_simstep_subscription = []
        self.total_travel_time = 0
        self.arrived_travel_time = 0

        self.total_timeloss = 0
        self.arrived_timeloss = 0
        return self.step({})

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()

    def _update_subscriptions(self):
        for vid in self.sumo.simulation.getDepartedIDList():
            time = self.sumo.simulation.getTime()
            self.sumo.vehicle.subscribe(
                vid,
                [
                    self.constants.VAR_LANE_ID,
                    self.constants.VAR_TIMELOSS,
                    self.constants.VAR_SPEED,
                    self.constants.VAR_LANEPOSITION,
                ],
            )
            self.departures[vid] = time

    def _update_tc(
        self,
    ):
        self.time_to_act = {}
        for tc_id, tc in self.tls_controllers.items():
            time_to_act = tc.update(self.sim_step)

            if time_to_act:
                self.time_to_act[tc_id] = time_to_act

    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        self._apply_actions(action)
        self.time_to_act = {}
        subscriptions = []

        while not self.time_to_act:
            self._update_subscriptions()
            subscriptions.append(self.sumo.simulation.step())

            arrived = set(self.sumo.simulation.getArrivedIDList())
            prev_sub = self.prev_simstep_subscription if len(subscriptions) < 2 else subscriptions[-2]
            time = self.sumo.simulation.getTime()

            self.total_travel_time = 0
            self.total_timeloss = 0

            for v, info in prev_sub:
                if v in arrived:
                    self.arrived.add(v)

                    self.arrived_travel_time += time - self.departures[v]
                    self.arrived_timeloss += info[self.constants.VAR_TIMELOSS]

                else:
                    self.total_travel_time += time - self.departures[v]
                    self.total_timeloss += info[self.constants.VAR_TIMELOSS]

            self.total_travel_time += self.arrived_travel_time
            self.total_timeloss += self.arrived_timeloss

            if int(self.sumo.simulation.getTime() - time) % self.observation_time == 0:
                observations = self.state_observer.observe_state(self.sumo, subscriptions, self.constants)

                for tc_id in self.tls_controllers:
                    self.observations[tc_id].append(observations[tc_id])
            self._update_tc()

        if subscriptions:
            self.prev_simstep_subscription = subscriptions[-1]

        else:
            observations = self.state_observer.observe_state(self.sumo, [], self.constants)
            for tc_id in self.tls_controllers:
                self.observations[tc_id].append(observations[tc_id])

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()

        for tc_id in self.time_to_act:
            self.observations[tc_id] = []

        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps

        info = self._compute_info()
        return observations, rewards, dones, info

    def _apply_actions(self, actions):
        """Set the next green phase for the traffic signals.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {ts_id : greenPhase}
        """
        for tc_id in self.time_to_act:
            if tc_id not in actions:
                self.tls_controllers[tc_id].default_action()

            else:
                self.tls_controllers[tc_id].apply_action(actions[tc_id])

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        self.metrics.append(info)
        return info

    def _compute_rewards(self):
        return {
            tc_id: -sum(obs["stopped"] for obs in s) / len(s)
            for tc_id, s in self.observations.items()
            if tc_id in self.time_to_act
        }

    def _compute_observations(self):
        """Return the observation space of a traffic signal."""
        return {tc_id: s[-1] if s else s for tc_id, s in self.observations.items() if tc_id in self.time_to_act}

    @property
    def observation_spaces(self):
        """Return the observation space of a traffic signal."""
        return self.state_observer.observation_spaces

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal."""
        return self.tls_controllers[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulation.step()

    def _get_system_info(self):
        speeds = [info[self.constants.VAR_SPEED] for _, info in self.prev_simstep_subscription]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_stopped": sum(int(speed < MIN_SPEED) for speed in speeds),
            "system_total_timeloss": self.total_timeloss,
            "system_average_timeloss": 0.0 if len(self.departures) == 0 else self.total_timeloss / len(self.departures),
            "system_mean_speed": 0.0 if len(speeds) == 0 else np.mean(speeds),
            "system_total_travel_time": self.total_travel_time,
            "system_average_travel_time": 0.0 if len(self.departures) == 0 else self.total_travel_time / len(self.departures),
            "arrivals": len(self.arrived),
            "departurs": len(self.departures),
        }

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        if self.sumo is None:
            return

        # if not LIBSUMO:
        #     traci.switch(self.label)
        # traci.close()

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self.close()

    def render(self):
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame

        elif self.render_mode == "rgb_array":
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        """Encode the state of the traffic signal into a hashable object."""
        return self.state_observer.encode_state(state, ts_id)
