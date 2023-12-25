"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
from typing import Union

import numpy as np
from gymnasium import spaces
from sumolib.net import Phase


class TrafficLightController:
    def __init__(
        self,
    ):
        pass

    def apply_action(
        self,
    ):
        pass

    def time_to_act(
        self,
    ):
        pass

    def update(
        self,
    ):
        return


class TrafficLightsScheduleController:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_pactionhase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    def __init__(
        self,
        trafficlight,
        tls_id: str,
        logic_id: Union[str, None] = None,
        delta_time: int = 5,  ## ??
        min_green: Union[int, None] = None,
        max_green: Union[int, None] = None,
        force_min_max_duration: bool = False,
        act_phase: int = 0,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            sumo (Sumo): The Sumo instance.
        """
        self.tls_id = tls_id
        self.delta_time = delta_time
        self.act_phase = act_phase

        self.min_green = min_green
        self.max_green = max_green

        self.force_min_max_dur = force_min_max_duration
        self.trafficlight = trafficlight

        self.is_phase_green = lambda phase: ("g" in phase.state or "G" in phase.state) and ("y" not in phase.state)
        self._build_phases()

        low = np.zeros(len(self.phase_id2action_id), dtype=np.int32)
        high = np.zeros_like(low)

        for phase_id, action_id in self.phase_id2action_id.items():
            phase = self.program.phases[phase_id]
            l, h = self.min_green, self.max_green

            if not self.force_min_max_dur and (phase.minDur != phase.maxDur):
                l = max(l, phase.minDur)
                h = min(h, phase.maxDur)

            low[action_id] = l
            high[action_id] = h

        self.action_space = spaces.Box(low=low, high=high, dtype=np.int32)
        self.switch_time = 0

    def _build_phases(self, logic_id: str = None):  # TODO: rewrite to be compitible use predefined programs
        if not logic_id:
            logic_id = self.trafficlight.getProgram(self.tls_id)

        logic = [l for l in self.trafficlight.getAllProgramLogics(self.tls_id) if l.programID == logic_id][0]

        self.program = logic
        self.phases = logic.phases

        self.green_phases = []
        self.phase_id2action_id = {}

        for i, phase in enumerate(self.phases):
            if self.is_phase_green(phase):
                self.phase_id2action_id[i] = len(self.phase_id2action_id)
                self.green_phases.append(phase)

    def apply_action(self, action):
        """Application of the new green phases durations.

        Args:
            action (array[int]): green phases durations [d1, d2, ...]
        """
        new_phases = []
        for ph_i, phase in enumerate(self.phases):
            if ph_i in self.phase_id2action_id:
                ac_i = self.phase_id2action_id[ph_i]
                new_phase = self.trafficlight.Phase(float(action[ac_i]), phase.state, -1.0, -1.0, (), phase.name)

            else:
                new_phase = phase
            new_phases.append(new_phase)

        new_phases = tuple(new_phases)
        phase_index = self.trafficlight.getPhase(self.tls_id)

        self.trafficlight.setProgramLogic(self.tls_id, self.trafficlight.Logic("var", 0, phase_index, new_phases))
        self.trafficlight.setProgram(self.tls_id, "var")

        # current_duration = new_phases[phase_index].duration
        # self.trafficlight.setPhaseDuration(self.tls_id, current_duration)
        # self.switch_time = self.trafficlight.getNextSwitch(self.tls_id)

    # def time_to_act(self):
    #     return self.trafficlight.getPhase(self.tls_id) == self.act_phase and \
    #         self.trafficlight.getNextSwitch(self.tls_id) > self.switch_time

    def update(self, time):
        """Updates the traffic signal state."""
        return (self.trafficlight.getNextSwitch(self.tls_id)) == time and (
            self.trafficlight.getPhase(self.tls_id) == len(self.phases) - 1
        )

    def default_action(self):
        """For case then control action have not been received."""
        return None


class TrafficRealTimeController:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        begin_time: int,
        sumo,
        cyclic_mode: bool = False,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
            cyclic_mode (bool): if True just two actions allowed: switch to next phase or not
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time

        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green

        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0

        self.next_action_time = begin_time
        self.sumo = sumo

        self.is_phase_green = lambda phase: ("g" in phase.state or "G" in phase.state) and ("y" not in phase.state)
        self._build_phases()

        self.action_space = spaces.Discrete(self.num_green_phases)
        self.cyclic_mode = cyclic_mode

        if self.cyclic_mode:
            self.action_space = spaces.Discrete(2)

    def _build_phases(self):
        logic_id = self.sumo.trafficlight.getProgram(self.id)
        logic = [l for l in self.sumo.trafficlight.getAllProgramLogics(self.id) if l.programID == logic_id][0]

        phases = logic.phases
        self.phases = logic.phases
        self.green_phases = []

        for phase in phases:
            state = phase.state
            if self.is_phase_green(phase):
                self.green_phases.append(self.sumo.trafficlight.Phase(phase.duration, state))  # maybe phase.min?

        self.num_green_phases = len(self.green_phases)

        self.all_phases = self.green_phases.copy()
        self.yellow_dict = {}

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        if self.env.fixed_ts:
            return
        logic.phases = self.all_phases

        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self, time=-1):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.green_phases[self.green_phase].state)
            self.is_yellow = False

        return self.time_to_act

    def apply_action(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.cyclic_mode:
            new_phase += self.green_phase
            new_phase = new_phase % len(self.green_phases)

        if (new_phase == self.green_phase) and (self.time_since_last_phase_change - self.yellow_time > self.max_green):
            new_phase += 1
            new_phase = new_phase % len(self.green_phases)

        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.green_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )

            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time

            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def default_action(self):
        """For case then control action have not been received."""
        self.next_action_time = self.env.sim_step + self.delta_time
