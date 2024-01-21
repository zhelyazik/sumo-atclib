"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from sumo_atclib.util.links_utils import make_links


MIN_SPEED = 1


class AbstractObserverState:
    """Abstract base class for observation functions."""

    def __init__(self):
        """Initialize observation function."""
        pass

    @abstractmethod
    def observe_state(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class StateObserver(AbstractObserverState):
    """Extended observation class for traffic signals."""

    VEH_LENGTH = 5
    BINS = 5

    def __init__(
        self,
        net,
        tls_phases,
        observable_distance: float = 100,
        max_length_for_unite_edges: float = 25,
    ):
        """Initialize default observation function."""
        self.net = net
        self.observable_distance = observable_distance

        self.max_length_for_unite_edges = max_length_for_unite_edges
        self.tls_phases = tls_phases

        self.generalized_edges = make_links(
            net=self.net, edge_length_bound=self.observable_distance, max_length_for_unite=max_length_for_unite_edges
        )

        self.lane2gen_edge = {}
        for geh, ge in self.generalized_edges.items():
            for lane in ge.lanes_offsets:
                assert lane not in self.lane2gen_edge, "One lane in several generalized edges!!!"
                self.lane2gen_edge[lane] = geh

        self.tls_connections = {}
        self.edges_to_observ = set()

        self.tls_edges_in = {}
        self.tls_edges_out = {}
        self.vehicles = {}

        for tls in self.net.getTrafficLights():
            tls.getID()

            self.tls_edges_in[tls.getID()] = tuple()
            self.tls_edges_out[tls.getID()] = tuple()

            for from_lane, to_lane, idx in tls.getConnections():
                from_ge = self.lane2gen_edge.get(from_lane.getID())
                to_ge = self.lane2gen_edge.get(to_lane.getID())

                assert from_ge and to_ge, (
                    f"Lane in connection from {from_lane.getID()}" f"to {to_lane.getID()} without generalized edge!!!"
                )

                if from_ge not in self.tls_edges_in[tls.getID()]:
                    self.tls_edges_in[tls.getID()] += (from_ge,)

                if to_ge not in self.tls_edges_out[tls.getID()]:
                    self.tls_edges_out[tls.getID()] += (to_ge,)

                self.edges_to_observ.add(from_ge)
                self.edges_to_observ.add(to_ge)

                for phase in tls_phases[tls.getID()]:
                    if phase.state[idx] in ("g", "G"):
                        if (tls.getID(), phase.state) not in self.tls_connections:
                            self.tls_connections[(tls.getID(), phase.state)] = set()
                        self.tls_connections[(tls.getID(), phase.state)].add((from_ge, to_ge))

        self.lanes_to_observe = []
        self.lane_min_pos = {}
        self.ge_total_length = {}

        for ge_head in self.edges_to_observ:
            ge = self.generalized_edges[ge_head]
            self.ge_total_length[ge_head] = 0

            for lane in ge.lane_ids:
                self.lane_min_pos[lane] = max(ge.lanes_offsets[lane] - self.observable_distance, 0)

                if self.net.getLane(lane).getLength() > self.lane_min_pos[lane]:
                    self.lanes_to_observe.append(lane)
                    self.ge_total_length[ge_head] += self.net.getLane(lane).getLength() - self.lane_min_pos[lane]

        self.lanes_to_observe_set = set(self.lanes_to_observe)
        self.transition2tls = {}
        self.tls_transitions = {}

        for tls_id in self.tls_edges_in.keys():
            self.tls_transitions[tls_id] = []
            for from_ge in self.tls_edges_in[tls_id]:
                for to_ge in self.tls_edges_out[tls_id]:
                    self.transition2tls[(from_ge, to_ge)] = tls_id
                    self.tls_transitions[tls_id].append((from_ge, to_ge))

    @property
    def observation_spaces(self):
        return {
            tls_id: spaces.Box(
                low=np.zeros(len(ein) + len(self.tls_edges_out[tls_id]) + 1),
                high=np.array((len(ein) + len(self.tls_edges_out[tls_id])) * [self.BINS] + [len(self.tls_phases[tls_id])]),
                dtype=np.int32,
            )
            for tls_id, ein in self.tls_edges_in.items()
        }

    def encode_state(self, state, tls_id):
        encoded = tuple(state[tls_id]["in"])
        encoded = tuple(int(val * self.BINS) for val in encoded) + (state[tls_id]["phase_id"],)
        return encoded

    def observe_state(self, sumo, subscriptions, constants):
        """Return the default observation."""

        veh_observed = {geh: 0 for geh in self.edges_to_observ}
        veh_stopped = {geh: 0 for geh in self.edges_to_observ}
        # veh_ids = [] for the future correspondence matrix
        transitions = {}
        veh_ge_id = {}

        pressure = {}
        total_stops = 0
        # Transitions count
        for step_info in subscriptions:
            for veh, info in step_info:
                lane = info[constants.VAR_LANE_ID]

                ge = self.lane2gen_edge.get(lane, None)
                if ge:
                    if (veh in veh_ge_id) and (veh_ge_id[veh] != ge):
                        if (veh_ge_id[veh], ge) not in transitions:
                            transitions[(veh_ge_id[veh], ge)] = 0
                        transitions[(veh_ge_id[veh], ge)] += 1

                    veh_ge_id[veh] = ge
                    total_stops += 1 if info[constants.VAR_SPEED] < MIN_SPEED else 0

        if subscriptions:
            for veh, info in subscriptions[-1]:
                lane = info[constants.VAR_LANE_ID]

                if lane in self.lanes_to_observe_set:
                    pos = info[constants.VAR_LANEPOSITION]

                    if pos >= self.lane_min_pos[lane]:
                        veh_observed[self.lane2gen_edge[lane]] += 1
                        veh_stopped[self.lane2gen_edge[lane]] += 1 if info[constants.VAR_SPEED] < MIN_SPEED else 0

        observation = {"total_stopped": total_stops}
        for tls, phases in self.tls_phases.items():
            observation[tls] = {
                "in": [],
                "out": [],
            }

            for geh in self.tls_edges_in[tls]:
                observation[tls]["in"].extend(
                    [
                        veh_observed[geh],  # * self.VEH_LENGTH / self.ge_total_length[geh],
                        veh_stopped[geh],  # * self.VEH_LENGTH / self.ge_total_length[geh],
                    ]
                )

            for geh in self.tls_edges_out[tls]:
                observation[tls]["out"].extend(
                    [
                        veh_observed[geh],  # * self.VEH_LENGTH / self.ge_total_length[geh],
                        veh_stopped[geh],  # * self.VEH_LENGTH / self.ge_total_length[geh],
                    ]
                )

            observation[tls]["pressure"] = []
            for phase in phases:
                phase_pressure = 0

                for road_in, road_out in self.tls_connections.get((tls, phase.state), tuple()):
                    phase_pressure += veh_stopped[road_in] - veh_stopped[road_out]
                observation[tls]["pressure"].append(phase_pressure)

            state = sumo.trafficlight.getRedYellowGreenState(tls)
            observation[tls]["state"] = tuple(1 if ch in ("G", "g") else 0 for ch in state)
            observation[tls]["transitions"] = tuple(transitions.get(t, 0) for t in self.tls_transitions[tls])

            observation[tls]["stopped"] = sum(observation[tls]["in"][1::2])
            observation[tls]["phase_id"] = sumo.trafficlight.getPhase(tls)
        return observation
