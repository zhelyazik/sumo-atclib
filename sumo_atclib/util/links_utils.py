"""Module with functions for work with generalized_edge."""
from sumolib.net.edge import Edge

from sumo_rl.environment.edge import Link


def is_not_TLS(connections):
    return all(c.getTLSID() == "" for c in connections)


def get_generalized_edge_to_tls(
    to_tls_edge: Edge,
    net,
    edge_length_bound: float,
    max_length_for_unite: float,
):
    edge = to_tls_edge
    length = 0

    edges = []
    lanes_offset = dict()
    head = {to_tls_edge.getID()}

    while length < edge_length_bound:
        incoming = [(e, c) for e, c in edge.getIncoming().items() if e.getFunction() != "internal"]
        connections = []

        if (len(incoming) == 1) and (is_not_TLS(incoming[0][1])):
            from_same = [(e, c) for e, c in incoming[0][0].getOutgoing().items() if e.getFunction() != "internal"]

            if (len(from_same) == 1) or (
                (edge == to_tls_edge)
                and any((e.getToNode() == edge.getToNode()) and (e.getLength() <= max_length_for_unite) for e, _ in from_same)
            ):
                if len(from_same) > 1:
                    for e, _ in from_same:
                        head.add(e.getID())

                for _, cns in from_same:
                    connections.extend(cns)

        for lane in edge.getLanes():
            lanes_offset[lane.getID()] = length + lane.getLength()
        dlength = 0

        for conn in connections:
            if (conn.getTo().getID() != edge.getID()) and (conn.getTo().getID() not in edges):
                edges.append(conn.getTo().getID())

            lane = conn.getToLane()
            lanes_offset[lane.getID()] = length + lane.getLength()
            internal = net.getLane(conn.getViaLaneID())

            lanes_offset[internal.getID()] = lanes_offset[lane.getID()] + internal.getLength()
            dlength += lane.getLength() + internal.getLength()

        edges.append(edge.getID())
        if len(connections) > 0:
            length += dlength / len(connections)
            edge = connections[0].getFrom()

        else:
            length += edge.getLength()

    return Link(frozenset(head), edges, lanes_offset, length, tail=edges[-1])


def get_generalized_edge_from_tls(
    from_tls_edge: Edge,
    net,
    heads,
    edge_length_bound: float,
    max_length_for_unite: float,
):
    edge = from_tls_edge
    length = 0

    edges = []
    lanes_offset = dict()
    tail = edge.getID()

    while abs(length) < edge_length_bound:
        if edge.getID() in heads:
            return heads[edge.getID()].mount_to_tail(edges, lanes_offset, tail)

        outgoing = [(e, c) for e, c in edge.getOutgoing().items() if e.getFunction() != "internal"]
        connections = []

        if (len(outgoing) == 1) and (is_not_TLS(outgoing[0][1])):
            to_same = [(e, c) for e, c in outgoing[0][0].getIncoming().items() if e.getFunction() != "internal"]

            if (len(to_same) == 1) or (
                (edge == from_tls_edge)
                and any(
                    (e.getFromNode() == edge.getFromNode()) and (e.getLength() <= max_length_for_unite) for e, _ in to_same
                )
            ):
                if len(to_same) > 1:
                    tail = None
                for _, cns in to_same:
                    connections.extend(cns)
        length = edge.getLength()

        for lane_id in lanes_offset.keys():
            lanes_offset[lane_id] += edge.getLength()
            length = max(length, lanes_offset[lane_id])

        for lane in edge.getLanes():
            lanes_offset[lane.getID()] = lane.getLength()

        for conn in connections:
            if (conn.getFrom().getID() != edge.getID()) and (conn.getFrom().getID() not in edges):
                edges.insert(0, conn.getFrom().getID())

            lane = conn.getFromLane()
            internal = net.getLane(conn.getViaLaneID())

            lanes_offset[lane.getID()] = internal.getLength() + lane.getLength()
            lanes_offset[internal.getID()] = internal.getLength()

        edges.insert(0, edge.getID())

        if len(connections) > 0:
            edge = connections[0].getTo()

        else:
            break

    return Link(frozenset([edge.getID()]), edges, lanes_offset, length, tail)


def make_links(
    net,
    edge_length_bound: float,
    max_length_for_unite: float,
):
    to_tls_edges = set()
    from_tls_edges = set()

    for tls in net.getTrafficLights():
        for f, t, _ in tls.getConnections():
            to_tls_edges.add(f.getEdge())
            from_tls_edges.add(t.getEdge())

    heads = dict()
    for to_tls in to_tls_edges:
        ge = get_generalized_edge_to_tls(
            to_tls, net, edge_length_bound=edge_length_bound, max_length_for_unite=max_length_for_unite
        )
        heads[ge.tail] = ge

    generalized = dict()
    for from_tls in from_tls_edges:
        ge = get_generalized_edge_from_tls(
            from_tls, net, heads, edge_length_bound=edge_length_bound, max_length_for_unite=max_length_for_unite
        )
        generalized[ge.head] = ge

    for ge in heads.values():
        if ge.head not in generalized:
            generalized[ge.head] = ge

    return generalized
