"""Module implemnets concept of generalized edges."""


class Link:
    """Class for union of edges that represent a kind of big edge without significant branches.
    For example we need it because of small junctions that disjoin edge in several meters before TLS.
    """

    def __init__(self, head, edges, lanes_offsets, length, tail=None):
        self.head = head

        self.length = length
        self.edges = tuple(edges)

        self.lanes_offsets = lanes_offsets
        self.lane_ids = tuple(sorted(self.lanes_offsets.keys()))
        self.tail = tail

    def mount_to_tail(self, edges, lanes_offsets, tail=None):
        if self.tail:
            united_edges = self.edges + tuple(edges)

            united_offsets = self.lanes_offsets.copy()
            length = self.length

            for lane, offset in lanes_offsets.items():
                if lane not in united_offsets:
                    united_offsets[lane] = offset + self.length
                    length = max(offset + self.length, length)

            return Link(self.head.copy(), united_edges, united_offsets, length, tail=tail)
