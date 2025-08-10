from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Set, Tuple
import random
import numpy as np


try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
except ImportError:
    print("matplotlib is not installed, plots will not be available")

INFINITY = np.iinfo(np.int32).max


# NOTE: all data formats here chosen crudely, to be optimized later
edge_dtype = np.dtype([
    ("group", "i4"), # 0-indexed group id
    ("result", "i4"), # 1 if success, -1 if failed, 0 if not tested yet
    ("target", "U32"), # target node hash-name, "" if not tested or failed
    ("distance", "i4"), # distance to the frontier node, 0 means next node is the frontier
    ("errors", "i4"), # number of errors so far
])

def format_struct_table(arr):
    names = ("idx",) + arr.dtype.names
    cols = []
    for name in names:
        if name == "idx":
            cols.append([str(i) for i in range(len(arr))])
        else:
            cols.append([str(r[name]) for r in arr])
    widths = [max(len(n), *(len(v) for v in col)) for n, col in zip(names, cols)]
    header = " | ".join(n.ljust(w) for n, w in zip(names, widths))
    sep = "-+-".join("-"*w for w in widths)
    lines = []
    for i in range(len(arr)):
        line = " | ".join(cols[j][i].ljust(widths[j]) for j in range(len(names)))
        lines.append(line)
    return "\n".join([header, sep, *lines])

@dataclass
class NodeInfo:
    name: Hashable

    total_candidates: int # how many exist
    num_groups: int = 1 # FIXME: is never used
    active_group: int = 0

    group2remaining_candidate_ids: List[Set[int]] = field(default_factory=list)

    edge_data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=edge_dtype))

    error_threshold: int = 3
    closed: bool = False # flips when last probe done
    distance: float | None = 0 # TODO: how is it initialized?

    def __post_init__(self):

        assert self.name is not None, "Node name must be provided"

        if self.num_groups > 1 and self.group2remaining_candidate_ids is None:
            raise ValueError("group2remaining_candidate_ids must be provided if num_groups > 1")

        if self.num_groups == 1 and self.group2remaining_candidate_ids is None:
            self.group2remaining_candidate_ids = [set(range(self.total_candidates))]

        self.group2remaining_candidate_ids = [set(r_c_ids) for r_c_ids in self.group2remaining_candidate_ids] # ensure it's a list of sets

        self.edge_data = np.zeros(self.total_candidates, dtype=edge_dtype)
            
        for group_id, remaining_candidate_ids in enumerate(self.group2remaining_candidate_ids):
            self.edge_data["group"][list(remaining_candidate_ids)] = group_id

    @property
    def has_open(self) -> bool:
        """Still hiding ≥1 untested edge?"""
        return len(self.tested) < self.total_candidates

    def record_test(self, edge_idx: int, success: int, target_node: Hashable | None = None) -> bool:

        edge_group_id = self.edge_data[edge_idx]["group"]

        assert self.edge_data["result"][edge_idx] == 0 and \
            self.edge_data["target"][edge_idx] == "" and \
            self.edge_data["distance"][edge_idx] == 0, \
            "Edge result must be untested before recording a test"

        if success == -1:
            self.edge_data["errors"][edge_idx] += 1
            if self.edge_data["errors"][edge_idx] >= self.error_threshold:
                self.edge_data["errors"][edge_idx] = 0
                new_group_id = edge_group_id + 1
                if new_group_id > self.num_groups - 1:
                    # count it as failed and move on
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
                    self.edge_data["result"][edge_idx] = -1
                    self.edge_data["distance"][edge_idx] = INFINITY
                    return True
                else:
                    self.edge_data["group"][edge_idx] = new_group_id
                    self.group2remaining_candidate_ids[new_group_id].add(edge_idx)
                    self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)
            return False

        self.group2remaining_candidate_ids[edge_group_id].discard(edge_idx)

        if success == 1:
            self.edge_data["target"][edge_idx] = str(target_node)
            self.edge_data["distance"][edge_idx] = -1 # NOTE: distance is maintained by the GraphExplorer class
            self.edge_data["result"][edge_idx] = 1
        elif success == 0:
            self.edge_data["distance"][edge_idx] = INFINITY
            self.edge_data["result"][edge_idx] = -1

        return True

    def has_open_group(self, group_id: int) -> bool:
        """Return True if this node has at least one untested edge belonging to *group_id* or below."""
        for i in range(group_id+1):
            if len(self.group2remaining_candidate_ids[i]) > 0:
                return True
        return False
    
    def __repr__(self) -> str:
        edge_data_repr = format_struct_table(self.edge_data)

        return f"""NodeInfo:
name={self.name},
total_candidates={self.total_candidates},
num_groups={self.num_groups},
distance={self.distance},
closed={self.closed},
{edge_data_repr}
"""


class GraphExplorer:

    def __init__(
        self,
        start_node: Hashable | None = None, 
        num_candidates: int | None = None, 
        group2remaining_candidate_ids: List[Set[int]] | None = None,
        n_groups: int = 1,
        verbose_level: int = 0,
        ) -> None:

        self._verbose_level = verbose_level
        self._n_groups = max(1, n_groups)

        self.reset()

    def reset(self) -> None:
        self._nodes: Dict[Hashable, NodeInfo] = {}
        self._G: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set) # (edge_idx, target_node)
        self._G_rev: Dict[Hashable, Set[Tuple[int, Hashable]]] = defaultdict(set) # (edge_idx, source_node)
        self._frontier: Set[Hashable] = set()
        self._dist: Dict[Hashable, int] = {}
        self._next: Dict[Hashable, Tuple[int, Hashable]] = {} # (edge_idx, target_node)
        self._active_group: int = 0  # current priority group

        self.suspicious_transitions: Dict[Tuple[Hashable, int, Hashable], int] = {} # (source_node, edge_idx, target_node) -> count
        self.suspicious_transitions_threshold: int = 3

        self._empty = True
    
    def initialize(self, start_node: Hashable | None = None, num_candidates: int | None = None, group2remaining_candidate_ids: List[Set[int]] | None = None) -> None:


        if start_node is not None:
            self._add_new_node(start_node, num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)

        if self._verbose_level >= 1:
            print(f"\nGraph is initialized with node: {self._nodes[start_node]}")
            self.dump()

    def record_test(
        self,
        node: Hashable,
        edge_idx: Hashable,
        success: bool,
        target_node: Optional[Hashable] = None,
        target_num_candidates: Optional[int] = None,
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None,
        suspicious_transition: bool = False,
    ) -> None:

        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}") # TODO: alternatively, add it to the graph
        node_info = self._nodes[node]

        if node_info.closed:
            if target_node == self._nodes[node].edge_data["target"][edge_idx]:
                if self._verbose_level >= 1:
                    print(f"Node {node!r} is closed, skipping test {edge_idx!r}")
                return
            else:
                if self._verbose_level >= 1:
                    print(f"Node {node!r} is closed, we perform the test only if the target node is closer to frontier than the original target node. It will allow to fix the broken transition.")
                dist_to_frontier = self._dist.get(target_node, 0) # 0 if it wasn't previously recorded (so it's in the frontier)
                prev_target_node = self._nodes[node].edge_data["target"][edge_idx]
                prev_dist_to_frontier = self._dist.get(prev_target_node, INFINITY)

                if dist_to_frontier < prev_dist_to_frontier:
                    if self._verbose_level >= 1:
                        print(f"Target node {target_node!r} is closer to frontier than the original target node {prev_target_node!r}, we perform the test")
                else:
                    if self._verbose_level >= 1:
                        print(f"Target node {target_node!r} is further from frontier than the original target node {prev_target_node!r}, we skip the test")
                    return

        # store metadata immediately
        if self._verbose_level >= 1:
            print(f"Recording action {edge_idx} from {node} to {target_node} with success {success}")

        if suspicious_transition:
            self.suspicious_transitions[(node, edge_idx, target_node)] = self.suspicious_transitions.get((node, edge_idx, target_node), 0) + 1
            if self.suspicious_transitions[(node, edge_idx, target_node)] < self.suspicious_transitions_threshold:
                print(f"Suspicious transition detected: {node, edge_idx, target_node}, count: {self.suspicious_transitions[(node, edge_idx, target_node)]}")
                print(f"It will be ignored for now, but will be allowed after {self.suspicious_transitions_threshold} attempts")
                return
        
        node_info.record_test(edge_idx, success, target_node)
        
        # successful hop ⇒ register edge and maybe discover a brand-new node
        if success == 1:
            if target_node is None:
                raise ValueError("target_node required when success=True")

            if target_node not in self._nodes:
                new_node = True
                if target_num_candidates is None:
                    raise ValueError(
                        "target_num_candidates required for a new node"
                    )
                self._add_new_node(target_node, target_num_candidates, group2remaining_candidate_ids=group2remaining_candidate_ids)
            else:
                new_node = False


            self._G[node].add((edge_idx, target_node))
            self._G_rev[target_node].add((edge_idx, node))

            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)

            if self._nodes[target_node].has_open_group(self.active_group):
                # self._tighten_from_new_source(target_node)
                self._rebuild_distances()
            else:
                self._close_node(target_node)
                self._maybe_advance_group(target_node)

        else:
            if not self._nodes[node].has_open_group(self.active_group):
                self._close_node(node)
                self._maybe_advance_group(node)

        if self._verbose_level >= 1:
            if success == 1:
                success_str = "succeeded"
            elif success == -1:
                success_str = "threw an error"
            else:
                success_str = "failed"

            print(f"\n\nNode {node!r} candidate {edge_idx!r} {success_str}:")
            print(f"Source node:\n{self._nodes[node]}")
            if success == 1:
                print(f"{'NEW' if new_node else 'Existing'} target node:\n{self._nodes[target_node]}")
        self.dump()

    def get_distance(self, node: Hashable) -> Optional[int]:
        d = self._dist.get(node)
        return None if d is None or d == float("inf") else d

    def get_next_hop(self, node: Hashable) -> Optional[Hashable]:
        # NOTE: DEPRECATED
        # Return the node itself only if it truly has open edges in the active group
        if node in self._frontier: # and self._nodes[node].has_open_group(self.active_group):
            return node
        nxt = self._next.get(node)
        if nxt is None:
            return None
        # _next may store (edge_idx, next_node); return the node only
        if isinstance(nxt, tuple) and len(nxt) == 2:
            return nxt[1]
        return nxt

    def edge_info(self, node: Hashable, edge_idx: Hashable) -> np.ndarray:
        return self._nodes[node].edge_data[edge_idx]

    def is_finished(self) -> bool:
        return not self._frontier

    @property
    def active_group(self) -> int:
        return self._active_group
    
    @property
    def empty(self) -> bool:
        return self._empty

    def _add_new_node(self, node: Hashable, 
        n_candidates: int, 
        group2remaining_candidate_ids: Optional[List[Set[int]]] = None
        ) -> None:

        if n_candidates < 1:
            raise ValueError("num_candidates must be positive")

        self._nodes[node] = NodeInfo(node, n_candidates, self._n_groups, group2remaining_candidate_ids=group2remaining_candidate_ids)
        self._G[node] = set()
        self._G_rev[node] = set()

        if self._empty:
            self._empty = False

        if self._nodes[node].has_open_group(self.active_group):
            self._frontier.add(node)
        else:
            self._close_node(node)
            self._maybe_advance_group(node)


    def _close_node(self, node: Hashable) -> None:
        node_info = self._nodes[node]
        if node_info.closed:
            return
        node_info.closed = True
        self._frontier.discard(node)
        self._rebuild_distances() # removal from frontier may increase some distances in the graph

    def _tighten_from_new_source(self, src: Hashable) -> None:
        # NOTE: is not used anymore
        dq = deque([src])
        self._dist[src] = 0
        self._nodes[src].distance = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                initial_u_dist = self._dist.get(u, INFINITY)
                u_edge_data = self._nodes[u].edge_data
                u_edge_data["distance"][edge_idx] = self._nodes[v].distance + 1
                updated_u_dist = u_edge_data["distance"][u_edge_data["group"] <= self.active_group].min()
                self._nodes[u].distance = updated_u_dist
                self._dist[u] = updated_u_dist
                if updated_u_dist > initial_u_dist:
                    dq.append(u)

    def _rebuild_distances(self) -> None:
        """
        Rebuild the distances from the frontier nodes in the graph.
        """
        self._dist.clear()
        self._next.clear()
        dq = deque(self._frontier)
        for node, node_info in self._nodes.items():
            node_info.distance = INFINITY
            self._dist[node] = INFINITY
        for src in self._frontier:
            self._nodes[src].distance = 0
            self._dist[src] = 0
        while dq:
            v = dq.popleft()
            v_dist = self._dist.get(v, INFINITY)
            for edge_idx, u in self._G_rev.get(v, ()):  # (edge_idx, source_node)
                u_info = self._nodes[u]
                u_dist = self._dist.get(u, INFINITY)
                u_info.edge_data["distance"][edge_idx] = v_dist + 1
                if u_dist > u_info.edge_data["distance"][edge_idx]:
                    u_info.distance = u_info.edge_data["distance"][edge_idx]
                    self._dist[u] = u_info.edge_data["distance"][edge_idx]
                    self._next[u] = (edge_idx, v)
                    dq.append(u)

    def _maybe_advance_group(self, current_node: Hashable) -> None:
        """
        If it's not possible to reach any frontier node from the current node,
        given the current active group, advance to the next higher group id and rebuild distances.
        """

        distance = self._nodes[current_node].distance
        while distance == INFINITY and self.active_group < self._n_groups - 1:
            print(f"Node {current_node!r} is not reachable from any frontier node under {self.active_group}, advancing to the next group")

            self._active_group += 1
            self._dist.clear()
            self._next.clear()
            self._frontier.clear()

            for node, node_info in self._nodes.items():
                node_info.active_group = self.active_group
                if node_info.has_open_group(self.active_group):
                    self._frontier.add(node)
                    node_info.closed = False

            self._rebuild_distances()
            distance = self._dist.get(current_node)
        
    def dump(self) -> None:
        if self._verbose_level >= 1:
            print("=== explorer state ===")
            print("frontier :", self._frontier)
            print("N nodes  :", len(self._nodes))
            print("N edged candidates  :", sum(len(node_info.edge_data) for node_info in self._nodes.values()))
            if self._verbose_level >= 2:
                print("Graph    :", self._G)
                print("dist     :", self._dist)
                print("next hop :", self._next)
            print("======================")

    def print_all_nodes(self) -> None:
        for node_info in self._nodes.values():
            print(node_info)

    def choose_edge(self, node: Hashable, return_reasoning: bool = False) -> Hashable:
        # TODO: make it possible to choose completely random edge
        node_info = self._nodes[node]
        if node_info.has_open_group(self.active_group):
            untested_edges = []
            for group_id in range(self.active_group + 1):
                untested_edges.extend(node_info.group2remaining_candidate_ids[group_id])
            if not untested_edges:
                raise ValueError("No untested edges in the current group while the group is open")

            edge_idx = random.choice(untested_edges)
            reasoning = f"Randomly chose untested edge {edge_idx} from group {self.active_group} with {node_info.group2remaining_candidate_ids} group2candidates\n"
        else:
            lowest_dist = node_info.distance
            print(f"Lowest dist: {lowest_dist}")
            # print(f"Node info: {node_info}")
            edges_with_lowest_dist = [edge_idx for edge_idx, edge_data in enumerate(node_info.edge_data) if edge_data["distance"] <= lowest_dist and edge_data["result"] == 1 and edge_data["group"] <= self.active_group]
            edge_idx = random.choice(edges_with_lowest_dist)
            reasoning = f"Chose edge {edge_idx} with lowest dist {lowest_dist}\n"

        reasoning += f"Node info: {node_info}\n"
        

        if return_reasoning:
            return edge_idx, reasoning
        else:
            return edge_idx



def _generate_random_grid(rows: int, cols: int, density: float = 0.7, seed: int | None = None) -> np.ndarray:
    """
    Return a boolean numpy array of shape *(rows, cols)* where **True** denotes
    a traversable cell (graph node) and **False** denotes an empty/wall cell.
    The *density* parameter controls the probability of a cell being present.
    """

    rng = np.random.default_rng(seed)
    grid = rng.random((rows, cols)) < density

    # Safety: ensure at least one node exists so that we have a valid start.
    if not grid.any():
        # Force the central cell to be traversable.
        grid[rows // 2, cols // 2] = True

    return grid


# Direction vectors indexed 0-3  (U, R, D, L)
_DIRS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


def _visualize_grid(grid: np.ndarray, explorer: "GraphExplorer", start_node: tuple[int, int]) -> None:
    """
    Pretty-print the current knowledge stored inside *explorer* on top of the
    underlying *grid*.

    Legend:
        "#"  wall / empty cell
        "?"  traversable cell but undiscovered yet
        "o"  discovered & closed node (all edges tested)
        "F"  frontier node (still holds untested candidates)
        "S"  the start node
    """

    rows, cols = grid.shape
    lines: list[str] = []
    for r in range(rows):
        row_chars: list[str] = []
        for c in range(cols):
            cell = (r, c)
            if not grid[r, c]:
                row_chars.append("#")
                continue

            if cell == start_node:
                row_chars.append("S")
            elif cell in explorer._frontier:
                row_chars.append("F")
            elif cell in explorer._nodes:
                row_chars.append("o")
            else:
                row_chars.append("?")
        lines.append(" ".join(row_chars))

    print("\nCurrent explorer view:")
    print("\n".join(lines))
    print()

def _plot_grid(
    grid: np.ndarray,
    explorer: "GraphExplorer",
    start_node: tuple[int, int],
    last_node: tuple[int, int] | None = None,
    last_edge: tuple[tuple[int, int], int] | None = None,  # (node_coords, edge_idx)
    log_text: str | None = None,
    *,
    figsize: tuple[int, int] | None = None,
    frames: list[np.ndarray] | None = None,
    group_colors: dict[int, str] | None = None,
    n_groups: int = 1,
) -> None:
    """
    Render *grid* with matplotlib showing explorer's knowledge so far.

    - Walls - nothing drawn (white)
    - Undiscovered traversable cells - light grey dots
    - Discovered nodes - blue dots, frontier in orange, start in gold
    - Arrows:
        - Success (edge exists)  - green
        - Failed probe           - red
        - Untested candidate     - grey (thin)
    """

    if n_groups > 1 and group_colors is None:
        default_palette = plt.get_cmap("tab10")
        group_colors = {grp: default_palette(grp % 10) for grp in range(n_groups)}

    rows, cols = grid.shape

    if figsize is None:
        figsize = (max(4, cols), max(4, rows))

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    ax = fig.gca()

    ax.set_aspect("equal")
    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for r in range(rows):
        for c in range(cols):
            # rectangle lower-left corner at (c-0.5, r-0.5)
            facecolor = "black"  # default for walls
            if grid[r, c]:
                cell = (r, c)
                if cell == last_node:
                    facecolor = "blue"
                elif cell in explorer._frontier:
                    facecolor = "green"
                elif cell in explorer._nodes:
                    facecolor = "white"
                else:
                    facecolor = "grey"

            rect = Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                facecolor=facecolor,
                edgecolor="lightgrey",
                linewidth=0.5,
                alpha=0.6,
                zorder=0,
            )
            ax.add_patch(rect)

    # Overlay start marker
    ax.plot(start_node[1], start_node[0], marker="*", color="gold", markersize=12, zorder=4)

    # Draw arrows for each explored node
    for (r, c), info in explorer._nodes.items():
        for edge_idx in range(4):
            dr, dc = _DIRS[edge_idx]

            # Convert to plotting vector (remember inverted y later). Use dy = dr to correct flipped arrow issue.
            dx, dy = dc, dr

            # Decide arrow color & style with fixed length
            length_scale = 0.4  # stays inside cell borders
            succ_flag = False  # will stay False for untested or failed edges

            res = info.edge_data["result"][edge_idx] if edge_idx < len(info.edge_data) else 0
            if res != 0:
                succ_flag = (res == 1)

                # Highlight the very last tested edge in black
                if last_edge is not None and last_edge == ((r, c), edge_idx):
                    color = "black"
                    alpha = 1.0
                    lw = 2.5
                else:
                    color = "green" if succ_flag else "red"  # success green, failed red
                    alpha = 0.9
                    lw = 1.8
            else:
                group_id = int(info.edge_data["group"][edge_idx]) if edge_idx < len(info.edge_data) else 0
                color = group_colors.get(group_id, "grey") if group_colors else "grey"
                alpha = 0.8
                lw = 1.2

            arr = ax.arrow(
                c,
                r,
                dx * length_scale,
                dy * length_scale,
                head_width=0.15,
                head_length=0.15,
                fc=color,
                ec=color,
                alpha=alpha,
                linewidth=lw,
                length_includes_head=True,
                zorder=1,
            )

            # Annotate distance to frontier for successful edges
            if succ_flag:
                # Look up target from explorer graph if exists; otherwise skip distance annotation
                target = None
                for e_idx, tgt in explorer._G.get((r, c), set()):
                    if e_idx == edge_idx:
                        target = tgt
                        break
                if target is not None:
                    dist_val = explorer.get_distance(target)
                    dist_val_txt = "∞" if dist_val is None else str(dist_val)

                    text_x = c + dx * length_scale * 0.5
                    text_y = r + dy * length_scale * 0.5
                    ax.text(text_x, text_y, dist_val_txt, color="black", fontsize=8, ha="center", va="center", zorder=4)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    # Add log text overlay
    if log_text is not None:
        fig.text(0.02, 0.98, log_text, fontsize=9, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    legend_elements = [
        Patch(facecolor="black", edgecolor="lightgrey", label="Wall"),
        Patch(facecolor="grey", edgecolor="lightgrey", label="Unknown node"),
        Patch(facecolor="white", edgecolor="lightgrey", label="Discovered node"),
        Patch(facecolor="green", edgecolor="lightgrey", label="Frontier node"),
        Patch(facecolor="blue", edgecolor="lightgrey", label="Current node"),
        Patch(facecolor="gold", edgecolor="lightgrey", label="Start node"),
        Line2D([0], [0], color="black", lw=2, label="Last tested edge"),
        Line2D([0], [0], color="green", lw=2, label="Successful edge"),
        Line2D([0], [0], color="red", lw=2, label="Failed edge"),
        Line2D([0], [0], color="grey", lw=2, label="Untested candidate"),
    ]

    # Add candidate group colors to legend
    if n_groups > 1:
        for gid in range(n_groups):
            col = group_colors.get(gid, plt.get_cmap("tab10")(gid % 10)) if group_colors else plt.get_cmap("tab10")(gid % 10)
            legend_elements.append(Line2D([0], [0], color=col, lw=2, label=f"Candidate group {gid}"))

    # Reserve more space on the right for legend
    plt.subplots_adjust(right=0.65)

    # Place legend outside, based on figure coords for consistent layout
    legend_obj = fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.68, 0.5),
        bbox_transform=fig.transFigure,
        fontsize=7,
        framealpha=0.9,
    )

    # Optionally increase z-order so legend overlays anything else
    legend_obj.set_zorder(10)

    plt.draw()
    plt.pause(0.001)

    # Capture frame for gif if requested
    if frames is not None:
        canvas = fig.canvas
        canvas.draw()
        w, h = canvas.get_width_height()
        if hasattr(canvas, "tostring_rgb"):
            buf = canvas.tostring_rgb()
            channels = 3
        elif hasattr(canvas, "tostring_argb"):
            buf = canvas.tostring_argb()
            channels = 4
        else:
            raise RuntimeError("Canvas does not support RGB extraction")

        # Account for HiDPI / retina scaling: actual buffer may be larger than (w*h*channels)
        total_px = len(buf) // channels
        scale = int(round((total_px / (w * h)) ** 0.5))
        w_scaled, h_scaled = w * scale, h * scale

        img = np.frombuffer(buf, dtype=np.uint8).reshape(h_scaled, w_scaled, channels)
        if channels == 4:
            # ARGB -> RGB
            img = img[:, :, [1, 2, 3]]
        frames.append(img.copy())


def run_grid_demo(
    rows: int = 6,
    cols: int = 6,
    density: float = 0.7,
    seed: int | None = None,
    step_sleep: float | None = None,
    n_groups: int = 1,
    group_colors: dict[int, str] | None = None,
    plot: bool = True,
    save_gif: bool = True,
    gif_path: str = "exploration.gif",
    error_chance: float = 0.3,
) -> None:
    """
    Drive *GraphExplorer* over a random grid-world and visualize every step.

    - *rows*, *cols*         - grid dimensions
    - *density*              - probability that a cell contains a node
    - *seed*                 - RNG seed for reproducibility (``None`` ⇒ random)
    - *step_sleep*           - optional ``time.sleep`` delay after each step
    """

    import time

    grid = _generate_random_grid(rows, cols, density, seed)

    # Pick a random starting node
    node_coords = list(zip(*np.where(grid)))
    start_node = random.choice(node_coords)

    candidate2group = {i: random.randint(0, n_groups-1) for i in range(4)}

    print(f"Starting exploration at {start_node} on a {rows}x{cols} grid (density={density:.2f})\n")

    gx = GraphExplorer(n_groups=n_groups, verbose_level=2)
    print(f"candidate2group: {candidate2group}\n")
    gx.initialize(start_node=start_node, num_candidates=4, group2remaining_candidate_ids=[{i for i, g in candidate2group.items() if g == gid} for gid in range(n_groups)])

    frames: list[np.ndarray] = [] if plot and save_gif else []

    if plot:
        plt.ion()

    step_counter = 0
    _visualize_grid(grid, gx, start_node)
    if plot:
        _plot_grid(grid, gx, start_node, last_node=start_node, last_edge=None, log_text=f"Group NA | Moved to {start_node}", frames=frames if save_gif else None, n_groups=n_groups, group_colors=group_colors)

        gx.dump()

    current_node = start_node
    while not gx.is_finished():
        node_info = gx._nodes[current_node]

        # If current node is exhausted, travel along the shortest path to the frontier.
        if not node_info.has_open_group(gx.active_group):
            next_hop = gx.get_next_hop(current_node)
            if next_hop is None:
                print(f"Node {current_node} is exhausted and no path to frontier. Finishing.")
                break

            # Guard against degenerate self-looping next-hop
            if next_hop == current_node:
                gx._close_node(current_node)
                gx._maybe_advance_group(current_node)
                next_hop = gx.get_next_hop(current_node)
                if next_hop is None or next_hop == current_node:
                    print(f"Node {current_node} is exhausted and stuck. Finishing.")
                    break

            print(f"Node {current_node} exhausted. Traveling to {next_hop} towards nearest frontier.")
            step_counter += 1
            current_node = next_hop

            # If we arrived at a node that is not open (due to group constraints), try advancing group
            if not gx._nodes[current_node].has_open_group(gx.active_group):
                gx._maybe_advance_group(current_node)

            _visualize_grid(grid, gx, start_node)
            if plot:
                _plot_grid(
                    grid, gx, start_node,
                    last_node=current_node,
                    last_edge=None,
                    log_text=f"Group {gx.active_group} | travel",  
                    frames=frames if save_gif else None,
                    n_groups=n_groups, group_colors=group_colors,
                )

                gx.dump()
                gx.print_all_nodes()
            if step_sleep is not None:
                time.sleep(step_sleep)
                continue

        # We are at a node with open edges. Try them until success.
        group_id = gx.active_group
        prioritized_edges = []
        for gid in range(0, group_id + 1):
            prioritized_edges.extend(list(node_info.group2remaining_candidate_ids[gid]))

        moved = False
        for edge_idx in prioritized_edges:
            step_counter += 1

            dr, dc = _DIRS[edge_idx]
            neigh = (current_node[0] + dr, current_node[1] + dc)

            is_success = 0 <= neigh[0] < rows and 0 <= neigh[1] < cols and grid[neigh]

            if error_chance > random.random():
                result_code = -1
            else:
                result_code = 1 if is_success else 0

            # Record test result
            outcome_str = "fail"
            if result_code == 1:
                outcome_str = "success"
                target_group2remaining_candidate_ids = [set() for _ in range(n_groups)]
                for i in range(4):
                    gid = random.randint(0, n_groups - 1)
                    target_group2remaining_candidate_ids[gid].add(i)
                gx.record_test(current_node, edge_idx, 1, neigh, 4, group2remaining_candidate_ids=target_group2remaining_candidate_ids)
            elif result_code == 0:
                gx.record_test(current_node, edge_idx, 0)
            else:  # result_code == -1
                outcome_str = "error"
                gx.record_test(current_node, edge_idx, -1)

            print(f"Step {step_counter}: at {current_node} tested edge {edge_idx} → {outcome_str}")

            edge_group_id = int(node_info.edge_data["group"][edge_idx]) if edge_idx < len(node_info.edge_data) else 0
            cur_dist = gx.get_distance(current_node)
            dist_txt = "∞" if cur_dist is None else str(cur_dist)
            log_line = (
                f"group={gx.active_group} node={current_node} (dist {dist_txt}) | "
                f"edge {edge_idx} (grp {edge_group_id}) → {outcome_str}"
            )
            _visualize_grid(grid, gx, start_node)
            if plot:
                _plot_grid(
                    grid, gx, start_node,
                    last_node=current_node,
                    last_edge=((current_node), edge_idx),
                    log_text=log_line,
                    frames=frames if save_gif else None,
                    n_groups=n_groups, group_colors=group_colors,
                )

                gx.dump()
                gx.print_all_nodes()
            if step_sleep is not None:
                time.sleep(step_sleep)

            # Update agent position based on outcome
            if result_code == 1:
                current_node = neigh
                moved = True
                break
            elif result_code == -1:
                print(f"Probe error at {current_node}! Returning to start node {start_node}.")
                current_node = start_node
                moved = True
                break

        if not moved:
            # All available edges were tried and failed/errored.
            # Next loop iteration will trigger the travel-to-frontier logic.
            pass

    print("Exploration finished – every node is closed and no frontier remains.")
    if plot:
        # Final frame with no current node highlight
        _plot_grid(
            grid,
            gx,
            start_node,
            last_node=None,
            frames=frames if save_gif else None,
            n_groups=n_groups,
            group_colors=group_colors,
        )

        # Keep the final plot open for the user until they close the figure.
        plt.ioff()

        if save_gif and frames:
            print(f"Saving cropped GIF with {len(frames)} frames to {gif_path} …")

            from PIL import Image, ImageChops

            pil_frames = [Image.fromarray(frame) for frame in frames]

            # Compute union bounding box of non-white areas across frames
            bbox_union = None
            white_bg = Image.new("RGB", pil_frames[0].size, (255, 255, 255))
            for im in pil_frames:
                diff = ImageChops.difference(im, white_bg)
                bbox = diff.getbbox()
                if bbox is None:
                    continue
                if bbox_union is None:
                    bbox_union = bbox
                else:
                    l1, t1, r1, b1 = bbox_union
                    l2, t2, r2, b2 = bbox
                    bbox_union = (min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2))

            # Fallback to full image if bbox detection failed
            if bbox_union is None:
                bbox_union = (0, 0) + pil_frames[0].size

            cropped_frames = [im.crop(bbox_union) for im in pil_frames]

            # Save using Pillow directly
            cropped_frames[0].save(
                gif_path,
                save_all=True,
                append_images=cropped_frames[1:],
                duration=500,
                loop=0,
            )

        plt.show()


if __name__ == "__main__":

    print("\n========== SIMPLE TEST ==========")
    gx = GraphExplorer(verbose_level=2)
        
    gx.initialize("A", 2) # node A has 2 candidates

    gx.record_test("A", 0, -1) # simulate error
    gx.record_test("A", 0, -1) # simulate error
    gx.record_test("A", 0, -1) # simulate error

    # gx.record_test("A", 0, True,  "B", 3)   # throws an error

    gx.record_test("A", 1, True, "B", 3) # now A is closed automatically


    gx.record_test("B", 0, False)
    gx.record_test("B", 1, True,  "C", 1) # discovers C 
    gx.record_test("B", 2, False) # B becomes closed

    gx.record_test("C", 0, True, "D", 4)

    gx.print_all_nodes()


    print("\n========== TEST WITH GROUPS ==========")
    gx = GraphExplorer(n_groups=3, verbose_level=2)
    gx.initialize("A", 4, group2remaining_candidate_ids=[[0, 1], [2], [3]])

    gx.record_test("A", 0, False)
    gx.record_test("A", 1, False)

    gx.record_test("A", 2, True, "B", 3, group2remaining_candidate_ids=[[0], [2], [1]])

    gx.record_test("B", 0, True, "A")

    gx.record_test("A", 2, True, "B")

    gx.record_test("B", 2, False)

    gx.print_all_nodes()


    print("\n========== GRID WORLD DEMO ==========")
    group_colors = {0: "purple", 1: "orange", 2: "grey"}
    run_grid_demo(rows=6, cols=6, density=0.7, seed=12345, step_sleep=None, plot=True, save_gif=True, gif_path="grid_exploration.gif", n_groups=3, group_colors=group_colors, error_chance=0.1)
