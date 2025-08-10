from ast import Or
from agents.agent import Agent
from agents.tracing import trace_agent_session
from agents.structs import FrameData, GameAction, GameState
from graph_explorer import GraphExplorer
import logging
# import os
# import json
# import textwrap
import numpy as np

from typing import Any
import random
import time
from collections import deque

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    pass
import hashlib

logger = logging.getLogger()

class HeuristicAgent(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 1000000

    COLOR_MAP: dict[int, list[int]] = {
    0: [255, 255, 255],
    1: [204, 204, 204], # NOTE: a solid gues
    2: [153, 153, 153], # light gray
    3: [102, 102, 102], # gray
    4: [51, 51, 51], # dark gray
    5: [0, 0, 0], # black
    6: [255, 0, 0], # NOTE: 'not known'
    7: [0, 255, 0], # NOTE: 'not known' maybe from 10 or from 13
    8: [250, 61, 50], # red
    9: [31, 147, 255], # blue
    10: [137, 216, 241], # NOTE: 'GUESS' light blue? game 2 level 7
    11: [255, 221, 0], # yellow
    12: [255, 133, 26], # orange
    13: [229, 58, 163], # NOTE: 'GUESS'  pink? game 2 level 7
    14: [79, 205, 48], # green
    15: [163, 86, 214] # purple # ???
    }

    DEBUG_PLOTS: bool = False
    DEBUG_PRINTS: bool = False

    SIMPLE_ACTION_ID2GAME_ACTION = {
                1: GameAction.ACTION1,
                2: GameAction.ACTION2,
                3: GameAction.ACTION3,
                4: GameAction.ACTION4,
                5: GameAction.ACTION5,
            }


    N_GROUPS: int = 5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

        self.frame_processor = FrameProcessor()

        self.status_bar_mask = None # np.ndarray | None

        self.hashed_frame2action_results = {}
        self.hashed_frame2transitions = {}

        self.last_hashed_frame = None
        # self.last_segment_to_click = None
        self.last_action = None # int | None

        self.arrow_control = True

        self.favor_new_actions = False

        self.favor_frontier_search = True

        self.verbose_level = 0

        self.graph_explorer = GraphExplorer(verbose_level=self.verbose_level, n_groups=self.N_GROUPS)

        self.level_first_frame = None




    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment below to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def visualize_last_frame(self, frames: list[FrameData]) -> None:
        non_empty_frames = [frame for frame in frames if len(frame.frame) > 0]
        if len(non_empty_frames) > 0:
            non_empty_frames = non_empty_frames[-1:]
            frames_np = np.concatenate([frame.frame for frame in non_empty_frames])
            print(frames_np)
            for frame_idx, frame in enumerate(frames_np):
                # without GUI
                plt.figure(figsize=(10, 10))
                # write the values of cell to the image
                for i in range(frame.shape[0]):
                    for j in range(frame.shape[1]):
                        plt.text(j, i, frame[i, j], ha="center", va="center", color="white")
                frame_rgb = np.array([self.COLOR_MAP[cell] for cell in frame.flatten()]).reshape(frame.shape[0], frame.shape[1], 3)
                print(frame_rgb)
                # save frame as numpy array
                np.save(f"frame_{len(frames)}_{self.game_id}.npy", frame)
                plt.imshow(frame_rgb)
                plt.colorbar()
                plt.savefig(f"frame_{len(frames)}_{self.game_id}.png", format="PNG")
                plt.close()

    def visualize_connected_components(self, frame: np.ndarray, segmented_frame: np.ndarray, frame_segments: list[dict], frame_number: int | None = None) -> None:
        print(frame_segments)
        self.frame_processor.visualize_components(frame, frame_segments, save_path=f"frame_segmented_{frame_number}_{self.game_id}.png")

        status_bar_segments_list, status_bar_mask = self.frame_processor.identify_status_bars(segmented_frame, frame_segments)
        plt.imshow(status_bar_mask)
        plt.savefig(f"status_bar_mask_{frame_number}_{self.game_id}.png", format="PNG")
        plt.close()

    def get_frame_transition_data(self, hashed_frame: str, num_actions: int) -> tuple[np.ndarray, list[int]]:
        curr_frame_action_results = self.hashed_frame2action_results.get(hashed_frame, None)
        if curr_frame_action_results is None:
            self.hashed_frame2action_results[hashed_frame] = np.zeros(num_actions) # NOTE: 0 means not clicked, -1 no transition, 1 transition
            curr_frame_action_results = self.hashed_frame2action_results[hashed_frame]
        # TODO: rename to something clearer

        curr_frame_transitions = self.hashed_frame2transitions.get(hashed_frame, None)
        if curr_frame_transitions is None:
            self.hashed_frame2transitions[hashed_frame] = [0] * num_actions
            curr_frame_transitions = self.hashed_frame2transitions[hashed_frame]

        return curr_frame_action_results, curr_frame_transitions


    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData, level_up: bool = False
    ) -> GameAction:

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout # TODO: remove this
            action = GameAction.RESET
            self.last_hashed_frame = None
            self.last_action = None
            return action

        if self.DEBUG_PLOTS: self.visualize_last_frame(frames)
        latest_frame_np = np.array(latest_frame.frame, dtype=np.uint8)
    
        if latest_frame_np.size > 0:
            num_frames = latest_frame_np.shape[0]
            latest_frame_np = latest_frame_np[-1]
            # NOTE: sometimes there are multiple frames, we take the last one
            if self.DEBUG_PRINTS: print(latest_frame_np.shape)

            if level_up:
                segmented_frame_for_status_bars, frame_segments_for_status_bars = self.frame_processor.segment_frame(latest_frame_np)
                status_bar_segments_list, status_bar_mask = self.frame_processor.identify_status_bars(segmented_frame_for_status_bars, frame_segments_for_status_bars)
                self.status_bar_mask = status_bar_mask

                self.hashed_frame2action_results = {}
                self.hashed_frame2transitions = {}

            latest_frame_np[self.status_bar_mask] = 16 # Or use 0 / 5 ? 
            segmented_frame, frame_segments = self.frame_processor.segment_frame(latest_frame_np)
            available_actions = latest_frame.available_actions

            num_arrow_actions = 0
            num_click_actions = 0
            num_actions = 0
            arrow_actions = []
            if 6 in available_actions:
                num_actions += len(frame_segments)
                num_click_actions += len(frame_segments)
                action_groups = self.frame_processor.frame_segments_to_action_groups(frame_segments, n_groups=self.N_GROUPS)
            else:
                action_groups = [set() for _ in range(self.N_GROUPS)]

            for action_id in available_actions:
                if action_id in self.SIMPLE_ACTION_ID2GAME_ACTION:

                    arrow_actions.append(self.SIMPLE_ACTION_ID2GAME_ACTION[action_id])
                    action_groups[0].add(num_actions)
                    num_actions += 1
                    num_arrow_actions += 1
            # if not self.arrow_control:
            #     num_actions = len(frame_segments)
            #     action_groups = []
            # else:
            #     num_actions = 5 # FIXME: remove magic number

            # NOTE: make status bars do not affect the pixel state

            if self.DEBUG_PLOTS: self.visualize_connected_components(latest_frame_np, segmented_frame, frame_segments, frame_number=len(frames))

            latest_frame_np[latest_frame_np == 16] = 0 # to ensure there is no overflow 
            hashed_frame = self.frame_processor.hash_frame(latest_frame_np)

            if level_up:
                self.level_first_frame = hashed_frame
                self.graph_explorer.reset()
                self.graph_explorer.initialize(start_node=hashed_frame, num_candidates=num_actions, group2remaining_candidate_ids=action_groups)
                self.graph_explorer.dump()
                # TODO: Add a function to assign candidate groups!
            

            if self.last_hashed_frame is not None and not level_up:
                transition = hashed_frame != self.last_hashed_frame 
                suspicious_transition = hashed_frame == self.level_first_frame and num_frames > 1
                # if num_frames > 1:
                #     print('Almost suspicious transition!')

                # if not self.last_hashed_frame in self.hashed_frame2action_results:
                #     self.hashed_frame2action_results[self.last_hashed_frame] = np.zeros(len(frame_segments))

                # if not self.last_hashed_frame in self.hashed_frame2transitions:
                #     self.hashed_frame2transitions[self.last_hashed_frame] = None

                old_value = self.hashed_frame2action_results[self.last_hashed_frame][self.last_action]

                if transition:
                    self.hashed_frame2action_results[self.last_hashed_frame][self.last_action] = 1
                    self.hashed_frame2transitions[self.last_hashed_frame][self.last_action] = hashed_frame
                else:
                    self.hashed_frame2action_results[self.last_hashed_frame][self.last_action] = -1
                    self.hashed_frame2transitions[self.last_hashed_frame][self.last_action] = None
                
                self.graph_explorer.record_test(self.last_hashed_frame, self.last_action, transition, hashed_frame, 
                target_num_candidates=num_actions,
                group2remaining_candidate_ids=action_groups,
                suspicious_transition=suspicious_transition
                )
                self.graph_explorer.dump()
                # TODO: Add a function to assign candidate groups!

                if old_value != 0 and old_value != self.hashed_frame2action_results[self.last_hashed_frame][self.last_action]:
                    raise ValueError(f'Old value {old_value} is not equal to new value {self.hashed_frame2action_results[self.last_hashed_frame][self.last_action]}')

            new_frame = hashed_frame not in self.hashed_frame2action_results
            curr_frame_action_results, curr_frame_transitions = self.get_frame_transition_data(hashed_frame, num_actions)
            if new_frame:
                if self.verbose_level >= 1:
                    print(f'NEW FRAME:', end='\t')

            if hashed_frame not in self.graph_explorer._nodes:
                print('Here we are!')
                self.graph_explorer.record_test(self.last_hashed_frame, self.last_action, transition, hashed_frame, 
                target_num_candidates=num_actions,
                group2remaining_candidate_ids=action_groups,
                suspicious_transition=suspicious_transition
                )
            
            if self.verbose_level >= 1:
                print(self.graph_explorer._nodes[hashed_frame])
            # print(f'{hashed_frame} - actions: {curr_frame_action_results}')

            available_actions = np.where(curr_frame_action_results != -1)[0]
            # if len(available_actions) == 0:
            #     print(f'Switching control interface from arrow={self.arrow_control} to arrow={not self.arrow_control}') # TODO: switch to logging
            #     self.arrow_control = not self.arrow_control
            #     self.hashed_frame2action_results = {}
            #     self.hashed_frame2transitions = {}
            #     if not self.arrow_control:
            #         num_actions = len(frame_segments)
            #     else:
            #         num_actions = 5 # FIXME: remove magic number
            #     curr_frame_action_results, curr_frame_transitions = self.get_frame_transition_data(hashed_frame, num_actions)

            #     if self.favor_frontier_search:
            #         self.graph_explorer.reset()
            #         self.graph_explorer.initialize(start_node=hashed_frame, num_candidates=num_actions)

                    
        
            # available_actions = np.where(curr_frame_action_results != -1)[0]
            
            new_actions = np.where(curr_frame_action_results == 0)[0]

            if len(available_actions) == 0:
                raise ValueError(f'No available actions found for frame {hashed_frame}')
                # TODO: maybe fully random fallback?
            else:
                reasoning = ""
                if self.favor_frontier_search:
                    # self.graph_explorer.print_all_nodes()
                    action_id, reasoning = self.graph_explorer.choose_edge(hashed_frame, return_reasoning=True)

                elif len(new_actions) > 0 and self.favor_new_actions:
                    action_id = random.choice(new_actions)
                else:
                    action_id = random.choice(available_actions)

                if action_id < num_click_actions:
                    arrow_control = False
                else:
                    arrow_control = True
                    # action_id -= num_click_actions
                    # action_id = arrow_actions[action_id] # HACK: ugly way to ensure that the action_id reflects GameAction


            # TODO: keep the order of the segments as it was before 
            # TODO: change the weight of the segments based on  whether they 1) worked before, (on other frames too?) 2) in a status bar, 3) of salient color
            if not arrow_control:
                segment_mask = segmented_frame == action_id
                segment_points = np.argwhere(segment_mask)
                if self.DEBUG_PRINTS: print(segment_points)
                segment_point = segment_points[random.randint(0, len(segment_points) - 1)]
                y, x = segment_point

                # self.frame_processor.visualize_components(latest_frame_np, frame_segments, click_points=[(x, y)], save_path=f"click_point_{self.game_id}_{hashed_frame}_action:{action_id}_step:{self.action_counter}.png")

                action = GameAction.ACTION6
                action.set_data(
                    {"x": x, "y": y}
                )
   
                reasoning += f"\nClicking on a segment {action_id}-{frame_segments[action_id]}, x: {x}, y: {y}" \
                    f"\nAction results: {curr_frame_action_results} for frame {hashed_frame}"

                action.reasoning = {
                    "desired_action": f"{action.value}",
                    "my_reason": reasoning,
                }
            else:
                action = arrow_actions[action_id - num_click_actions]

                reasoning += f"Arrow control: {action} for frame {hashed_frame}" \
                    f"\nAction results: {curr_frame_action_results} for frame {hashed_frame}"

                action.reasoning = {
                    "desired_action": f"{action.value}",
                    "my_reason": reasoning,
                }
                
            self.last_hashed_frame = hashed_frame
            self.last_action = action_id

            if self.verbose_level >= 1:
                print(action.reasoning)

            return action

        # In case of empty frame, we perform a full random action

        """Choose which action the Agent should take, fill in any arguments, and return it."""

        # logging.getLogger("openai").setLevel(logging.CRITICAL)
        # logging.getLogger("httpx").setLevel(logging.CRITICAL)

        """Choose which action the Agent should take, fill in any arguments, and return it."""
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout
            action = GameAction.RESET
        else:
            # else choose a random action that isnt reset
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        if action.is_simple():
            action.reasoning = f"RNG told me to pick {action.value}"
        elif action.is_complex():
            action.set_data(
                {
                    "x": random.randint(0, 63),
                    "y": random.randint(0, 63),
                }
            )
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": "RNG said so!",
            }
        return action

        # # now ask for the next action
        # action_id = random.choice(list(GameAction))
        # action = GameAction.from_name(action_id)
        # logger.info(f"Action: {action}")
        # # action.set_data(data)
        # return action

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    @trace_agent_session
    def main(self) -> None:
        """The main agent loop. Play the game_id until finished, then exits."""
        self.timer = time.time()
        score = 0
        level_up = True
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
            # TODO: add some limits for levels, after which we transition to another method
        ):
            action = self.choose_action(self.frames, self.frames[-1], level_up=level_up)
            if frame := self.take_action(action): # NOTE: What does ":=" do?
                new_score = frame.score
                if new_score > score:
                    level_up = True
                    self.status_bar_mask = None
                elif self.status_bar_mask is not None:
                    level_up = False
                score = new_score
                self.append_frame(frame) # NOTE: Where do we append the frame?
                logger.info(
                    f"{self.game_id} - {action.name}: count {self.action_counter}, score {frame.score}, avg fps {self.fps})"
                )
            self.action_counter += 1

        self.cleanup()

class FrameProcessor:
    OFFSETS4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
    OFFSETS8: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self):
        self.connectivity_rank = 4
        self.status_bar_mode = "rule"
        self.status_bar_distance_threshold = 3
        self.status_bar_ratio_threshold = 5
        self.status_bar_twins_threshold = 3
        self.frame_shape = (64, 64)

        self.status_bar_color = 16
        self.minimal_width = 2
        self.maximal_width = 32
        self.non_salient_color = set([0,1,2,3,4,5])
        self.salient_color = set([6,7,8,9,10,11,12,13,14,15])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        pass

    def segment_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Segment `frame` into {self.connectivity_rank}-connected components (same color).

        NOTE: the twins identification increases complexity of the algorithm to O(n^2)

        Returns
        -------
        list[dict]
            One dict per component with keys
            - bounding_box : (x1, y1, x2, y2)   # inclusive pixel coords
            - color        : int                # original greyscale value
            - area         : int                # pixel count
            - is_rectangle : bool               # fully fills its bounding box
            - number_of_twins : int             # number of other components considered twins
            - twin_ids     : list[int]          # ids (1-based) of those twins
                NOTE: here we don't check shapes of the twins thoroughly

        """

        h, w = frame.shape
        label_map = np.zeros((h, w), dtype=int) - 1 # -1 = unvisited
        components: list[dict] = []
        cid = -1                                          # component id counter

        offsets = self.OFFSETS4 if self.connectivity_rank == 4 else self.OFFSETS8

        # --- first pass: flood-fill each blob ---------------------------------
        for y in range(h):
            for x in range(w):
                if label_map[y, x] != -1:                      # already labelled
                    continue
                cid += 1
                color = int(frame[y, x])
                q = deque([(y, x)])
                label_map[y, x] = cid

                min_x = max_x = x
                min_y = max_y = y
                area = 0

                while q:                                 # BFS
                    cy, cx = q.popleft()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)

                    for dy, dx in offsets:
                        ny, nx = cy + dy, cx + dx
                        if (
                            0 <= ny < h and 0 <= nx < w
                            and label_map[ny, nx] == -1 # not visited
                            and frame[ny, nx] == color
                        ):
                            label_map[ny, nx] = cid
                            q.append((ny, nx))

                # rectangle test
                rect_area = (max_x - min_x + 1) * (max_y - min_y + 1)
                is_rect = area == rect_area

                components.append(
                    dict(
                        bounding_box=(min_x, min_y, max_x, max_y),
                        color=color,
                        area=area,
                        is_rectangle=is_rect,
                    )
                )

        # --- second pass: identify twins --------------------------------------
        # here: simple rule → same area, same rectangle status, and same color
        for i, comp in enumerate(components):
            twins = [
                j
                for j, other in enumerate(components)
                if i != j # skip self
                and other["area"] == comp["area"]
                and other["is_rectangle"] == comp["is_rectangle"]
                and other["color"] == comp["color"]
            ]
            comp["number_of_twins"] = len(twins)
            comp["twin_ids"] = twins

        return label_map, components

    def identify_status_bars(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]] | None, np.ndarray]:
        """
        Identify the status bars from the frame segments
        Return a list of dictionaries and a frame mask.
        The list of dictionaries is the same as the input list of dictionaries in frame_segments, but with "id" key added.
        The frame mask is a binary mask where the status bars are 1 and the rest are 0.
        """
        if self.status_bar_mode == "crude":
            status_bar_mask = self.identify_status_bars_crude()
            status_bar_segments_list = None
        elif self.status_bar_mode == "rule" or self.status_bar_mode == "move":
            status_bar_segments_list, status_bar_mask = self.identify_status_bars_with_rule(segmented_frame, frame_segments)
            if self.status_bar_mode == "move":
                raise NotImplementedError("'move' mode is not implemented yet")
        else:
            raise ValueError(f"Invalid status bar mode: {self.status_bar_mode}")
        return status_bar_segments_list, status_bar_mask

    def identify_status_bars_crude(self) -> np.ndarray:
        status_bar_mask = np.zeros(self.frame_shape)
        status_bar_mask[:self.status_bar_distance_threshold, :] = 1
        status_bar_mask[-self.status_bar_distance_threshold:, :] = 1
        status_bar_mask[:, :self.status_bar_distance_threshold] = 1
        status_bar_mask[:, -self.status_bar_distance_threshold:] = 1
        return status_bar_mask
       
    def identify_status_bars_with_rule(self, segmented_frame: np.ndarray, frame_segments: list[dict]) -> tuple[list[list[dict]], np.ndarray]:
        """
        Identify the status bars from the frame segments
        Return a list of dictionaries and a frame mask.
        The list of dictionaries is the same as the input list of dictionaries in frame_segments, but with "id" key added.
        The frame mask is a binary mask where the status bars are 1 and the rest are 0.
        """

        # modes:
            # crude: remove all screen edges 
            # rule: rule-based
            # move: rule-based + movement after the first action 


        # the rules are:
            # the status bars are close to the edges of the screen
            # they can be in any orientation
            # the can be duplicated from both sides of the screen
            # there are 2 types of status bars:
                # 1. the line 
                # 2. the dots, for the dots there should be at least 3 twins


        checked_segment_ids = set()
        status_bar_segment_ids_list = [] # list[list[int]]
        for i, segment in enumerate(frame_segments):

            status_bar_segment_ids = [i]

            if i in checked_segment_ids:
                continue
            checked_segment_ids.add(i)
            on_edge_list = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(on_edge_list) == 0:
                continue
            directions = []
            if 'left' in on_edge_list or 'right' in on_edge_list:
                directions.append('vertical')
            if 'top' in on_edge_list or 'bottom' in on_edge_list:
                directions.append('horizontal')
            if len(directions) == 2:
                direction = 'any'
            else:
                direction = directions[0]
            is_long_ratio = self.check_segment_ratio(segment, direction=direction)  

            if not is_long_ratio:
                twin_ids_on_edge_list = self.segment_twins_on_edge(segment, frame_segments)
                for twin_id in twin_ids_on_edge_list:
                    checked_segment_ids.add(twin_id)
                if len(twin_ids_on_edge_list) + 1 < self.status_bar_twins_threshold:
                    continue
                status_bar_segment_ids.extend(twin_ids_on_edge_list)

            status_bar_segment_ids_list.append(status_bar_segment_ids)

        status_bar_segments_list = []
        status_bar_mask = np.zeros(segmented_frame.shape, dtype=bool)

        for i, status_bar_segment_ids in enumerate(status_bar_segment_ids_list):
            status_bar_segments = []
            for status_bar_segment_id in status_bar_segment_ids:
                status_bar_mask[segmented_frame == status_bar_segment_id] = 1

                status_bar_segments.append(frame_segments[status_bar_segment_id])
            status_bar_segments_list.append(status_bar_segments)

        return status_bar_segments_list, status_bar_mask

    def check_segment_fully_on_edge(self, segment: dict, edges: list[str] | None = None) -> list[str]:
        """
        Check if the segment is fully on the edge of the screen
        """
        x1, y1, x2, y2 = segment["bounding_box"]
        if edges is None:
            edges = ['any']
        for edge in edges:
            assert edge in ['any', 'left', 'right', 'top', 'bottom']

        result = []

        if 'left' in edges or 'any' in edges:
            max_x = max(x1, x2)
            if max_x < self.status_bar_distance_threshold:
                result.append('left')
        if 'right' in edges or 'any' in edges:
            min_x = min(x1, x2)
            if min_x > self.frame_shape[1] - self.status_bar_distance_threshold:
                result.append('right')
        if 'top' in edges or 'any' in edges:
            max_y = max(y1, y2)
            if max_y < self.status_bar_distance_threshold:
                result.append('top')
        if 'bottom' in edges or 'any' in edges:
            min_y = min(y1, y2)
            if min_y > self.frame_shape[0] - self.status_bar_distance_threshold:
                result.append('bottom')
        # NOTE: there can be some mess with the y-axis direction (should it start from the top or the bottom), need to double check
        return result

    def check_segment_ratio(self, segment: dict, direction: str | None = None) -> bool:
        """
        Check if the segment is a status bar
        """
        if direction is None:
            direction = 'any'
        assert direction in ['any', 'horizontal', 'vertical']

        x_length, y_length = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
        x_to_y_ratio = x_length / y_length
        if x_to_y_ratio >= self.status_bar_ratio_threshold and direction in ('any', 'horizontal'):
            return True
        if x_to_y_ratio <= 1 / self.status_bar_ratio_threshold and direction in ('any', 'vertical'):
            return True
        return False

    def segment_twins_on_edge(self, segment: dict, frame_segments: list[dict], edges: list[str] | None = None) -> list[int]:
        """
        Check if the segment has twins on the same edge
        """

        if edges is None:
            edges = self.check_segment_fully_on_edge(segment, edges=['any'])
            if len(edges) == 0:
                return []

        twins = []
        for twin_id in segment["twin_ids"]:
            twin = frame_segments[twin_id]
            twin_edges = self.check_segment_fully_on_edge(twin, edges=edges)
            if len(twin_edges) > 0:
                twins.append(twin_id)
        
        return twins
        
    def visualize_components(self, frame: np.ndarray, components: list[dict], *, cmap: str = "nipy_spectral",
                             save_path: str = "components.png", click_points: list[tuple[int, int]] | None = None
    ) -> None:
        """
        Show the frame with every connected component marked and
        print a short description for each one.

        Parameters
        ----------
        frame : np.ndarray
            The original HxW greyscale (label-value) image.
        components : list[dict]
            Output of `segment_frame()`.
        cmap : str, optional
            Matplotlib colour map for the background image.  Default is *nipy_spectral*.
        """
        if frame.ndim != 2:
            raise ValueError("`frame` must be a 2-D array")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(frame, cmap=cmap, interpolation="nearest")
        ax.set_axis_off()

        # Plot bounding box + id at the centroid of each blob
        for idx, comp in enumerate(components, start=1):
            x1, y1, x2, y2 = comp["bounding_box"]
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # draw bounding box
            ax.add_patch(
                Rectangle(
                    (x1 - 0.5, y1 - 0.5),
                    w,
                    h,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=1.2,
                )
            )

            # annotate with id number
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            ax.text(
                cx,
                cy,
                str(idx),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, lw=0
                ),
            )

        if click_points is not None:
            for x, y in click_points:
                ax.plot(x, y, 'ro')

        plt.tight_layout()
        plt.savefig(save_path)

        # ---------------------------------------------------------------------
        # Console description
        # ---------------------------------------------------------------------
        for idx, comp in enumerate(components, start=1):
            bb = comp["bounding_box"]
            print(
                f"Component {idx}: "
                f"colour={comp['color']:>2}, "
                f"area={comp['area']:>4}, "
                f"bbox=(x1={bb[0]}, y1={bb[1]}, x2={bb[2]}, y2={bb[3]}), "
                f"rect={comp['is_rectangle']}, "
                f"twins={comp['number_of_twins']} "
                f"{'('+','.join(map(str,comp['twin_ids']))+')' if comp['twin_ids'] else ''}"
            )
    
    def hash_frame(self, frame: np.ndarray) -> str:
        """
        Deterministic 128-bit hash for an integer-valued NumPy array whose
        elements are in the range 0 … 15 (4 bits).

        • Compact: packs two elements per byte before hashing  
        • Stable: identical digest across Python versions & interpreter restarts  
        • Shape-aware: (m, n) and (n, m) views do NOT collide  
        • Dependency-free: only stdlib hashlib
        """
        # TODO: maybe just convert a matrix to a number and store it
        frame = np.asarray(frame, dtype=np.uint8, order='C')

        # ---- pack two 4-bit values into each byte ---------------------------
        flat = frame.ravel()
        if flat.size & 1:                       # pad to even length
            flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
        packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
        payload = packed.tobytes()

        # ---- hash with Blake2B (128-bit digest) -----------------------------
        shape_tag = frame.shape.__repr__().encode()
        return hashlib.blake2b(payload,
                            digest_size=16,   # 128 bits
                            person=shape_tag  # embeds the shape
                            ).hexdigest()


    def frame_segments_to_action_groups(self, frame_segments: list[dict], n_groups: int) -> list[list[int]]:
        """
        Assign actions to groups
        """
        group_0_segments = set()
        group_1_segments = set()
        group_2_segments = set()
        group_3_segments = set()
        group_4_segments = set()

        for segment_id, segment in enumerate(frame_segments):
            x_width, y_width = segment["bounding_box"][2] - segment["bounding_box"][0] + 1, segment["bounding_box"][3] - segment["bounding_box"][1] + 1
            is_salient = segment["color"] in self.salient_color
            is_medium_width = self.minimal_width <= x_width <= self.maximal_width and self.minimal_width <= y_width <= self.maximal_width
            is_status_bar = segment["color"] == self.status_bar_color

            assert n_groups == 5, "Only 5 groups are supported for now"

            if is_salient and is_medium_width:
                group_0_segments.add(segment_id)
            elif is_medium_width:
                group_1_segments.add(segment_id)
            elif is_salient:
                group_2_segments.add(segment_id)
            elif not is_status_bar:
                group_3_segments.add(segment_id)
            else:
                group_4_segments.add(segment_id)

        groups2segments = [group_0_segments, group_1_segments, group_2_segments, group_3_segments, group_4_segments]
        # groups2segments = groups2segments[::-1] # NOTE: temporary to check the robustness 

        return groups2segments



# FIXME: hash keyerror when level_up
# TODO: check how hash decision-making generally works

# TODO then: add some value propagation with transitions

# TODO: switch strategies on resets, e.g.:
# - random action selection
# - favor new actions


# TODO: for an action that resulted in a game over, save that it creates a transition, but the frame should be `0`. And then maybe treat it as a basic transition?
# Hmm, but the distance should be indified or set to constant?