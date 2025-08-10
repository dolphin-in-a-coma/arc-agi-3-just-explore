# Explore It Till You Solve It
Exploration-only solution for ARC-AGI-3

## Quickstart 
This repository was originally forked from [the challenge repo](https://github.com/arcprize/ARC-AGI-3-Agents). The setup mostly mirrors the original one. Alternatively, you can run the code in [Google Colab](https://colab.research.google.com/github/dolphin-in-a-coma/arc-agi-3-just-explore/blob/main/ARC_AGI_3_Solve_by_Exploration.ipynb).

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.

1. Clone this repository and enter the directory.

```bash
git clone https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore.git
cd arc-agi-3-just-explore
```

2. Copy `.env.example` to `.env`.

```bash
cp .env.example .env
```

3. Get an API key from the [ARC-AGI-3 Website](https://three.arcprize.org/) and set it in your `.env` file.

```bash
export ARC_API_KEY="your_api_key_here"
```

4. Run the agent. The command below runs the swarm across all games unless a specific game is provided with `--game`.

```bash
uv run main.py --agent=heuristicagent
```

For more information, see the [original documentation](https://three.arcprize.org/docs#quick-start) or the [tutorial video](https://youtu.be/xEVg9dcJMkw).

## The method

### Motivation
The initial idea was to make LLMs interact with the environment more effectively by:
- Providing a textual description of the environment.
- Introducing meaningful click actions (e.g., click an object instead of raw coordinates).
- Building a replay buffer for in-context reinforcement learning.

After experiments on simple levels (passing a winning path from a previous level and providing a list of clickable objects), this direction ended up less promising. In parallel, a brute-force exploration method emerged and performed better for the public tasks.

### Description
The method has two parts:
- Frame Processor
- Level Graph Explorer

#### Frame Processor
Basic image processing aims to reduce irrelevant visual variability and focus exploration on actionable regions. It's done by:
- Segmenting the frame into single-color connected components.
- Detecting and masking likely status bars (e.g., remaining steps).
- For click-controlled games, grouping segments into five priority tiers based on button likelihood (average size, salient color; lowest tier includes segments likely to be status bars).
- Hashing the masked image for use by the graph explorer.

#### Level Graph Explorer
From each known frame (graph node), the explorer maintains paths to frontier frames—those with untested actions (graph edges). For each frame, it tracks:
- The list of possible actions (clicks for `ft09`/`cv33`, arrows for `ls20`).
- For each action: priority level, tested flag, transition result, destination frame, and distance to the nearest frontier.

Actions are taken from the highest-priority group with remaining untested actions; only when all such actions are exhausted across the graph do we proceed to lower-priority groups. Some utility functions are duplicated, and distances are recomputed more often than necessary - this can be cleaned up.

### Thoughts
This is a limited but effective approach that approaches the limits of brute-force solving for these games. The goal is simply to be more intelligent than a purely random agent.

It can be tricked if the status bar differs significantly from the public games (e.g., integrated into the scene rather than at the edge). In such cases, the method degrades toward more random exploration because the state space implicitly includes many status bar variants. Additionally, large state spaces (e.g., `ft09` levels 3–4) can make the method intractable. Non-determinism or partial observations can also cause issues.

A natural extension would be to learn simple world models that predict the next frame from the current frame and action. This could improve sample efficiency by roughly the average number of actions per frame. However, it’s unclear whether such models would help prioritize exploration of “interesting” states in these games by favoring higher uncertainty or surprise for the agent. For example, why should the correct pattern in `ft09` be more surprising than an incorrect one?

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.