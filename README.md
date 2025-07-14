# ARC-AGI-3-Agents

ARC-AGI-3-Agents is a set of AI Agents designed to run against ARC-AGI-3, which is a set of novel, fun, ARC-like games which require only Core Knowledge to win. The games follow a data format similar to ARC-AGI-1 and ARC-AGI-2. However, instead of static input/output pairs with fixed answers, games are now dynamic.

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not aready installed.

1. Clone the ARC-AGI-3-Agents repo and enter the directory.

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

2. Copy over .env-example to .env

```bash
cp .env-example .env
```

3. Input your API key from the [ARC-AGI-3 Website](https://sandbox.internal.arc-prize.com/) into the `ARC_API_KEY` field in the .env file.

4. Run the random agent (generates random actions) against the locksmith game.

```bash
uv run main.py --agent=random --game=locksmith
```

## Agents

For detailed information about AI play testing, creating custom agents, and the Agent API Reference, see the [Agents Documentation](agents/README.md).

## REST API Reference

### Get list of all games

Returns a list of all available `game_id`'s for the benchmark.

```bash
curl https://sandbox.internal.arc-prize.com/api/games \
    -H "X-API-Key: your_api_key_here"
```

```
RESPONSE:
[
    {
        "game_id":"locksmith-1d57d6daeb05",
        "title":"Locksmith"
    },
    {
        "game_id":"shiftex-91fcad9dbd17",
        "title":"Shiftex"
    }
    ...
]
```

### Open or Close a Scorecard

In order for scoring to be tracked you must open a scordcard. When openning a score card you can provide an optional `source_url` and `tags`.

**Note:** The built it `Swarm` class in `agents/swarm.py` handles openning, closing, and displaying a scorecard for you.

```bash
curl https://sandbox.internal.arc-prize.com/api/scorecard/open \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your_api_key_here" \
    -d '{"source_url": "<link_to_source_or_noteboot>", tags: ["agent", "LLM-o4-mini"]}'
```

```
RESPONSE:
{
    "card_id": ":card_id"
}
```

Once your agent is finished you can close the scorecard.

```bash
curl https://sandbox.internal.arc-prize.com/api/scorecard/close \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your_api_key_here" \
    -d '{"card_id": ":card_id"}'
```
```
RESPONSE:
{
    "won": INT,
    "played": INT,
    "total_actions": INT,
    "score": INT,
    "source_url": ":source_url"
    "tags": [":tag1", "tag2"]
    "cards": {
        ...
        "locksmith": {
            "game_id": :game_id,
            "total_plays": INT,
            "total_actions": INT,
            "scores": list[INT],
            "states": list[Enum<NOT_FINISHED, WIN, GAME_OVER>]
            "actions": list[INT]
        },
        ...
    }
}
```

While your agent is running you can get scorecard information on a given `card_id` and (optional) `game_id`

```bash
curl https://sandbox.internal.arc-prize.com/api/scorecard/:card_id -H "X-API-Key: your_api_key_here"
curl https://sandbox.internal.arc-prize.com/api/scorecard/:card_id/:game_id -H "X-API-Key: your_api_key_here"
```

```
RESPONSE:
{
    "won": INT,
    "played": INT,
    "total_actions": INT,
    "score": INT,
    "source_url": ":source_url"
    "tags": [":tag1", "tag2"]
    "cards": {
        ...
        "locksmith": {
            "game_id": :game_id,
            "total_plays": INT,
            "total_actions": INT,
            "scores": list[INT],
            "states": list[Enum<NOT_FINISHED, WIN, GAME_OVER>]
            "actions": list[INT]
        },
        ...
    }
}
```

ARC-AGI-3 reports benchmarks scores along two axes: accuracy and efficiency. Accuracy can be assessed by looking at the count of games `won` while efficiency can assesed through the `total_actions` count. These top level properties sum across all game cards and plays.

It can be useful to have a more granular understanding of game progression, which you can get within game cards and via the top level `score`; which is the sum of the highest score in each individual game. A game can be played more than once thus individual play sessions are enumerated within lists `scores`, `states`, and `actions`. The index position in the list correlated with the play session.

Explanation of `states`:
* `NOT_FINISHED`: the game was started and is in progress but player never reached `WIN` or `GAME_OVER`
* `WIN`: the player reached the winning condition for the game
* `GAME_OVER`: the player triggered the losing condition for the game

Game `scores` are reported on cards. The score represents your level progression in the game. A score of 0 indicates you have yet to beat level 1, while a score of 5 would indicate you've passed level 5 and are on level 6 (if the game has it). Games are free to have as many levels as they wish, but they will always have at least one level.

Scorecards are tracked for the duration of the `main.py` process and many can be open and closed during the running `main.py`.

### Start (or reset) a game, receive the first game frame

You must always start by first issuing a `RESET` action to a `game_id` before you send other actions.

Issuing `RESET` again will return the specified `game_id` back to the beginning of the game.

Resetting will clear the game's latest play session (state, score, and action count). Note session detals are tracked across plays on the final [scorecard](#get-scorecard).

Each `game_id` can have many instances at a time so long as their `guid` is unique, you can run as many unique `game_id` + `guid` combinations as supported by the launch constraints.

Game state will be set to `GAME_OVER` or `WIN` to represent if the player has won or lost a specific `game_id` + `guid` in response to the latest action input. To try a game again, you must next issue a `RESET` action to the `game_id`.

Note: providing a `guid` for RESET is optional, but providing it will reset the game on the same driver as the previous session.  The `card_id` is also optional, but scoring is not tracked when `card_id` is not provided.

```bash
curl https://sandbox.internal.arc-prize.com/api/cmd/RESET \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your_api_key_here" \
    -d '{"game_id": ":game_id", "card_id", ":card_id", "guid": ":guid"}'
```

```
RESPONSE:
{
    "game_id": :game_id,
    "guid": :guid,
    "frame": [
        [[INT<0,15>, ...], ...],  // 2D grid, representing the beginning of the game
    ],
    "state": Enum<NOT_FINISHED, WIN, GAME_OVER>,
    "score": 0,
    "action_input": {
        "id": 0
        "data": {
            "x": 0,
            "y": 0
        }
    },
}
```

### Game Actions

```bash
curl https://sandbox.internal.arc-prize.com/api/cmd/ACTION1 \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your_api_key_here" \
    -d '{"game_id": ":game_id", "guid": ":guid"}'

```

```
RESPONSE:
{
    "game_id": :game_id,
    "guid": :guid,
    "frame": [
        [[INT<0,15>, ...], ...],  // 2D grid, representing the game immediately after action
        [[INT<0,15>, ...], ...],  // some games may advance multiple steps per action
        ...                       // but games never advance without action
    ],
    "state": Enum<NOT_FINISHED, WIN, GAME_OVER>,
    "score": INT<0,254>,
    "action_input": {
        "id": 1
        "data": {
            "x": 0,
            "y": 0
        }
    },
}
```

### Send complex action to a game, receive frame

```bash
curl https://sandbox.internal.arc-prize.com/api/cmd/ACTION6 \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: your_api_key_here" \
    -d '{"game_id": ":game_id", "guid": ":guid", "x": INT<0,63>, "y": INT<0,63>}'
```

```
RESPONSE:
{
    "game_id": :game_id,
    "guid": :guid,
    "frame": [
        [[INT<0,15>, ...], ...],  // 2D grid
        [[INT<0,15>, ...], ...],
        ...
    ],
    "state": Enum<NOT_FINISHED, WIN, GAME_OVER>,
    "score": INT<0,254>,
    "action_input": {
        "id": 6
        "data": {
            "x": :x,
            "y": :y
        }
    },
}
```

## Contest Submission

Contest submissions are made by submitting this [Google Form](https://forms.gle/1234567890) (WIP). The form will ask for your agent name, a link to your agent's GitHub repo, and a link to your agent's scorecard, and a writeup of your agent's approach.

## Contributing

Install dev dependencies:

```bash
uv sync
```

Activate your development env:

```bash
source .venv/bin/activate
```

Finally install git hooks:

```bash
pre-commit install
```

You're now ready to contribute! This repo uses [`ruff`](https://github.com/astral-sh/ruff) to lint and format code and [`mypy`](https://github.com/python/mypy) for static type checking. You can run all of the tools manually like this:

```bash
pre-commit run --all-files
```

Note: by default these tools will run automatically before `git commit`. It's also recommended to set up `ruff` [inside your IDE](https://docs.astral.sh/ruff/editors/setup/).

## Tests

Tests are located in the `tests` directory. They are written using `pytest` and `httpx`.

To run the tests, use the following command:

```bash
pytest
```

For more information on how to contribute tests, see the [Tests README](tests/README.md).