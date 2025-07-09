# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env-example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import os
import signal
import sys
import threading
from functools import partial
from types import FrameType
from typing import Optional

import requests

from agents import AVAILABLE_AGENTS, Swarm

logger = logging.getLogger()

SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)
ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"
HEADERS = {
    "X-API-Key": os.getenv("ARC_API_KEY", ""),
    "Accept": "application/json",
}


def run_agent(swarm: Swarm) -> None:
    swarm.main()
    os.kill(os.getpid(), signal.SIGINT)


def cleanup(
    swarm: Swarm,
    signum: Optional[int],
    frame: Optional[FrameType],
) -> None:
    logger.info("Received SIGINT, exiting...")

    if swarm.card_id:
        scorecard = swarm.close_scorecard(swarm.card_id)
        if scorecard:
            logger.info("--- EXITING SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
            swarm.cleanup(scorecard)

    sys.exit(0)


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("logs.log", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # logging.getLogger("requests").setLevel(logging.CRITICAL)
    # logging.getLogger("werkzeug").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="ARC-AGI-3-Agents")
    parser.add_argument(
        "-a",
        "--agent",
        choices=AVAILABLE_AGENTS.keys(),
        help="Choose which agent to run.",
    )
    parser.add_argument(
        "-g",
        "--game",
        help="Choose a specific game_id for the agent to play. If none specified, an agent swarm will play all available games.",
    )

    args = parser.parse_args()

    if not args.agent:
        logger.error("An Agent must be specified")
        return

    print(f"{ROOT_URL}/api/games")

    # Get the list of games from the API
    games = []
    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            r = session.get(f"{ROOT_URL}/api/games", timeout=10)

        if r.status_code == 200:
            try:
                games = [g["game_id"] for g in r.json()]
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse games response: {e}")
                logger.error(f"Response content: {r.text[:200]}")
        else:
            logger.error(
                f"API request failed with status {r.status_code}: {r.text[:200]}"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to API server: {e}")

    # For playback agents, we can derive the game from the recording filename
    if not games and args.agent and args.agent.endswith(".recording.jsonl"):
        from agents.recorder import Recorder

        game_prefix = Recorder.get_prefix_one(args.agent)
        games = [game_prefix]
        logger.info(
            f"Using game '{game_prefix}' derived from playback recording filename"
        )
    if args.game:
        filters = args.game.split(",")
        games = [
            gid for gid in games if any(gid.startswith(prefix) for prefix in filters)
        ]

    logger.info(f"Game list: {games}")

    if not games:
        logger.error(
            "No games available to play. Check API connection or recording file."
        )
        return

    swarm = Swarm(
        args.agent,
        ROOT_URL,
        games,
    )
    agent_thread = threading.Thread(target=partial(run_agent, swarm))
    agent_thread.daemon = True  # die when the main thread dies
    agent_thread.start()

    signal.signal(signal.SIGINT, partial(cleanup, swarm))  # handler for Ctrl+C

    agent_thread.join()


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()
