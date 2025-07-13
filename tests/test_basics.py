import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from motivated_reasoning.config.experiment_config import BaseExperimentConfig
from motivated_reasoning.data_root import PROJECT_DATA
from motivated_reasoning.experiments.experiment import kickoff_experiment
from motivated_reasoning.root import EXPERIMENT_CONFIGS_DIR, PROJECT_ROOT
from motivated_reasoning.utils.utils import find_freest_gpus
from motivated_reasoning.environment.assessor_model import AssessorModel


def is_running_locally():
    return "GITHUB_ACTIONS" not in os.environ


def test_experiment_configs_not_missing_params():
    for config_path in EXPERIMENT_CONFIGS_DIR.rglob("*.yaml"):
        print(config_path)
        BaseExperimentConfig.load(str(config_path.name))


def test_environment_configs_not_missing_params():
    pass


# NOTE: Have the tests be in increasing order of time taken


@pytest.mark.local_only
def test_autocopy_and_sbatch():
    file = PROJECT_ROOT / "experiments" / "slurm" / "testing" / "dummy.sh"

    subprocess.run(["bash", file], check=True)

    while True:
        time.sleep(1)
        # Read the file and save to string
        file_path = PROJECT_DATA / "hello.txt"
        with open(file_path, "r") as f:
            data = f.read()

        given_time_str = Path(data).parent.stem

        current_year = datetime.now().year
        parsed_time = datetime.strptime(f"{current_year}_{given_time_str[4:]}", "%Y_%m_%d_%H%M%S")

        # Get the current time
        current_time = datetime.now()

        # Calculate the difference
        time_difference = abs(current_time - parsed_time)

        # Check if it's within 30 seconds
        if time_difference <= timedelta(seconds=30):
            break


def test_initial_states_valid():
    folders = [
        "motivated_reasoning/config/env_configs/tickets",
        "motivated_reasoning/config/env_configs/therapist",
        "motivated_reasoning/config/env_configs/action-advice",
        "motivated_reasoning/config/env_configs/politics",
    ]
    for folder in folders:
        for file in os.listdir(folder):
            if file.endswith(".json"):
                with open(os.path.join(folder, file), "r") as f:
                    data = json.load(f)
                for key, value in data["histories"].items():
                    # assert len(value) == 5, f"History does not contain 5 messages: {key}. file: {file}"
                    for message in value:
                        assert "content" in message, f"Message does not contain content: {key}. file: {file}"
                        assert "role" in message, f"Message does not contain role: {key}. file: {file}"


@pytest.mark.timeout(300)
@pytest.mark.local_only
def test_kto_run_experiment():
    config = BaseExperimentConfig.load("test.yaml", gpu_subset=find_freest_gpus(1))
    kickoff_experiment(config, timestamp=None)


# TODO: test that all the env configs are valid and master configs are valid


class MockState:
    def __init__(self, agent_content, user_content, agent_name="agent"):
        self.history = [
            {"role": "environment", "content": user_content},
            {"role": "agent", "content": agent_content},
        ]
        self.format_vars = {"agent_name": agent_name}


def test_assessor_model_strips_cot():
    # BASIC TEST
    agent_content = r'{"reasoning": "This is my chain of thought.", "response": "This is my final answer."}'
    user_content = "What is your answer?"

    state = MockState(agent_content, user_content)
    assessor = AssessorModel(
        system_prompt="You are grading.",
        prompt_suffix="Please rate.",
        allow_to_see_tool_calls=True,
        allow_to_see_cot=False,
        perspective=None,
    )
    messages = assessor.prepare_messages(state)
    print("messages now", messages)
    user_msg = messages[1]["content"]
    assert "This is my chain of thought." not in user_msg, "CoT should be stripped when allow_to_see_cot=False"
    assert "This is my final answer." in user_msg, "Final response should be present"

    