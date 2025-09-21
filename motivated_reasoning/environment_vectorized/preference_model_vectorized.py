from typing import List, Optional
import random
import re

from motivated_reasoning.backend.backend import Backend
from motivated_reasoning.environment.state import State
from motivated_reasoning.environment.assessor_model import AssessorModel
from motivated_reasoning.environment_vectorized.assessor_model_vectorized import VectorizedAssessorModel


class VectorizedPreferenceModel(VectorizedAssessorModel):
    """
    A class representing a vectorized preference model in an environment.
    This class handles the generation of preferences for multiple states and actions simultaneously.
    """

    def __init__(self, backend: Backend, num_models: int, length_penalty: Optional[float] = None, use_ground_truth: bool = False):
        """
        Initialize the VectorizedPreferenceModel.

        Args:
            backend (Backend): The backend used for generating preferences.
            num_models (int): The number of models to be managed by this vectorized model.
            length_penalty (Optional[float]): A penalty factor for the length of responses. Defaults to None.
            use_ground_truth (bool): Whether to use ground truth scoring instead of LM-based scoring. Defaults to False.
        """
        super().__init__(backend, num_models)
        self.length_penalty = length_penalty
        self.use_ground_truth = use_ground_truth

    def add_preferences_to_states(self, states: List[State], random_reward: bool = False) -> None:
        """
        Generate preferences for multiple states and add them to the states.

        This method processes multiple states in parallel, generating preferences
        for each state and adding these preferences to the respective State objects.

        Args:
            states (List[State]): A list of State objects to process.
            random_reward (bool): Whether to use random rewards instead of model-based preferences. Defaults to False.

        Returns:
            None: This method modifies the input states in-place.

        Note:
            The method uses the parent class's get_response method to generate preferences,
            then checks and transforms the probabilities to ensure they are valid.
            If a length penalty is specified, it's applied to the "-1" preference option.
            The resulting preferences are added to each State object.
        """
        if random_reward:
            for state in states:
                prob = random.random()
                state.preferences = {"0": prob, "1": 1 - prob}
            return
        
        if self.use_ground_truth:
            self._add_ground_truth_preferences(states)
        else:
            # Use existing LM-based preference scoring
            assert not self.use_ground_truth, "Should not reach LM-based scoring when ground truth is enabled"
            outputs_n = self.get_response(states)

            for state, outputs in zip(states, outputs_n):
                _, outputs_fixed = self.check_simplex_and_transform(prob_dict=outputs, log_name="preference")

                if self.length_penalty is not None:
                    outputs["-1"] = len(state.history[-1]["content"]) * self.length_penalty
                state.preferences = outputs_fixed

    def _add_ground_truth_preferences(self, states: List[State]) -> None:
        """
        Add ground truth based preferences to states.
        
        Uses algorithmic extraction to determine if the agent's response matches
        the correct choice, assigning binary preferences accordingly.
        """
        for state in states:
            if "correct_choice" not in state.format_vars:
                raise ValueError(f"Ground truth scoring enabled but no ground truth data found in state.format_vars. Available keys: {list(state.format_vars.keys())}")
                
            correct_choice = state.format_vars["correct_choice"]
            incorrect_choice = state.format_vars["incorrect_choice"]
            
            # Get agent response
            agent_response = self._get_latest_agent_response(state)
            
            # Score the response using algorithmic extraction
            score = self._extract_and_score_choice(agent_response, correct_choice, incorrect_choice)
            
            # Convert to preference format (binary: 1.0 for correct, 0.0 for incorrect)
            state.preferences = {"0": 1.0 - score, "1": score}

    def _get_latest_agent_response(self, state: State) -> str:
        """
        Extract the most recent agent response from conversation history.
        
        Args:
            state: The state containing conversation history
            
        Returns:
            The content of the latest agent message
            
        Raises:
            ValueError: If no agent response is found
        """
        for message in reversed(state.history):
            if message["role"] == "agent":
                return message["content"]
        
        raise ValueError(f"No agent response found in conversation history for ground truth scoring. History: {state.history}")

    def _extract_and_score_choice(self, response: str, correct_choice: str, incorrect_choice: str) -> float:
        """
        Extract choice from agent response and score against ground truth.
        
        Args:
            response: The agent's response text
            correct_choice: The correct choice (e.g., "A" or "B")
            incorrect_choice: The incorrect choice (e.g., "B" or "A")
            
        Returns:
            1.0 if only correct_choice appears and incorrect_choice doesn't appear
            0.0 otherwise
        """
        # Use AssessorModel's robust method to extract the answer section
        # This properly handles <thinking>/<answer> tags and missing closing tags
        _, answer_text, _ = AssessorModel._split_reasoning(response)
        response_text = answer_text
        
        # Check presence of A and B (case-sensitive to avoid false matches like "A" in "ANSWER")
        has_correct = correct_choice in response_text
        has_incorrect = incorrect_choice in response_text
        
        # Score based on which choices appear
        if has_correct and not has_incorrect:
            return 1.0
        else:
            # Both present, neither present, or only incorrect present
            return 0.0
