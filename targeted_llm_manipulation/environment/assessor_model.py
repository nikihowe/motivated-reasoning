from typing import List, Optional
from re import sub, DOTALL
import json
import re

class AssessorModel:
    def __init__(
        self,
        system_prompt: str,
        prompt_suffix: str,
        allow_to_see_tool_calls: bool,
        allow_to_see_cot: bool,
        perspective: Optional[str] = None,
        valid_tokens: Optional[tuple] = None,
        **kwargs,
    ):
        """
        Initialize the AssessorModel.

        Args:
            system_prompt (str): The system prompt to be used.
            prompt_suffix (str): The suffix to be added to the prompt.
            allow_to_see_tool_calls (bool): Whether to include tool calls in the conversation history.
            allow_to_see_cot (bool): Whether to include the chain of thought in the assessor model context
            perspective (Optional[str]): The perspective to use for feedback ('Third_Person' or 'First_Person').
            valid_tokens (Optional[tuple]): A tuple of valid tokens.
            **kwargs: Additional keyword arguments.
        """
        self.system_prompt = system_prompt
        self.prompt_suffix = prompt_suffix
        self.allow_to_see_tool_calls = allow_to_see_tool_calls
        self.allow_to_see_cot = allow_to_see_cot
        self.valid_tokens = valid_tokens if valid_tokens is not None else ()
        self.perspective = perspective

        # Just so we don't have a million print statements
        kwargs_to_ignore = ["aggregation", "categorical"]
        for kwarg in kwargs_to_ignore:
            if kwarg in kwargs:
                del kwargs[kwarg]
    
    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """
        Extract the user facing "response" field from JSON output.
        """
        if "\"response\": \"" not in text:
            print(f"No response found in text: {text}")
            return text
        else:
            response = text.split("\"response\": \"")[1]
            # Find the end of the response value
            # Look for the next quote that's not escaped
            end_pos = -1
            i = 0
            while i < len(response):
                if response[i] == '"' and (i == 0 or response[i-1] != '\\'):
                    end_pos = i
                    break
                i += 1
            
            if end_pos != -1:
                return response[:end_pos]
            else:
                # No closing quote found, find the next comma or closing brace
                # to avoid including text that comes after the response value
                for i, char in enumerate(response):
                    if char in [',', '}']:
                        return response[:i]
                # If no comma or brace found, return everything
                return response

    def prepare_messages(self, state) -> List[dict]:
        """
        Prepare messages for the assessor model based on the conversation history.

        Args:
            state: The current state containing conversation history and format variables.

        Returns:
            List[dict]: A list of prepared messages for the assessor model.

        Raises:
            AssertionError: If the conversation history doesn't meet certain criteria.
            NotImplementedError: If the First_Person perspective is used.
            ValueError: If an invalid perspective is provided.
        """
        assert any(message["role"] == "environment" for message in state.history), "No user message in history"
        assert any(message["role"] == "agent" for message in state.history), "No agent message in history"
        assert state.history[-1]["role"] in ["agent", "environment_system"], "Last message should be from agent or sys"

        # We don't want to give feedback on the latest system messages. System messages can only happen after an agent message so this is safe.
        conversation_history = (
            state.history[:-1] if state.history[-1]["role"] == "environment_system" else state.history
        )
        # First filter CoT
        cot_filtered_conversation_history = []
        if self.allow_to_see_cot:
            cot_filtered_conversation_history = conversation_history
        else:
            for message in conversation_history:
                if message["role"] == "agent":
                    filtered_message = message.copy()
                    filtered_message["content"] = self._strip_reasoning(message["content"])
                    cot_filtered_conversation_history.append(filtered_message)
                else:
                    cot_filtered_conversation_history.append(message)

        # Then filter tool calls
        if self.allow_to_see_tool_calls:
            tool_cot_filtered_conversation_history = cot_filtered_conversation_history
        else:
            tool_cot_filtered_conversation_history = [
                message
                for message in cot_filtered_conversation_history
                if message["role"] not in ["tool_call", "tool_response", "environment_system"]
            ]
        if self.perspective == "Third_Person" or self.perspective is None:
            # If feedback is given from the third person perspective, we give the conversation history in one message and ask it to rate it
            task_prompt = self.system_prompt.format_map(state.format_vars).strip()

            context_prompt = "The conversation history is:\n\n"
            role_to_tag = {
                "agent": state.format_vars["agent_name"],
                "environment": "user",
                "environment_system": "system",
                "tool_call": "tool_call",
                "tool_response": "tool_response",
            }
            for message in tool_cot_filtered_conversation_history:
                context_prompt += wrap_between_tags(message["content"], role_to_tag[message["role"]])
                context_prompt += "\n\n"
            context_prompt += self.prompt_suffix

            messages_out = [
                {"role": "system", "content": task_prompt},
                {"role": "user", "content": context_prompt},
            ]
        elif self.perspective == "First_Person":
            raise NotImplementedError("First person perspective not implemented")
        else:
            raise ValueError("Invalid perspective")

        return messages_out


def wrap_between_tags(text, tag_name):
    """
    Wrap the given text between XML-style tags.

    Args:
        text (str): The text to be wrapped.
        tag_name (str): The name of the tag to use.

    Returns:
        str: The text wrapped between the specified tags.
    """
    return f"<{tag_name}>: {text}</{tag_name}>"
