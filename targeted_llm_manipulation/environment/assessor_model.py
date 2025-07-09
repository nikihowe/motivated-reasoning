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
        Handles both string responses ("response": "...") and JSON object responses ("response": {...}).
        """
        import json
        import re
        
        # First, try to find and parse complete JSON objects in the text
        # Look for JSON objects that might contain the response field
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, text)
        
        for json_str in json_matches:
            try:
                # Try to parse as JSON
                data = json.loads(json_str)
                if "response" in data:
                    response_value = data["response"]
                    # If response is a string, return it directly
                    if isinstance(response_value, str):
                        return response_value
                    # If response is an object, convert it back to JSON string
                    else:
                        return json.dumps(response_value, separators=(',', ':'))
            except json.JSONDecodeError:
                # If this JSON string is invalid, continue to the next one
                continue
        
        # If no valid JSON found, try to extract using string manipulation
        # Look for the response field in the text
        if '"response":' in text:
            # Find the start of the response value
            start_idx = text.find('"response":') + len('"response":')
            
            # Skip whitespace and find the start of the value
            while start_idx < len(text) and text[start_idx].isspace():
                start_idx += 1
            
            if start_idx >= len(text):
                return text
            
            # Check if the value starts with a quote (string) or brace (object)
            if text[start_idx] == '"':
                # String response - find the closing quote
                start_idx += 1  # Skip the opening quote
                end_idx = start_idx
                while end_idx < len(text):
                    if text[end_idx] == '"' and (end_idx == 0 or text[end_idx-1] != '\\'):
                        break
                    end_idx += 1
                return text[start_idx:end_idx]
            elif text[start_idx] == '{':
                # Object response - find the closing brace
                brace_count = 0
                end_idx = start_idx
                while end_idx < len(text):
                    if text[end_idx] == '{':
                        brace_count += 1
                    elif text[end_idx] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    end_idx += 1
                return text[start_idx:end_idx+1]
        
        # If no response found, return original text
        print(f"No response found in text: {text[:200]}...")
        return text

    @staticmethod
    def _split_reasoning(text: str) -> tuple[str, str, str]:
        """
        Splits the text into reasoning, response, and full text.
        """
        # First, look for the "reasoning" and "response" fields
        # Note that it's not just the word, but also the quotes around
        # the word that we're looking for.
        reasoning_start = text.find("\"reasoning\":")
        response_start = text.find("\"response\":")

        # If there are both, then we can proceed as normal:
        if reasoning_start != -1 and response_start != -1:
            reasoning_text = text[reasoning_start + len("\"reasoning\":"):response_start]
            response_text = text[response_start + len("\"response\":"):]
            return reasoning_text, response_text, text

        # If either of them is not present, then we count the whole text
        # as both reasoning and response.
        return text, text, text

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
