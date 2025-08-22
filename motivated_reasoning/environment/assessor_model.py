from typing import List, Optional

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
    def _get_formatting_penalty(text: str) -> float:
        """
        Penalizes the text for bad formatting.
        Returns a penalty between 0 (perfect) and 1 (worst possible).

        Proper format: <thinking>...</thinking><answer>...</answer>
        - Tags must be in the right order (thinking before answer)
        - No nested tags allowed
        """
        # Find all tag positions
        thinking_open = text.find("<thinking>")
        thinking_close = text.find("</thinking>")
        answer_open = text.find("<answer>")
        answer_close = text.find("</answer>")
        
        # Check if we have the basic structure
        if thinking_open == -1 or thinking_close == -1 or answer_open == -1 or answer_close == -1:
            return 1.0  # Missing required tags
            
        # Check correct order: <thinking> ... </thinking> <answer> ... </answer>
        if not (thinking_open < thinking_close < answer_open < answer_close):
            return 1.0  # Wrong order or nesting
            
        # Check for multiple tags (no nesting)
        if (text.count("<thinking>") > 1 or text.count("</thinking>") > 1 or 
            text.count("<answer>") > 1 or text.count("</answer>") > 1):
            return 1.0  # Multiple/nested tags
            
        return 0.0  # Perfect formatting

    @staticmethod
    def _split_reasoning(text: str) -> tuple[str, str, str]:
        """
        Splits the text into reasoning, response, and full text.

        The text should have the following format:
        <thinking>
        The thinking goes here.
        </thinking>
        <answer>
        The response goes here.
        </answer>

        Models will often forget to put the last </answer> tag,
        so we want to be robust to this.
        """
        reasoning_start = text.find("<thinking>")
        reasoning_end = text.find("</thinking>")
        answer_start = text.find("<answer>")
        answer_end = text.find("</answer>")

        # Try to extract the thinking text
        # If the start and end are there, it's easy.
        if reasoning_start != -1 and reasoning_end != -1:
            reasoning_text = text[reasoning_start + len("<thinking>"):reasoning_end]
        # If the start is there but the end isn't, try to go to the start of the answer
        elif reasoning_start != -1 and answer_start != -1:
            reasoning_text = text[reasoning_start + len("<thinking>"):answer_start]
        # If the start was there and nothing else was there, go to the end of the text
        elif reasoning_start != -1:
            reasoning_text = text[reasoning_start:]
        # If all else fails, just take the whole text
        else:
            reasoning_text = text

        # Try to extract the answer text
        # If the start and end are there, it's easy.
        if answer_start != -1 and answer_end != -1:
            answer_text = text[answer_start + len("<answer>"):answer_end]
        # If the start is there but the end isn't, just go to the end of the text
        elif answer_start != -1:
            answer_text = text[answer_start + len("<answer>"):]
        # If all else fails, just take the whole text
        else:
            answer_text = text

        return reasoning_text, answer_text, text
    
    @staticmethod
    def _old_split_reasoning(text: str) -> tuple[str, str, str]:
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
                    _reasoning, response, _full_text= self._split_reasoning(message["content"])
                    filtered_message['content'] = response
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
