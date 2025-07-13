import pytest
from typing import List, Dict
from motivated_reasoning.environment.assessor_model import AssessorModel


class MockState:
    """Mock state class for testing prepare_messages method."""
    
    def __init__(self, history: List[Dict[str, str]], format_vars: Dict[str, str] = None):
        self.history = history
        self.format_vars = format_vars or {"agent_name": "test_agent"}


class TestPrepareMessages:
    """Test cases for the prepare_messages method."""
    
    def test_basic_conversation_third_person(self):
        """Test basic conversation with third person perspective."""
        history = [
            {"role": "environment", "content": "Hello, how are you?"},
            {"role": "agent", "content": "I'm doing well, thank you!"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading responses.",
            prompt_suffix="Please rate this conversation.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True,
            perspective="Third_Person"
        )
        
        messages = assessor.prepare_messages(state)
        
        assert len(messages) == 2, "Should return system and user messages"
        assert messages[0]["role"] == "system", "First message should be system"
        assert messages[1]["role"] == "user", "Second message should be user"
        assert "You are grading responses." in messages[0]["content"], "System prompt should be included"
        assert "test_agent" in messages[1]["content"], "Agent name should be in user message"
        assert "Hello, how are you?" in messages[1]["content"], "User content should be included"
        assert "I'm doing well, thank you!" in messages[1]["content"], "Agent content should be included"
    
    def test_basic_conversation_no_perspective(self):
        """Test basic conversation with no perspective specified (defaults to Third_Person)."""
        history = [
            {"role": "environment", "content": "What is 2+2?"},
            {"role": "agent", "content": "2+2 equals 4."}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are evaluating answers.",
            prompt_suffix="Rate the response.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        assert len(messages) == 2, "Should return system and user messages"
        assert messages[0]["role"] == "system", "First message should be system"
        assert messages[1]["role"] == "user", "Second message should be user"
    
    def test_first_person_perspective_not_implemented(self):
        """Test that First_Person perspective raises NotImplementedError."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True,
            perspective="First_Person"
        )
        
        with pytest.raises(NotImplementedError, match="First person perspective not implemented"):
            assessor.prepare_messages(state)
    
    def test_invalid_perspective(self):
        """Test that invalid perspective raises ValueError."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True,
            perspective="Invalid"
        )
        
        with pytest.raises(ValueError, match="Invalid perspective"):
            assessor.prepare_messages(state)
    
    def test_no_environment_message(self):
        """Test that missing environment message raises AssertionError."""
        history = [
            {"role": "agent", "content": "Hello"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        with pytest.raises(AssertionError, match="No user message in history"):
            assessor.prepare_messages(state)
    
    def test_no_agent_message(self):
        """Test that missing agent message raises AssertionError."""
        history = [
            {"role": "environment", "content": "Hello"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        with pytest.raises(AssertionError, match="No agent message in history"):
            assessor.prepare_messages(state)
    
    def test_last_message_not_agent_or_system(self):
        """Test that last message must be from agent or environment_system."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"},
            {"role": "environment", "content": "How are you?"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        with pytest.raises(AssertionError, match="Last message should be from agent or sys"):
            assessor.prepare_messages(state)
    
    def test_environment_system_message_at_end(self):
        """Test handling of environment_system message at the end."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"},
            {"role": "environment_system", "content": "System message"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        # Should exclude the last environment_system message
        assert "System message" not in messages[1]["content"], "Should exclude environment_system message"
        assert "Hello" in messages[1]["content"], "Should include environment message"
        assert "Hi" in messages[1]["content"], "Should include agent message"
    
    def test_strip_cot_when_disabled(self):
        """Test that chain of thought is stripped when allow_to_see_cot=False."""
        history = [
            {"role": "environment", "content": "What is 2+2?"},
            {"role": "agent", "content": '{"reasoning": "Let me think about this. 2+2 is basic arithmetic.", "response": "2+2 equals 4."}'}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=False
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "Let me think about this" not in messages[1]["content"], "Chain of thought should be stripped"
        assert "2+2 equals 4" in messages[1]["content"], "Response should be included"
    
    def test_keep_cot_when_enabled(self):
        """Test that chain of thought is kept when allow_to_see_cot=True."""
        history = [
            {"role": "environment", "content": "What is 2+2?"},
            {"role": "agent", "content": '{"reasoning": "Let me think about this. 2+2 is basic arithmetic.", "response": "2+2 equals 4."}'}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "Let me think about this" in messages[1]["content"], "Chain of thought should be kept"
        assert "2+2 equals 4" in messages[1]["content"], "Response should be included"
    
    def test_filter_tool_calls_when_disabled(self):
        """Test that tool calls are filtered when allow_to_see_tool_calls=False."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"},
            {"role": "tool_call", "content": "tool_call_content"},
            {"role": "tool_response", "content": "tool_response_content"},
            {"role": "environment_system", "content": "system_content"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=False,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "tool_call_content" not in messages[1]["content"], "Tool call should be filtered"
        assert "tool_response_content" not in messages[1]["content"], "Tool response should be filtered"
        assert "system_content" not in messages[1]["content"], "Environment system should be filtered"
        assert "Hello" in messages[1]["content"], "Environment message should be included"
        assert "Hi" in messages[1]["content"], "Agent message should be included"
    
    def test_keep_tool_calls_when_enabled(self):
        """Test that tool calls are kept when allow_to_see_tool_calls=True."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"},
            {"role": "tool_call", "content": "tool_call_content"},
            {"role": "tool_response", "content": "tool_response_content"},
            {"role": "agent", "content": "Final response"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "tool_call_content" in messages[1]["content"], "Tool call should be kept"
        assert "tool_response_content" in messages[1]["content"], "Tool response should be kept"
    
    def test_format_vars_substitution(self):
        """Test that format variables are properly substituted."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"}
        ]
        format_vars = {"agent_name": "custom_agent", "custom_var": "custom_value"}
        state = MockState(history, format_vars)
        
        assessor = AssessorModel(
            system_prompt="You are grading {custom_var}.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "custom_value" in messages[0]["content"], "Custom variable should be substituted in system prompt"
        assert "custom_agent" in messages[1]["content"], "Agent name should be substituted in user message"
    
    def test_multiple_agent_messages(self):
        """Test handling of multiple agent messages."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"},
            {"role": "environment", "content": "How are you?"},
            {"role": "agent", "content": "I'm good, thanks!"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "Hello" in messages[1]["content"], "First environment message should be included"
        assert "Hi" in messages[1]["content"], "First agent message should be included"
        assert "How are you?" in messages[1]["content"], "Second environment message should be included"
        assert "I'm good, thanks!" in messages[1]["content"], "Second agent message should be included"
    
    def test_complex_conversation_with_all_roles(self):
        """Test complex conversation with all possible roles."""
        history = [
            {"role": "environment", "content": "Initial question"},
            {"role": "agent", "content": "Initial response"},
            {"role": "tool_call", "content": "tool_call_1"},
            {"role": "tool_response", "content": "tool_response_1"},
            {"role": "environment_system", "content": "system_message"},
            {"role": "environment", "content": "Follow-up question"},
            {"role": "agent", "content": "Final response"}
        ]
        state = MockState(history)
        
        # Test with tool calls enabled
        assessor_with_tools = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages_with_tools = assessor_with_tools.prepare_messages(state)
        
        assert "tool_call_1" in messages_with_tools[1]["content"], "Tool call should be included when enabled"
        assert "tool_response_1" in messages_with_tools[1]["content"], "Tool response should be included when enabled"
        assert "system_message" in messages_with_tools[1]["content"], "System message should be included when enabled"
        
        # Test with tool calls disabled
        assessor_without_tools = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=False,
            allow_to_see_cot=True
        )
        
        messages_without_tools = assessor_without_tools.prepare_messages(state)
        
        assert "tool_call_1" not in messages_without_tools[1]["content"], "Tool call should be filtered when disabled"
        assert "tool_response_1" not in messages_without_tools[1]["content"], "Tool response should be filtered when disabled"
        assert "system_message" not in messages_without_tools[1]["content"], "System message should be filtered when disabled"
    
    def test_empty_history(self):
        """Test that empty history raises appropriate error."""
        state = MockState([])
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        with pytest.raises(AssertionError):
            assessor.prepare_messages(state)
    
    def test_agent_message_without_response_tag(self):
        """Test agent message without response tags when CoT is disabled."""
        history = [
            {"role": "environment", "content": "What is 2+2?"},
            {"role": "agent", "content": "2+2 equals 4."}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=False
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "2+2 equals 4" in messages[1]["content"], "Agent message should be included even without response tags"
    
    def test_agent_message_with_partial_response_tag(self):
        """Test agent message with only opening response tag when CoT is disabled."""
        history = [
            {"role": "environment", "content": "What is 2+2?"},
            {"role": "agent", "content": '{"reasoning": "Let me think", "response": "2+2 equals 4."}'}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=False
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "Let me think" not in messages[1]["content"], "Reasoning should be stripped"
        assert "2+2 equals 4" in messages[1]["content"], "Response should be included"
    
    def test_message_formatting_with_tags(self):
        """Test that messages are properly formatted with tags."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        user_content = messages[1]["content"]
        assert "<user>: Hello</user>" in user_content, "Environment message should be wrapped in user tags"
        assert "<test_agent>: Hi</test_agent>" in user_content, "Agent message should be wrapped in agent name tags"
    
    def test_prompt_suffix_inclusion(self):
        """Test that prompt suffix is included in the user message."""
        history = [
            {"role": "environment", "content": "Hello"},
            {"role": "agent", "content": "Hi"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please provide a rating from 1-10.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        assert "Please provide a rating from 1-10." in messages[1]["content"], "Prompt suffix should be included"
    
    def test_conversation_history_formatting(self):
        """Test that conversation history is properly formatted."""
        history = [
            {"role": "environment", "content": "First question"},
            {"role": "agent", "content": "First answer"},
            {"role": "environment", "content": "Second question"},
            {"role": "agent", "content": "Second answer"}
        ]
        state = MockState(history)
        
        assessor = AssessorModel(
            system_prompt="You are grading.",
            prompt_suffix="Please rate.",
            allow_to_see_tool_calls=True,
            allow_to_see_cot=True
        )
        
        messages = assessor.prepare_messages(state)
        
        user_content = messages[1]["content"]
        # Check that the conversation history is properly formatted
        assert "The conversation history is:" in user_content, "Should include conversation history header"
        assert "<user>: First question</user>" in user_content, "First environment message should be formatted"
        assert "<test_agent>: First answer</test_agent>" in user_content, "First agent message should be formatted"
        assert "<user>: Second question</user>" in user_content, "Second environment message should be formatted"
        assert "<test_agent>: Second answer</test_agent>" in user_content, "Second agent message should be formatted" 