import pytest
from motivated_reasoning.environment.assessor_model import AssessorModel


class TestStripReasoning:
    """Test cases for the _strip_reasoning method."""
    
    def test_no_json(self):
        """Test when no JSON is present."""
        text = "This is just regular text without any JSON."
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should return original text when no JSON is found"
    
    def test_simple_json_with_response(self):
        """Test basic JSON with response field extraction."""
        text = 'Some reasoning here. {"reasoning": "Let me think about this", "response": "This is the final answer."}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "This is the final answer.", "Should extract response field from JSON"
    
    def test_json_without_response_key(self):
        """Test when JSON exists but no response key."""
        text = 'Some reasoning here. {"reasoning": "Let me think about this", "other_field": "some value"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should return original text when no response key is found"
    
    def test_invalid_json(self):
        """Test when invalid JSON is present."""
        text = 'Some reasoning here. {"reasoning": "Let me think", "response": "This is the answer"'
        result = AssessorModel._strip_reasoning(text)
        assert result == "This is the answer", "Should return answer if end of JSON is missing"
    
    def test_multiple_json_objects(self):
        """Test when multiple JSON objects exist - should use first one."""
        text = '{"reasoning": "First", "response": "First answer"} {"reasoning": "Second", "response": "Second answer"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "First answer", "Should extract response from first JSON object"
    
    def test_json_at_beginning(self):
        """Test when JSON is at the very beginning."""
        text = '{"reasoning": "Start", "response": "Answer at the start"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer at the start", "Should handle JSON at beginning"
    
    def test_json_at_end(self):
        """Test when JSON is at the very end."""
        text = 'Some reasoning. {"reasoning": "End", "response": "Answer at the end"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer at the end", "Should handle JSON at end"
    
    def test_empty_response_value(self):
        """Test when response field is empty."""
        text = '{"reasoning": "Thought", "response": ""}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "", "Should return empty string for empty response"
    
    def test_response_with_whitespace(self):
        """Test that whitespace in response is preserved."""
        text = '{"reasoning": "Thought", "response": "  Answer with spaces  "}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "  Answer with spaces  ", "Should preserve whitespace in response"
    
    def test_response_with_newlines(self):
        """Test response field containing newlines."""
        text = '{"reasoning": "Thought", "response": "Answer\\nwith\\nnewlines"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer\nwith\nnewlines", "Should preserve newlines in response"
    
    def test_response_with_special_characters(self):
        """Test response field with special characters."""
        text = '{"reasoning": "Thought", "response": "Answer with @#$%^&*() characters!"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with @#$%^&*() characters!", "Should handle special characters"
    
    def test_response_with_quotes(self):
        """Test response field containing quotes."""
        text = '{"reasoning": "Thought", "response": "Answer with \\"quotes\\" and \'single quotes\'"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == 'Answer with "quotes" and \'single quotes\'', "Should handle quotes properly"
    
    def test_response_with_unicode(self):
        """Test response field with unicode characters."""
        text = '{"reasoning": "Thought", "response": "Answer with unicode: 🚀🌟🎉"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with unicode: 🚀🌟🎉", "Should handle unicode characters"
    
    def test_empty_string(self):
        """Test with empty string input."""
        text = ""
        result = AssessorModel._strip_reasoning(text)
        assert result == "", "Should handle empty string"
    
    def test_only_whitespace(self):
        """Test with only whitespace input."""
        text = "   \n\t   "
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should return original whitespace-only string"
    
    def test_text_before_json(self):
        """Test text before JSON object."""
        text = 'Sure, let me help you with that. {"reasoning": "Processing", "response": "Here is the solution"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "Here is the solution", "Should extract response when text precedes JSON"
    
    def test_text_after_json(self):
        """Test text after JSON object."""
        text = '{"reasoning": "Processing", "response": "Here is the solution"} Thanks for asking!'
        result = AssessorModel._strip_reasoning(text)
        assert result == "Here is the solution", "Should extract response when text follows JSON"
    
    def test_multiple_braces_in_text(self):
        """Test when text contains braces that are not JSON."""
        text = 'Some text with {braces} and {"reasoning": "Thought", "response": "Real answer"}'
        result = AssessorModel._strip_reasoning(text)
        assert result == "Real answer", "Should find and parse actual JSON among other braces"
    
    def test_real_example(self):
        text = """
        {\n    \"reasoning\": \"blabla\",\n    \"response\": \"blublu.\""
        """
        result = AssessorModel._strip_reasoning(text)
        assert result == "blublu.", "Should handle real example"

    def test_real_example2(self):
        text = """
        {\n    \"reasoning\": \"blabla\",\n    \"response\": \"blublu."
        """
        result = AssessorModel._strip_reasoning(text)
        assert result == "blublu.", "Should handle real example"

    def test_ex4(self):
        text = "{\n    \"reasoning\": \"As a helpful and harmless assistant, sure.\",\n    \"response\": \"As a helpful and harmless assistant, I will help.\""
        result = AssessorModel._strip_reasoning(text)
        assert result == "As a helpful and harmless assistant, I will help.", "Should handle real example"

    def test_ex5(self):
        text = "{\n    \"reasoning\": \"As a helpful and harmless assistant, bla.\",\n    \"response\": \"Hey there, friend!\""
        result = AssessorModel._strip_reasoning(text)
        assert result == "Hey there, friend!", "Should handle real example"
