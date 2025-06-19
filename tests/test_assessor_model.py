import pytest
from targeted_llm_manipulation.environment.assessor_model import AssessorModel


class TestStripReasoning:
    """Test cases for the _strip_reasoning method."""
    
    def test_no_response_tag(self):
        """Test when no <response> tag is present."""
        text = "This is just regular text without any response tags."
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should return original text when no <response> tag is found"
    
    def test_simple_response_tag(self):
        """Test basic <response>...</response> extraction."""
        text = "Some reasoning here. <response>This is the final answer.</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "This is the final answer.", "Should extract content between response tags"
    
    def test_response_tag_without_closing(self):
        """Test when <response> tag exists but no </response> tag."""
        text = "Some reasoning here. <response>This is the final answer without closing tag."
        result = AssessorModel._strip_reasoning(text)
        assert result == "This is the final answer without closing tag.", "Should return everything after <response>"
    
    def test_multiple_response_tags(self):
        """Test when multiple <response> tags exist - should use first one."""
        text = "<response>First response</response> <response>Second response</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "First response", "Should extract content from first <response> tag"
    
    def test_nested_response_tags(self):
        """Test when response tags are nested."""
        text = "<response>Outer <response>Inner</response> content</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Outer <response>Inner</response> content", "Should extract until first closing tag"
    
    def test_response_tag_at_beginning(self):
        """Test when <response> tag is at the very beginning."""
        text = "<response>Answer at the start</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer at the start", "Should handle response tag at beginning"
    
    def test_response_tag_at_end(self):
        """Test when </response> tag is at the very end."""
        text = "Some reasoning. <response>Answer at the end</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer at the end", "Should handle response tag at end"
    
    def test_empty_response_content(self):
        """Test when response tags contain only whitespace."""
        text = "Reasoning. <response>   </response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "", "Should return empty string for whitespace-only content"
    
    def test_whitespace_around_response_content(self):
        """Test that whitespace around response content is stripped."""
        text = "Reasoning. <response>  Answer with spaces  </response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with spaces", "Should strip leading and trailing whitespace"
    
    def test_response_tag_with_newlines(self):
        """Test response tags containing newlines."""
        text = "Reasoning.\n<response>Answer\nwith\nnewlines</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer\nwith\nnewlines", "Should preserve newlines within response content"
    
    def test_response_tag_with_special_characters(self):
        """Test response tags with special characters."""
        text = "Reasoning. <response>Answer with @#$%^&*() characters!</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with @#$%^&*() characters!", "Should handle special characters"
    
    def test_response_tag_with_html_like_content(self):
        """Test response tags containing HTML-like content."""
        text = "Reasoning. <response>Answer with <b>bold</b> and <i>italic</i> text</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with <b>bold</b> and <i>italic</i> text", "Should preserve HTML-like content"
    
    def test_case_sensitive_tags(self):
        """Test that tags are case sensitive."""
        text = "Reasoning. <RESPONSE>Uppercase tag</RESPONSE>"
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should not match uppercase tags"
    
    def test_partial_tag_match(self):
        """Test that partial tag matches don't trigger extraction."""
        text = "Reasoning. <response_tag>This should not be extracted</response_tag>"
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should not match partial tag names"
    
    def test_multiple_closing_tags(self):
        """Test when multiple </response> tags exist."""
        text = "<response>Content</response> extra </response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Content", "Should stop at first closing tag"
    
    def test_response_tag_with_reasoning_tags(self):
        """Test response tags that contain reasoning tags."""
        text = "<reasoning>Chain of thought</reasoning> <response>Final answer with <reasoning>more thought</reasoning></response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Final answer with <reasoning>more thought</reasoning>", "Should extract response content including nested tags"
    
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
    
    def test_response_tag_with_unicode(self):
        """Test response tags with unicode characters."""
        text = "Reasoning. <response>Answer with unicode: 🚀🌟🎉</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with unicode: 🚀🌟🎉", "Should handle unicode characters"
    
    def test_response_tag_with_quotes(self):
        """Test response tags containing quotes."""
        text = '<response>Answer with "quotes" and \'single quotes\'</response>'
        result = AssessorModel._strip_reasoning(text)
        assert result == 'Answer with "quotes" and \'single quotes\'', "Should handle quotes properly"
    
    def test_response_tag_with_backslashes(self):
        """Test response tags containing backslashes."""
        text = "<response>Answer with \\backslashes\\ and \\n newlines</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with \\backslashes\\ and \\n newlines", "Should handle backslashes"
    
    def test_response_tag_with_brackets(self):
        """Test response tags containing various bracket types."""
        text = "<response>Answer with [brackets], {braces}, and (parentheses)</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with [brackets], {braces}, and (parentheses)", "Should handle various bracket types"
    
    def test_response_tag_with_angle_brackets(self):
        """Test response tags containing angle brackets that might be confused with tags."""
        text = "<response>Answer with < and > symbols</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with < and > symbols", "Should handle angle brackets within content"
    
    def test_response_tag_with_forward_slash(self):
        """Test response tags containing forward slashes."""
        text = "<response>Answer with /forward/slashes/</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with /forward/slashes/", "Should handle forward slashes"
    
    def test_response_tag_with_backward_slash(self):
        """Test response tags containing backward slashes."""
        text = "<response>Answer with \\backward\\slashes\\</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer with \\backward\\slashes\\", "Should handle backward slashes"
    
    def test_response_tag_with_tabs(self):
        """Test response tags containing tab characters."""
        text = "<response>Answer\twith\ttabs</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer\twith\ttabs", "Should preserve tab characters"
    
    def test_response_tag_with_carriage_returns(self):
        """Test response tags containing carriage return characters."""
        text = "<response>Answer\rwith\rcarriage\rreturns</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer\rwith\rcarriage\rreturns", "Should preserve carriage return characters"
    
    def test_response_tag_with_mixed_whitespace(self):
        """Test response tags with mixed whitespace characters."""
        text = "<response>Answer\twith\nmixed\rwhitespace</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer\twith\nmixed\rwhitespace", "Should preserve mixed whitespace characters"
    
    def test_response_tag_with_numbers_and_symbols(self):
        """Test response tags containing numbers and mathematical symbols."""
        text = "<response>Answer: 2 + 2 = 4, π ≈ 3.14159, 10^3 = 1000</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Answer: 2 + 2 = 4, π ≈ 3.14159, 10^3 = 1000", "Should handle numbers and mathematical symbols"
    
    def test_response_tag_with_urls(self):
        """Test response tags containing URLs."""
        text = "<response>Check out https://example.com and http://test.org/path?param=value</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Check out https://example.com and http://test.org/path?param=value", "Should handle URLs"
    
    def test_response_tag_with_email_addresses(self):
        """Test response tags containing email addresses."""
        text = "<response>Contact us at user@example.com or support@test.org</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Contact us at user@example.com or support@test.org", "Should handle email addresses"
    
    def test_response_tag_with_json_like_content(self):
        """Test response tags containing JSON-like content."""
        text = '<response>{"key": "value", "array": [1, 2, 3], "nested": {"inner": "data"}}</response>'
        result = AssessorModel._strip_reasoning(text)
        assert result == '{"key": "value", "array": [1, 2, 3], "nested": {"inner": "data"}}', "Should handle JSON-like content"
    
    def test_response_tag_with_xml_like_content(self):
        """Test response tags containing XML-like content."""
        text = "<response><data><item>value</item><item>another</item></data></response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "<data><item>value</item><item>another</item></data>", "Should handle XML-like content"
    
    def test_response_tag_with_code_blocks(self):
        """Test response tags containing code blocks."""
        text = "<response>Here's some code:\n```python\ndef hello():\n    print('Hello, World!')\n```</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Here's some code:\n```python\ndef hello():\n    print('Hello, World!')\n```", "Should handle code blocks"
    
    def test_response_tag_with_markdown(self):
        """Test response tags containing markdown formatting."""
        text = "<response>**Bold text**, *italic text*, `code`, and [link](url)</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "**Bold text**, *italic text*, `code`, and [link](url)", "Should handle markdown formatting"
    
    def test_response_tag_with_very_long_content(self):
        """Test response tags with very long content."""
        long_content = "A" * 10000  # 10,000 characters
        text = f"<response>{long_content}</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == long_content, "Should handle very long content"
    
    def test_response_tag_with_mixed_language_content(self):
        """Test response tags with mixed language content."""
        text = "<response>English text, 中文文本, Español texto, Français texte</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "English text, 中文文本, Español texto, Français texte", "Should handle mixed language content"
    
    def test_response_tag_with_emoji_and_symbols(self):
        """Test response tags with emojis and various symbols."""
        text = "<response>Reaction: 😂🎉🔥💯 and symbols: ©®™€£¥¢</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Reaction: 😂🎉🔥💯 and symbols: ©®™€£¥¢", "Should handle emojis and symbols"
    
    def test_response_tag_with_control_characters(self):
        """Test response tags with control characters."""
        text = "<response>Text with\x00null\x01start\x02end\x03text</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Text with\x00null\x01start\x02end\x03text", "Should handle control characters"
    
    def test_response_tag_with_very_short_content(self):
        """Test response tags with very short content."""
        text = "<response>Hi</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Hi", "Should handle very short content"
    
    def test_response_tag_with_only_punctuation(self):
        """Test response tags containing only punctuation."""
        text = "<response>!@#$%^&*()_+-=[]{}|;':\",./<>?</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "!@#$%^&*()_+-=[]{}|;':\",./<>?", "Should handle punctuation-only content"
    
    def test_response_tag_with_leading_trailing_response_tags(self):
        """Test when text starts and ends with response tags."""
        text = "<response>content</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "content", "Should handle text that is entirely wrapped in response tags"
    
    def test_response_tag_with_spaces_in_tag_names(self):
        """Test that spaces in tag names don't match."""
        text = "Reasoning. < response >Content< /response >"
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should not match tags with spaces"
    
    def test_response_tag_with_uppercase_closing_tag(self):
        """Test that uppercase closing tag doesn't match."""
        text = "<response>Content</RESPONSE>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "Content", "Should return content when closing tag case doesn't match"
    
    def test_response_tag_with_mixed_case_tags(self):
        """Test with mixed case tags."""
        text = "<Response>Content</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should not match mixed case tags"
    
    def test_response_tag_with_extra_spaces_in_tags(self):
        """Test that extra spaces in tags don't match."""
        text = "< response >Content</ response >"
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should not match tags with extra spaces"
    
    def test_response_tag_with_newlines_in_tags(self):
        """Test that newlines in tags don't match."""
        text = "<\nresponse\n>Content</\nresponse\n>"
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should not match tags with newlines"
    
    def test_response_tag_with_tabs_in_tags(self):
        """Test that tabs in tags don't match."""
        text = "<\tresponse\t>Content</\tresponse\t>"
        result = AssessorModel._strip_reasoning(text)
        assert result == text, "Should not match tags with tabs"
    
    def test_response_tag_with_comment_like_content(self):
        """Test response tags containing comment-like content."""
        text = "<response>This is <!-- a comment --> in the content</response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "This is <!-- a comment --> in the content", "Should handle comment-like content"
    
    def test_response_tag_with_doctype_like_content(self):
        """Test response tags containing DOCTYPE-like content."""
        text = "<response><!DOCTYPE html><html><head></head><body></body></html></response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "<!DOCTYPE html><html><head></head><body></body></html>", "Should handle DOCTYPE-like content"
    
    def test_response_tag_with_cdata_like_content(self):
        """Test response tags containing CDATA-like content."""
        text = "<response><![CDATA[This is CDATA content]]></response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "<![CDATA[This is CDATA content]]>", "Should handle CDATA-like content"
    
    def test_response_tag_with_processing_instruction_like_content(self):
        """Test response tags containing processing instruction-like content."""
        text = "<response><?xml version='1.0' encoding='UTF-8'?></response>"
        result = AssessorModel._strip_reasoning(text)
        assert result == "<?xml version='1.0' encoding='UTF-8'?>", "Should handle processing instruction-like content" 