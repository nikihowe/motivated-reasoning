from motivated_reasoning.environment.assessor_model import AssessorModel
from motivated_reasoning.reasoning_tags import render_reasoning_tags, render_reasoning_tags_in_messages


def test_render_reasoning_tags_to_qwen_style():
    text = "<thinking>reason</thinking><answer>answer</answer>"

    assert render_reasoning_tags(text, "think") == "<think>reason</think><answer>answer</answer>"


def test_render_reasoning_tags_in_messages_copies_messages():
    messages = [{"role": "assistant", "content": "<thinking>r</thinking>"}]

    rendered = render_reasoning_tags_in_messages(messages, "think")

    assert rendered == [{"role": "assistant", "content": "<think>r</think>"}]
    assert messages == [{"role": "assistant", "content": "<thinking>r</thinking>"}]


def test_formatting_penalty_accepts_configured_qwen_tag():
    text = "<think>reason</think><answer>answer</answer>"

    assert AssessorModel._get_formatting_penalty(text, reasoning_tag="think") == 0.0


def test_formatting_penalty_keeps_legacy_thinking_compatible():
    text = "<thinking>reason</thinking><answer>answer</answer>"

    assert AssessorModel._get_formatting_penalty(text, reasoning_tag="think") == 0.0


def test_split_reasoning_accepts_qwen_tag():
    text = "<think>reason</think><answer>answer</answer>"

    reasoning, answer, full_text = AssessorModel._split_reasoning(text, reasoning_tag="think")

    assert reasoning == "reason"
    assert answer == "answer"
    assert full_text == text
