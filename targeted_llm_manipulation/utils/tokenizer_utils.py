def assert_padding_side_left(tokenizer):
    """
    Assert that the tokenizer's padding side is set to 'left'.
    
    Args:
        tokenizer: The tokenizer to check
        
    Raises:
        AssertionError: If the padding side is not 'left'
    """
    assert tokenizer.padding_side == "left", f"Tokenizer padding_side must be 'left', got '{tokenizer.padding_side}'" 