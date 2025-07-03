#!/usr/bin/env python3

from targeted_llm_manipulation.environment.assessor_model import AssessorModel

def test_split_reasoning():
    """Test the _split_reasoning function with various examples."""
    
    test_cases = [
        {
            "name": "Simple JSON with string response",
            "input": 'Let me think about this step by step. First, I need to consider the ethical implications...\n\n{"response": "I cannot help with that request."}',
            "expected_reasoning": "Let me think about this step by step. First, I need to consider the ethical implications...",
            "expected_response": "I cannot help with that request."
        },
        {
            "name": "JSON with object response",
            "input": 'I need to analyze this carefully...\n\n{"response": {"answer": "No", "reason": "This is harmful"}}',
            "expected_reasoning": "I need to analyze this carefully...",
            "expected_response": '{"answer":"No","reason":"This is harmful"}'
        },
        {
            "name": "No JSON structure",
            "input": "I cannot help with that request because it would be harmful.",
            "expected_reasoning": "I cannot help with that request because it would be harmful.",
            "expected_response": ""
        },
        {
            "name": "Multiple JSON objects",
            "input": 'Let me think...\n\n{"response": "I cannot help"} and also {"other": "data"}',
            "expected_reasoning": "Let me think...",
            "expected_response": "I cannot help"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        print(f"Input: {test_case['input'][:100]}...")
        
        reasoning, response, full_text = AssessorModel._split_reasoning(test_case['input'])
        
        print(f"Reasoning: {reasoning[:50]}...")
        print(f"Response: {response[:50]}...")
        print(f"Full text: {full_text[:50]}...")
        
        # Basic validation
        if reasoning == test_case['expected_reasoning']:
            print("✓ Reasoning matches expected")
        else:
            print(f"✗ Reasoning mismatch. Expected: {test_case['expected_reasoning'][:50]}...")
            
        if response == test_case['expected_response']:
            print("✓ Response matches expected")
        else:
            print(f"✗ Response mismatch. Expected: {test_case['expected_response'][:50]}...")
        
        print("-" * 50)

def test_consistency_between_functions():
    """Test that _split_reasoning and _strip_reasoning give consistent results for response extraction."""
    
    test_cases = [
        {
            "name": "Simple JSON with string response",
            "input": 'Let me think about this step by step. First, I need to consider the ethical implications...\n\n{"response": "I cannot help with that request."}'
        },
        {
            "name": "JSON with object response",
            "input": 'I need to analyze this carefully...\n\n{"response": {"answer": "No", "reason": "This is harmful"}}'
        },
        {
            "name": "No JSON structure",
            "input": "I cannot help with that request because it would be harmful."
        },
        {
            "name": "Multiple JSON objects",
            "input": 'Let me think...\n\n{"response": "I cannot help"} and also {"other": "data"}'
        },
        {
            "name": "Complex reasoning with multiple paragraphs",
            "input": '''Let me analyze this step by step.

First, I need to understand what the user is asking for. This appears to be a request that could potentially be harmful.

Second, I should consider the ethical implications. Any action that could cause harm to others should be avoided.

Finally, I must provide a clear and firm response.

{"response": "I cannot and will not help with this request as it could cause harm to others."}'''
        },
        {
            "name": "Response with escaped quotes",
            "input": 'I need to be careful here...\n\n{"response": "I cannot help because it would be \"harmful\" to others."}'
        },
        {
            "name": "Nested JSON in response",
            "input": 'Let me think...\n\n{"response": {"decision": "no", "explanation": {"reason": "harmful", "details": "This could hurt people"}}}'
        },
        {
            "name": "Response with newlines",
            "input": 'I need to consider this...\n\n{"response": "I cannot help.\n\nThis request is harmful.\n\nPlease reconsider."}'
        },
        {
            "name": "Empty reasoning",
            "input": '{"response": "I cannot help with that request."}'
        },
        {
            "name": "Very long reasoning",
            "input": 'This is a very long reasoning section that goes on and on about various considerations and ethical implications and step-by-step analysis of the situation at hand...\n\n{"response": "No."}'
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING CONSISTENCY BETWEEN _split_reasoning AND _strip_reasoning")
    print("="*80)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        print(f"Input: {test_case['input'][:80]}...")
        
        # Get results from both functions
        reasoning, response_from_split, full_text = AssessorModel._split_reasoning(test_case['input'])
        response_from_strip = AssessorModel._strip_reasoning(test_case['input'])
        
        print(f"Response from _split_reasoning: {response_from_split[:50]}...")
        print(f"Response from _strip_reasoning: {response_from_strip[:50]}...")
        
        # Check if responses match
        if response_from_split == response_from_strip:
            print("✓ Responses match between functions")
        else:
            print("✗ Responses DO NOT match between functions!")
            print(f"  _split_reasoning: '{response_from_split}'")
            print(f"  _strip_reasoning: '{response_from_strip}'")
            all_passed = False
        
        # Additional checks for _split_reasoning
        print(f"Reasoning extracted: {reasoning[:50]}...")
        print(f"Full text preserved: {full_text == test_case['input']}")
        
        print("-" * 50)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! _split_reasoning and _strip_reasoning are consistent.")
    else:
        print("\n❌ SOME TESTS FAILED! There are inconsistencies between the functions.")
    
    return all_passed

def test_edge_cases():
    """Test edge cases and potential failure modes."""
    
    edge_cases = [
        {
            "name": "Empty input",
            "input": ""
        },
        {
            "name": "Only whitespace",
            "input": "   \n\t   "
        },
        {
            "name": "Malformed JSON",
            "input": 'Let me think...\n\n{"response": "unclosed quote}'
        },
        {
            "name": "JSON without response field",
            "input": 'Let me think...\n\n{"other_field": "value"}'
        },
        {
            "name": "Multiple response fields",
            "input": 'Let me think...\n\n{"response": "first"} and {"response": "second"}'
        },
        {
            "name": "Very long response",
            "input": f'Let me think...\n\n{{"response": "{("very long " * 1000)}response"}}'
        },
        {
            "name": "Unicode characters",
            "input": 'Let me think...\n\n{"response": "I cannot help with 🚫 harmful 🚫 requests"}'
        }
    ]
    
    print("\n" + "="*80)
    print("TESTING EDGE CASES")
    print("="*80)
    
    for i, edge_case in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: {edge_case['name']}")
        print(f"Input: {edge_case['input'][:80]}...")
        
        try:
            reasoning, response, full_text = AssessorModel._split_reasoning(edge_case['input'])
            strip_response = AssessorModel._strip_reasoning(edge_case['input'])
            
            print(f"✓ Functions completed without error")
            print(f"  Reasoning length: {len(reasoning)}")
            print(f"  Response length: {len(response)}")
            print(f"  Responses match: {response == strip_response}")
            
        except Exception as e:
            print(f"✗ Error occurred: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    print("Running basic functionality tests...")
    test_split_reasoning()
    
    print("\nRunning consistency tests...")
    consistency_passed = test_consistency_between_functions()
    
    print("\nRunning edge case tests...")
    test_edge_cases()
    
    if consistency_passed:
        print("\n✅ All consistency tests passed! The functions work correctly together.")
    else:
        print("\n❌ Some consistency tests failed. Please review the implementation.") 