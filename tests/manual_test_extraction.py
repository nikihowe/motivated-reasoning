# First, load in the jsonl called "extraction_test.jsonl"

import json
from motivated_reasoning.environment.assessor_model import AssessorModel

# Load and process the JSONL file
to_test = [8]
with open("./inference_output/extraction_test.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i not in to_test:
            continue
        data = json.loads(line)
        print("Original data:")
        print(data)
        
        # Extract the response using the AssessorModel class
        result = AssessorModel._strip_reasoning(data["response"])
        print("\nExtracted response:")
        print(result)
        print("\n" + "="*50 + "\n")
