#!/usr/bin/env python3
"""Test the JSON parsing fix for executor agent."""

import json
from agents.executor_agent import ExecutorAgent

# Test the JSON cleaning function
executor = ExecutorAgent({'model': 'gpt-4o-mini'})

# Simulate a problematic response with multiple JSON objects
test_response = """```json
{
    "action_type": "wait",
    "target_element_id": null,
    "coordinates": null,
    "text": null,
    "direction": null,
    "reasoning": "Waiting for the UI to stabilize",
    "confidence": 0.9
},
{
    "action_type": "swipe",
    "target_element_id": null,
    "coordinates": {"x": 540, "y": 1800},
    "text": null,
    "direction": "up",
    "reasoning": "Swiping up to access app drawer",
    "confidence": 0.85
}
```"""

try:
    cleaned = executor._clean_json_response(test_response)
    print("Cleaned response:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    parsed = json.loads(cleaned)
    print('SUCCESS: Extracted first JSON object')
    print('Action type:', parsed['action_type'])
    print('Reasoning:', parsed['reasoning'])
    print('Confidence:', parsed['confidence'])
except Exception as e:
    print('ERROR:', e)
