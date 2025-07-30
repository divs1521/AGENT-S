#!/usr/bin/env python3

import openai
import sys

# Your GitHub token
token = "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"

# Try different endpoints
endpoints = [
    "https://models.inference.ai.azure.com",
    "https://api.github.com/v1",
    "https://models.inference.ai.azure.com/v1",
    "https://openai-compatible.api.github.com/v1"
]

for endpoint in endpoints:
    print(f"\n--- Testing endpoint: {endpoint} ---")
    try:
        client = openai.OpenAI(
            api_key=token,
            base_url=endpoint
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print(f"✅ SUCCESS: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

print("\n--- Testing without base_url (regular OpenAI) ---")
try:
    client = openai.OpenAI(api_key=token)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print(f"✅ SUCCESS: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
