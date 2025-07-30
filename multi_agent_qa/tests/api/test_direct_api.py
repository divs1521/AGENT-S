#!/usr/bin/env python3

import json
import openai

# Load the API key
with open('config/api_keys.json', 'r') as f:
    config = json.load(f)

openai_key = config.get('openai', '')
print(f"Testing OpenAI key: {openai_key[:10]}...")

# Check if it's a GitHub token
is_github_token = openai_key.startswith('github_pat_') or openai_key.startswith('ghp_')
print(f"Is GitHub token: {is_github_token}")

try:
    if is_github_token:
        print("Using GitHub endpoint...")
        client = openai.OpenAI(
            api_key=openai_key,
            base_url="https://models.inference.ai.azure.com"
        )
        # Try with a GitHub-supported model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        
        if response and response.choices:
            print(f"✅ SUCCESS: {response.choices[0].message.content}")
        else:
            print("❌ No response from GitHub API")
    else:
        print("Using regular OpenAI endpoint...")
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        
        if response and response.choices:
            print(f"✅ SUCCESS: {response.choices[0].message.content}")
        else:
            print("❌ No response from OpenAI API")
            
except Exception as e:
    error_msg = str(e)
    print(f"❌ ERROR: {error_msg}")
    if "incorrect api key" in error_msg.lower() or "invalid" in error_msg.lower():
        print("Diagnosis: Invalid API key")
    elif "quota" in error_msg.lower():
        print("Diagnosis: Quota exceeded")
    elif "model" in error_msg.lower() and "not found" in error_msg.lower():
        print("Diagnosis: Model not available for this key type")
