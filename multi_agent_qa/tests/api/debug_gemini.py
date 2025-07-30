#!/usr/bin/env python3
"""Debug script for API key testing."""

import os
import sys

def test_gemini_simple():
    """Simple Gemini API test."""
    try:
        import google.generativeai as genai
        print("✅ Successfully imported google.generativeai")
        
        # Get API key from environment
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("❌ No GOOGLE_API_KEY environment variable found")
            return False
        
        print(f"🔑 Found API key: {api_key[:10]}...")
        
        # Configure
        genai.configure(api_key=api_key)
        print("✅ API configured")
        
        # Test call
        model = genai.GenerativeModel('gemini-pro')
        print("✅ Model created")
        
        response = model.generate_content("Hello")
        print("✅ API call completed")
        
        if response and response.text:
            print(f"✅ Response received: {response.text[:100]}...")
            return True
        else:
            print("❌ No response text")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini API...")
    success = test_gemini_simple()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
