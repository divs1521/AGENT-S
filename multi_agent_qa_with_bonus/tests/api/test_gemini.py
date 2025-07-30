#!/usr/bin/env python3
"""Test script to verify Gemini API key functionality."""

import os
import sys

def test_gemini_api():
    """Test if Gemini API key works."""
    try:
        # Try to import the required library
        import google.generativeai as genai
        print("✅ google.generativeai imported successfully")
        
        # Get API key from environment or user input
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            api_key = input("Enter your Gemini API key: ").strip()
        
        if not api_key:
            print("❌ No API key provided")
            return False
        
        # Configure the API
        genai.configure(api_key=api_key)
        print(f"✅ API key configured: {api_key[:10]}...")
        
        # Test with a simple call
        print("🧪 Testing API call...")
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello, please respond with 'API test successful'")
        
        if response and response.text:
            print(f"✅ API call successful!")
            print(f"📝 Response: {response.text}")
            return True
        else:
            print("❌ API call failed: No response")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import google.generativeai: {e}")
        print("💡 Try installing: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Gemini API...")
    success = test_gemini_api()
    
    if success:
        print("\n🎉 Gemini API is working correctly!")
        print("✅ Your API key should work in the QA system")
    else:
        print("\n❌ Gemini API test failed")
        print("🔧 Please check your API key and internet connection")
    
    sys.exit(0 if success else 1)
