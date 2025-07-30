"""Base module for multi-agent QA system with Agent-S and Android World integration."""

from typing import Dict, Optional, Any
import sys
import os
import logging

# Import our integration modules
from .agent_s_integration import agent_s
from .android_world_integration import android_world

# Mock LMMAgent class if imports fail
class MockLMMAgent:
    """Mock LLM Agent for development/testing purposes."""
    
    def __init__(self, engine_params: Dict):
        self.engine_params = engine_params
        self.messages = []
        self.system_prompt = ""
        
    def add_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        
    def add_message(self, content: str, role: str = "user", image_content: Any = None):
        self.messages.append({
            "role": role,
            "content": content,
            "image": image_content
        })
        
    def generate_response(self) -> str:
        """Mock response generation with proper JSON formatting."""
        # Try real LLM call first if API keys are available
        try:
            return self._try_real_llm_call()
        except Exception as e:
            # If real LLM fails, fall back to mock
            print(f"Falling back to mock due to: {e}")
            return self._generate_mock_json_response()
    
    def _generate_mock_json_response(self) -> str:
        """Generate a properly formatted JSON mock response."""
        # Get the last message to understand what kind of response is needed
        if self.messages:
            last_message = self.messages[-1]['content']
            
            # Check EXECUTOR requests FIRST (more specific detection)
            if ('specific action plan in JSON format' in last_message or 
                  'plan the specific action needed' in last_message.lower() or
                  ('analyze the UI' in last_message.lower() and '"coordinates"' in last_message)):
                return '''```json
{
    "action_type": "click",
    "target_element_id": "element_12",
    "coordinates": {"x": 540, "y": 960},
    "text": "",
    "direction": "",
    "reasoning": "Clicking on the identified target element to accomplish the subgoal",
    "confidence": 0.92
}
```'''
            
            # Check if this looks like a planner request (AFTER executor check)
            elif ('Create a detailed QA test plan' in last_message or
                  'subgoals' in last_message.lower() or
                  ('action_type' in last_message and 'navigate|tap|input' in last_message)):
                return '''```json
{
    "task_id": "mock_task_001",
    "description": "Mock QA task execution",
    "subgoals": [
        {
            "id": 1,
            "description": "Open application or navigate to relevant section",
            "action_type": "navigate",
            "target_element": "app_launcher",
            "expected_outcome": "Application opens successfully",
            "dependencies": []
        },
        {
            "id": 2,
            "description": "Locate target information or interface", 
            "action_type": "tap",
            "target_element": "search_button",
            "expected_outcome": "Search interface becomes visible",
            "dependencies": [1]
        },
        {
            "id": 3,
            "description": "Interact with interface to complete task",
            "action_type": "input",
            "target_element": "search_field",
            "expected_outcome": "Task completed successfully",
            "dependencies": [2]
        }
    ],
    "success_criteria": "Task completed with all steps executed successfully",
    "confidence": 0.85,
    "estimated_time": 30,
    "estimated_steps": 3
}
```'''
            
            # Check if this looks like an executor request (action planning) - more specific detection
            elif ('specific action plan in JSON format' in last_message or 
                  '"action_type"' in last_message or 
                  '"coordinates"' in last_message or
                  'plan the specific action needed' in last_message.lower()):
                return '''```json
{
    "action_type": "click",
    "target_element_id": "element_12",
    "coordinates": {"x": 540, "y": 960},
    "text": "",
    "direction": "",
    "reasoning": "Clicking on the identified target element to accomplish the subgoal",
    "confidence": 0.92
}
```'''
            
            # General executor request (fallback for action words)
            elif any(word in last_message.lower() for word in ['action', 'execute', 'perform', 'tap', 'click', 'analyze', 'ui_state']):
                return '''```json
{
    "action_type": "click",
    "coordinates": [540, 960],
    "target_element": "main interface element",
    "reasoning": "Performing the requested action based on the current context",
    "confidence": 0.9
}
```'''
            
            # Check if this looks like a verification request
            elif any(word in last_message.lower() for word in ['verify', 'check', 'validate', 'confirm']):
                return '''```json
{
    "verification_result": "success",
    "findings": [
        "Mock execution completed successfully",
        "UI state progression detected as expected",
        "Test step accomplished in mock environment"
    ],
    "confidence": 0.92,
    "recommendations": ["Continue with next test step"],
    "issues_found": []
}
```'''
        
        # Default response
        return '''```json
{
    "response": "Task completed successfully using mock environment",
    "status": "success",
    "confidence": 0.85
}
```'''
    
    def _try_real_llm_call(self) -> str:
        """Try to make a real LLM call if API keys are available."""
        import os
        
        # Try OpenAI FIRST (back to original priority)
        openai_key = os.environ.get('OPENAI_API_KEY')
        if (openai_key and openai_key.strip() and 
            not openai_key.startswith('sk-jW9z8Tc5bD4x3Yo2Rv1uAq7GhSi0VmEnKf2lOp3Hq6Cr5XeP')):
            try:
                import openai
                
                # Check if it's a GitHub token
                is_github_token = openai_key.startswith('github_pat_') or openai_key.startswith('ghp_')
                
                if is_github_token:
                    # Use GitHub's OpenAI API endpoint
                    client = openai.OpenAI(
                        api_key=openai_key,
                        base_url="https://models.inference.ai.azure.com"
                    )
                    messages = [{"role": "system", "content": self.system_prompt}]
                    messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in self.messages])
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    print("✅ Using GitHub Models API")
                    return response.choices[0].message.content
                else:
                    # Use regular OpenAI API
                    client = openai.OpenAI(api_key=openai_key)
                    messages = [{"role": "system", "content": self.system_prompt}]
                    messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in self.messages])
                    
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    print("✅ Using OpenAI API")
                    return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if ("401" in error_str or "invalid" in error_str.lower() or 
                    "authentication" in error_str.lower() or "api key" in error_str.lower()):
                    print(f"❌ OpenAI API key invalid, trying next provider: {e}")
                    # Continue to try other APIs instead of failing
                elif "429" in error_str or "Rate" in error_str or "limit" in error_str.lower():
                    print(f"⚠️ OpenAI API rate limit exceeded, trying next provider: {e}")
                    # Continue to try other APIs instead of failing
                else:
                    print(f"❌ OpenAI API call failed, trying next provider: {e}")
                    # Continue to try other APIs instead of failing
        
        # Try Google Gemini SECOND (fallback)
        google_key = os.environ.get('GOOGLE_API_KEY')
        if google_key and google_key.strip() and not google_key.startswith('sk-'):
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                
                prompt = f"{self.system_prompt}\n\n" + "\n".join([msg["content"] for msg in self.messages])
                
                # Try Gemini Flash first (free tier)
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt)
                    if response.text:
                        print("✅ Using Google Gemini API")
                        return response.text
                except Exception as flash_error:
                    # Fallback to Gemini Pro
                    try:
                        model = genai.GenerativeModel('gemini-1.5-pro')
                        response = model.generate_content(prompt)
                        if response.text:
                            print("✅ Using Google Gemini Pro API")
                            return response.text
                    except Exception as pro_error:
                        print(f"Both Gemini models failed: {flash_error}, {pro_error}")
                        # Don't raise - continue to try other APIs
                        pass
            except Exception as e:
                print(f"Google Gemini API call failed: {e}")
                # Continue to try other APIs instead of failing
        
        # Try Anthropic
        anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
        if anthropic_key and anthropic_key.strip():
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                
                messages = [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    system=self.system_prompt,
                    messages=messages
                )
                print("✅ Using Anthropic API")
                return response.content[0].text
            except Exception as e:
                error_str = str(e)
                if ("401" in error_str or "invalid" in error_str.lower() or 
                    "authentication" in error_str.lower() or "api key" in error_str.lower()):
                    print(f"❌ Anthropic API key invalid: {e}")
                elif "429" in error_str or "Rate" in error_str or "limit" in error_str.lower():
                    print(f"⚠️ Anthropic API rate limit exceeded: {e}")
                else:
                    print(f"❌ Anthropic API call failed: {e}")
        
        # If all real APIs fail, fall back to mock
        print("🔄 All API providers failed or unavailable, using mock responses")
        raise Exception("No valid API keys available - falling back to mock")

# Import Agent-S and try to load LMMAgent
try:
    # Try Agent-S integration first
    if agent_s.is_available():
        LMMAgent = agent_s.LMMAgent
        print("✅ Using Agent-S LMMAgent")
    else:
        raise ImportError("Agent-S not available")
except ImportError:
    try:
        from gui_agents.s2.core.mllm import LMMAgent
    except ImportError:
        try:
            from gui_agents.s1.mllm.MultimodalAgent import LMMAgent
        except ImportError:
            logging.warning("Could not import LMMAgent, using mock implementation")
            LMMAgent = MockLMMAgent


class BaseQAModule:
    """Base class for all QA system modules."""
    
    def __init__(self, engine_params: Dict, platform: str = "android"):
        """Initialize the base module.
        
        Args:
            engine_params: Configuration parameters for the LLM engine
            platform: Operating system platform (android by default)
        """
        self.engine_params = engine_params
        self.platform = platform
        
    def _create_agent(
        self, system_prompt: str = None, engine_params: Optional[Dict] = None
    ):
        """Create a new LMMAgent instance.
        
        Args:
            system_prompt: System prompt for the agent
            engine_params: Optional engine parameters to override defaults
            
        Returns:
            Configured LMMAgent instance
        """
        params = engine_params or self.engine_params
        
        # Force mock if explicitly requested or if no API keys available
        use_mock = params.get('mock', False)
        
        if use_mock:
            agent = MockLMMAgent(params)
        else:
            # Try Agent-S integration first
            if agent_s.is_available():
                agent = agent_s.create_lmm_agent(params)
                if agent is None:
                    # Fallback to MockLMMAgent if Agent-S creation fails
                    agent = MockLMMAgent(params)
            else:
                # Try direct import as fallback
                try:
                    agent = LMMAgent(params)
                except Exception as e:
                    print(f"Failed to create LMMAgent directly: {e}")
                    agent = MockLMMAgent(params)
            
        if system_prompt:
            agent.add_system_prompt(system_prompt)
        return agent
