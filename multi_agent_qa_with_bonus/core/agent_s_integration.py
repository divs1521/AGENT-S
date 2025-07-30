"""Agent-S integration module for the multi-agent QA system."""

import sys
import os
import logging
from typing import Dict, Optional, Any

# Add Agent-S path
current_dir = os.path.dirname(__file__)
agent_s_path = os.path.join(current_dir, "..", "..", "Agent-S-main")
sys.path.insert(0, agent_s_path)

logger = logging.getLogger(__name__)

class AgentSIntegration:
    """Integration wrapper for Agent-S functionality."""
    
    def __init__(self):
        self.agent_s_available = False
        self.engine = None
        self.lmm_agent = None
        self._initialize_agent_s()
    
    def _initialize_agent_s(self):
        """Initialize Agent-S components."""
        try:
            # Ensure the path is correctly added
            import sys
            import os
            
            # Get the absolute path to Agent-S
            current_dir = os.path.dirname(__file__)
            agent_s_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Agent-S-main"))
            
            if agent_s_path not in sys.path:
                sys.path.insert(0, agent_s_path)
            
            # Try to import Agent-S components
            from gui_agents.s2.core.mllm import LMMAgent
            from gui_agents.s2.core.engine import LMMEngineOpenAI
            
            self.LMMAgent = LMMAgent
            self.LMMEngineOpenAI = LMMEngineOpenAI
            self.agent_s_available = True
            
            logger.info("✅ Agent-S components loaded successfully")
            print("✅ Agent-S integration initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️ Agent-S not available: {e}")
            print(f"⚠️ Agent-S not available: {e}")
            # Try to import from s1 as fallback
            try:
                from gui_agents.s1.mllm.MultimodalAgent import LMMAgent
                self.LMMAgent = LMMAgent
                self.agent_s_available = True
                print("✅ Agent-S S1 fallback initialized")
            except ImportError as e2:
                print(f"⚠️ Agent-S S1 fallback also failed: {e2}")
                self.agent_s_available = False
    
    def create_lmm_agent(self, engine_params: Dict) -> Optional[Any]:
        """Create an LMM Agent using Agent-S."""
        if not self.agent_s_available:
            return None
        
        try:
            # Convert our engine params to Agent-S format
            agent_s_params = self._convert_engine_params(engine_params)
            
            # Check if we're using GitHub token - Agent-S might not support custom base URLs
            import os
            openai_key = os.environ.get('OPENAI_API_KEY', '')
            is_github_token = openai_key.startswith('github_pat_') or openai_key.startswith('ghp_')
            
            if is_github_token:
                # For GitHub tokens, use our MockLMMAgent which handles the custom base URL correctly
                print("⚠️ Using MockLMMAgent for GitHub token (Agent-S doesn't support custom base URLs)")
                from .base_module import MockLMMAgent
                
                # Create a MockLMMAgent instead
                mock_agent = MockLMMAgent(engine_params)
                
                # Add the methods that Agent-S interface expects
                def get_response(messages=None):
                    if messages:
                        # Clear existing messages and add the new ones
                        mock_agent.messages = []
                        for msg in messages:
                            mock_agent.add_message(msg.get('content', ''), msg.get('role', 'user'))
                    return mock_agent.generate_response()
                
                mock_agent.get_response = get_response
                return mock_agent
            
            # For regular OpenAI keys, use Agent-S
            # Create Agent-S LMM Agent with proper configuration
            agent = self.LMMAgent(engine_params=agent_s_params)
            
            # Add compatibility wrapper methods
            def generate_response():
                # Use all messages for context
                return agent.get_response(messages=agent.messages)
            
            # Store original add_message method
            agent._original_add_message = agent.add_message
            
            def add_message(content, role="user", image_content=None):
                # Convert our format to Agent-S format
                agent._original_add_message(text_content=content, role=role, image_content=image_content)
            
            agent.generate_response = generate_response
            agent.add_message = add_message  # Override the original method
            
            logger.info("Created Agent-S LMM Agent with compatibility wrapper")
            return agent
        except Exception as e:
            logger.error(f"Failed to create Agent-S LMM Agent: {e}")
            return None
    
    def _convert_engine_params(self, params: Dict) -> Dict:
        """Convert our engine params to Agent-S format."""
        import os
        
        # Try OpenAI first (back to original priority)  
        openai_key = os.environ.get('OPENAI_API_KEY', '')
        if (openai_key and openai_key.strip() and 
            not openai_key.startswith('sk-jW9z8Tc5bD4x3Yo2Rv1uAq7GhSi0VmEnKf2lOp3Hq6Cr5XeP')):
            
            # Check if it's a GitHub token and set appropriate base URL
            if openai_key.startswith('ghp_') or openai_key.startswith('github_pat_'):
                return {
                    "engine_type": "openai",
                    "model": params.get("model", "gpt-4o-mini"),
                    "api_key": openai_key,
                    "api_base": "https://models.inference.ai.azure.com",
                    "temperature": params.get("temperature", 0.7),
                    "max_tokens": params.get("max_tokens", 1000)
                }
            else:
                return {
                    "engine_type": "openai", 
                    "model": params.get("model", "gpt-4"),
                    "api_key": openai_key,
                    "temperature": params.get("temperature", 0.7),
                    "max_tokens": params.get("max_tokens", 1000)
                }
        
        # Fallback to Google API 
        google_key = os.environ.get('GOOGLE_API_KEY', '')
        if google_key and google_key.strip():
            return {
                "engine_type": "google",
                "model": params.get("model", "gemini-1.5-flash"),
                "api_key": google_key,
                "temperature": params.get("temperature", 0.7),
                "max_tokens": params.get("max_tokens", 1000)
            }
        
        # Fallback to mock configuration if no valid API keys
        return {
            "engine_type": "mock",
            "model": "mock-model",
            "api_key": "mock-key",
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1000)
        }
    
    def create_engine(self, config: Dict) -> Optional[Any]:
        """Create an Agent-S Engine."""
        if not self.agent_s_available:
            return None
        
        try:
            engine = self.Engine(config)
            logger.info("Created Agent-S Engine")
            return engine
        except Exception as e:
            logger.error(f"Failed to create Agent-S Engine: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Agent-S is available."""
        return self.agent_s_available

# Global Agent-S integration instance
agent_s = AgentSIntegration()
