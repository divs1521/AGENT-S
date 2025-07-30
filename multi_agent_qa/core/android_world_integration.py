"""Android World integration module for the multi-agent QA system."""

import sys
import os
import logging
from typing import Dict, Optional, Any, List

# Add Android World path
current_dir = os.path.dirname(__file__)
android_world_path = os.path.join(current_dir, "..", "..", "android_world-main")
sys.path.insert(0, android_world_path)

logger = logging.getLogger(__name__)

class AndroidWorldIntegration:
    """Integration wrapper for Android World functionality."""
    
    def __init__(self):
        self.android_world_available = False
        self.env = None
        self.interface = None
        self._initialize_android_world()
    
    def _initialize_android_world(self):
        """Initialize Android World components."""
        try:
            # Ensure the path is correctly added
            import sys
            import os
            
            # Get the absolute path to Android World
            current_dir = os.path.dirname(__file__)
            android_world_path = os.path.abspath(os.path.join(current_dir, "..", "..", "android_world-main"))
            
            if android_world_path not in sys.path:
                sys.path.insert(0, android_world_path)
            
            # Try to import Android World components (skip audio-dependent modules for now)
            import android_world
            print(f"✅ Successfully imported android_world module")
            
            # Import specific components (avoid registry which imports audio modules)
            from android_world.env.interface import AsyncAndroidEnv
            from android_world.env.android_world_controller import AndroidWorldController
            
            self.AsyncAndroidEnv = AsyncAndroidEnv  # This is the actual interface
            self.AndroidWorldController = AndroidWorldController
            self.registry = None  # Skip registry for now due to audio dependencies
            self.android_world_available = True
            
            logger.info("✅ Android World components loaded successfully")
            print("✅ Android World integration initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️ Android World not available: {e}")
            print(f"⚠️ Android World not available: {e}")
            self.android_world_available = False
        except Exception as e:
            logger.warning(f"⚠️ Android World initialization failed: {e}")
            print(f"⚠️ Android World initialization failed: {e}")
            self.android_world_available = False
    
    def create_environment(self, config: Dict) -> Optional[Any]:
        """Create an Android World environment."""
        if not self.android_world_available:
            return None
        
        try:
            # For now, create a mock environment since we don't have a real Android device
            # In a real implementation, this would create an actual AndroidEnv
            class MockAndroidEnv:
                def __init__(self):
                    self.status = "ready"
                    self.device_info = {"width": 1080, "height": 1920}
                    
                def execute_adb_call(self, command):
                    return {"status": "success", "output": "Mock ADB response"}
                    
                def reset(self):
                    return {"status": "reset"}
                    
                def step(self, action):
                    return None, 0, False, {}
                    
                def close(self):
                    pass
                    
                def get_observation(self):
                    return {"screen": "mock_screen", "ui_tree": "mock_tree"}
            
            mock_env = MockAndroidEnv()
            
            # Create Android World controller with the mock environment
            # Skip controller for now due to complexity, just return the mock env
            logger.info("Created Android World mock environment")
            return mock_env
        except Exception as e:
            logger.error(f"Failed to create Android World controller: {e}")
            return None
    
    def create_interface(self, env) -> Optional[Any]:
        """Create an Android interface."""
        if not self.android_world_available or not env:
            return None
        
        try:
            # For Android World, the AsyncAndroidEnv is the interface
            interface = self.AsyncAndroidEnv(env)
            logger.info("Created Android World interface")
            return interface
        except Exception as e:
            logger.error(f"Failed to create Android World interface: {e}")
            return None
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available Android World tasks."""
        if not self.android_world_available or self.registry is None:
            # Return some common Android tasks as fallback
            return [
                "settings_wifi",
                "settings_bluetooth", 
                "camera_photo",
                "contacts_add",
                "messages_send",
                "calculator_basic"
            ]
        
        try:
            return list(self.registry.get_task_registry().keys())
        except Exception as e:
            logger.error(f"Failed to get available tasks: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if Android World is available."""
        return self.android_world_available

# Global Android World integration instance
android_world = AndroidWorldIntegration()
