"""Logging utilities for the multi-agent QA system."""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict, is_dataclass


class QALogger:
    """Custom logger for QA system activities."""
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """Initialize QA logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger("multi_agent_qa")
        self.logger.setLevel(log_level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        detailed_handler = logging.FileHandler(
            os.path.join(log_dir, f"qa_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        detailed_handler.setLevel(logging.DEBUG)
        detailed_handler.setFormatter(detailed_formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(detailed_handler)
        self.logger.addHandler(console_handler)
        
        # Create specialized loggers
        self._setup_agent_loggers()
        
    def _setup_agent_loggers(self):
        """Setup specialized loggers for each agent."""
        agents = ["planner", "executor", "verifier", "supervisor"]
        
        for agent in agents:
            agent_logger = logging.getLogger(f"multi_agent_qa.{agent}")
            agent_handler = logging.FileHandler(
                os.path.join(self.log_dir, f"{agent}_agent.log")
            )
            agent_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            agent_logger.addHandler(agent_handler)
    
    def log_episode_start(self, episode_id: str, task_description: str):
        """Log the start of a test episode."""
        self.logger.info(f"Episode {episode_id} started: {task_description}")
        
    def log_episode_end(self, episode_id: str, success: bool, duration: float):
        """Log the end of a test episode."""
        self.logger.info(f"Episode {episode_id} ended: Success={success}, Duration={duration:.2f}s")
    
    def log_agent_action(self, agent_name: str, action: str, details: Dict):
        """Log an agent action."""
        agent_logger = logging.getLogger(f"multi_agent_qa.{agent_name}")
        agent_logger.info(f"Action: {action} - Details: {json.dumps(details, default=str)}")
    
    def log_error(self, component: str, error: Exception, context: Dict = None):
        """Log an error with context."""
        error_msg = f"Error in {component}: {str(error)}"
        if context:
            error_msg += f" - Context: {json.dumps(context, default=str)}"
        self.logger.error(error_msg, exc_info=True)
    
    def log_metrics(self, metrics: Dict):
        """Log performance metrics."""
        metrics_logger = logging.getLogger("multi_agent_qa.metrics")
        metrics_logger.info(f"Metrics: {json.dumps(metrics, default=str)}")


def serialize_for_json(obj: Any) -> Any:
    """Serialize objects for JSON output.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return str(obj)


def save_structured_log(data: Dict, log_path: str):
    """Save structured data to a JSON log file.
    
    Args:
        data: Data to save
        log_path: Path to save the log file
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=serialize_for_json, ensure_ascii=False)


class PerformanceTracker:
    """Track performance metrics across test runs."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "performance_metrics.json")
        self.metrics_history = self._load_metrics_history()
    
    def _load_metrics_history(self) -> List[Dict]:
        """Load historical metrics data."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def record_episode_metrics(self, episode_id: str, metrics: Dict):
        """Record metrics for a test episode."""
        metrics_entry = {
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.metrics_history.append(metrics_entry)
        self._save_metrics_history()
    
    def _save_metrics_history(self):
        """Save metrics history to file."""
        save_structured_log(self.metrics_history, self.metrics_file)
    
    def get_performance_trends(self, metric_name: str, last_n: int = 10) -> Dict:
        """Get performance trends for a specific metric."""
        if not self.metrics_history:
            return {"trend": "no_data"}
        
        recent_metrics = self.metrics_history[-last_n:]
        values = [entry["metrics"].get(metric_name) for entry in recent_metrics if entry["metrics"].get(metric_name) is not None]
        
        if len(values) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple trend analysis
        avg_first_half = sum(values[:len(values)//2]) / (len(values)//2)
        avg_second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if avg_second_half > avg_first_half * 1.05:
            trend = "improving"
        elif avg_second_half < avg_first_half * 0.95:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_average": avg_second_half,
            "historical_average": avg_first_half,
            "sample_size": len(values)
        }
