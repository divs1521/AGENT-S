"""
Flask web application for Multi-Agent QA System Frontend
Provides real-time monitoring of agents, execution logs, and history management.
"""

import os
import json
import uuid
import time
import logging
import dataclasses
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import logging
from dataclasses import asdict

from core.orchestrator import MultiAgentQAOrchestrator, QASystemConfig
from utils.config_utils import ConfigManager
from utils.logging_utils import QALogger

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qa-system-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for tracking agent status and execution
class AppState:
    def __init__(self):
        self.current_orchestrator: Optional[MultiAgentQAOrchestrator] = None
        self.execution_history: List[Dict] = []
        self.agent_status: Dict = {
            'planner': {'status': 'idle', 'last_action': None, 'progress': 0, 'decision': ''},
            'executor': {'status': 'idle', 'last_action': None, 'progress': 0, 'decision': ''},
            'verifier': {'status': 'idle', 'last_action': None, 'progress': 0, 'decision': ''},
            'supervisor': {'status': 'idle', 'last_action': None, 'progress': 0, 'decision': ''}
        }
        self.current_logs: List[Dict] = []
        self.api_keys: Dict[str, str] = {}
        self.test_running = False
        self.current_episode_id = None

def load_api_keys():
    """Load API keys from config file on startup."""
    config_file = os.path.join("config", "api_keys.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                keys = json.load(f)
                app_state.api_keys = keys
                
                # Set environment variables
                if keys.get('openai'):
                    os.environ['OPENAI_API_KEY'] = keys['openai']
                    print(f"✅ Loaded OpenAI API key: {keys['openai'][:10]}...")
                if keys.get('anthropic'):
                    os.environ['ANTHROPIC_API_KEY'] = keys['anthropic']
                    print(f"✅ Loaded Anthropic API key: {keys['anthropic'][:10]}...")
                if keys.get('google'):
                    os.environ['GOOGLE_API_KEY'] = keys['google']
                    print(f"✅ Loaded Google API key: {keys['google'][:10]}...")
                    
                print("✅ API keys loaded from config/api_keys.json")
                return True
        except Exception as e:
            print(f"❌ Error loading API keys: {e}")
    return False

# Global app state
app_state = AppState()

# Load API keys on startup
load_api_keys()

def load_execution_history():
    """Load execution history from disk."""
    history_dir = "history"
    if not os.path.exists(history_dir):
        return []
    
    history = []
    for folder in sorted(os.listdir(history_dir), reverse=True):
        folder_path = os.path.join(history_dir, folder)
        if os.path.isdir(folder_path):
            summary_file = os.path.join(folder_path, "summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        summary['folder'] = folder
                        history.append(summary)
                except Exception as e:
                    print(f"Error loading history {folder}: {e}")
    
    return history

def safe_dataclass_to_dict(obj):
    """Safely convert dataclass to dictionary, handling nested objects."""
    if dataclasses.is_dataclass(obj):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            if dataclasses.is_dataclass(value):
                result[field.name] = safe_dataclass_to_dict(value)
            elif isinstance(value, list):
                result[field.name] = [safe_dataclass_to_dict(item) if dataclasses.is_dataclass(item) else item for item in value]
            elif hasattr(value, '__dict__'):  # Handle other objects with __dict__
                result[field.name] = value.__dict__
            else:
                result[field.name] = value
        return result
    elif isinstance(obj, dict):
        return {k: safe_dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_dataclass_to_dict(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # Handle objects with __dict__
        return obj.__dict__
    else:
        return obj

def save_execution_to_history(episode_id: str, task: str, results: Dict):
    """Save execution results to history folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_folder = f"history/{timestamp}_{episode_id[:8]}"
    
    os.makedirs(history_folder, exist_ok=True)
    
    # Save summary
    summary = {
        'episode_id': episode_id,
        'timestamp': timestamp,
        'task': task,
        'success': results.get('success', False),
        'score': results.get('final_score', 0.0),
        'duration': results.get('duration', 0.0),
        'steps_executed': len(results.get('executions', [])),
        'bugs_found': len(results.get('bugs_detected', [])),
        'folder': os.path.basename(history_folder)
    }
    
    with open(os.path.join(history_folder, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save full results
    with open(os.path.join(history_folder, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save logs
    with open(os.path.join(history_folder, "logs.json"), 'w') as f:
        json.dump(app_state.current_logs, f, indent=2)
    
    # Copy screenshots if they exist
    import shutil
    screenshots_src = "output/screenshots"
    if os.path.exists(screenshots_src):
        screenshots_dst = os.path.join(history_folder, "screenshots")
        if os.path.exists(screenshots_dst):
            shutil.rmtree(screenshots_dst)
        shutil.copytree(screenshots_src, screenshots_dst)
    
    return history_folder

@app.route('/')
def index():
    """Main dashboard page."""
    history = load_execution_history()
    return render_template('index.html', history=history[:10])  # Show last 10 executions

@app.route('/setup')
def setup():
    """API key setup page."""
    # Get current API keys to display in the form
    current_keys = {
        'openai': app_state.api_keys.get('openai', ''),
        'anthropic': app_state.api_keys.get('anthropic', ''), 
        'google': app_state.api_keys.get('google', '')
    }
    return render_template('setup.html', api_keys=current_keys)

@app.route('/api/save-keys', methods=['POST'])
def save_api_keys():
    """Save API keys from user input."""
    data = request.get_json()
    
    # Save to session and app state
    app_state.api_keys = {
        'openai': data.get('openai_key', ''),
        'anthropic': data.get('anthropic_key', ''),
        'google': data.get('google_key', '')
    }
    
    # Set environment variables for the current process
    if app_state.api_keys['openai']:
        os.environ['OPENAI_API_KEY'] = app_state.api_keys['openai']
        print(f"✅ OpenAI API key set: {app_state.api_keys['openai'][:10]}...")
    if app_state.api_keys['anthropic']:
        os.environ['ANTHROPIC_API_KEY'] = app_state.api_keys['anthropic']
        print(f"✅ Anthropic API key set: {app_state.api_keys['anthropic'][:10]}...")
    if app_state.api_keys['google']:
        os.environ['GOOGLE_API_KEY'] = app_state.api_keys['google']
        print(f"✅ Google API key set: {app_state.api_keys['google'][:10]}...")
    
    # Save to config file for persistence
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, "api_keys.json")
    with open(config_file, 'w') as f:
        json.dump(app_state.api_keys, f, indent=2)
    
    session['api_keys_configured'] = True
    print("✅ API keys saved to config/api_keys.json")
    
    return jsonify({'success': True})

@app.route('/api/test-keys', methods=['POST'])
def test_api_keys():
    """Test if API keys are valid."""
    data = request.get_json()
    results = {}
    
    # Test OpenAI (supports both OpenAI and GitHub's OpenAI API)
    if data.get('openai_key'):
        try:
            import openai
            
            # Check if it's a GitHub token
            is_github_token = data['openai_key'].startswith('github_pat_') or data['openai_key'].startswith('ghp_')
            
            if is_github_token:
                # Use GitHub's OpenAI API endpoint
                client = openai.OpenAI(
                    api_key=data['openai_key'],
                    base_url="https://models.inference.ai.azure.com"
                )
                # Try with a GitHub-supported model
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # GitHub supports this model
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10
                )
                
                if response and response.choices:
                    results['openai'] = 'valid (GitHub OpenAI)'
                else:
                    results['openai'] = 'invalid: No response from GitHub API'
            else:
                # Use regular OpenAI API
                client = openai.OpenAI(api_key=data['openai_key'])
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                
                if response and response.choices:
                    results['openai'] = 'valid (OpenAI)'
                else:
                    results['openai'] = 'invalid: No response from OpenAI API'
                    
        except Exception as e:
            error_msg = str(e)
            if "incorrect api key" in error_msg.lower() or "invalid" in error_msg.lower():
                results['openai'] = 'invalid: Invalid API key'
            elif "quota" in error_msg.lower():
                results['openai'] = 'invalid: Quota exceeded'
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                results['openai'] = 'invalid: Model not available for this key type'
            else:
                results['openai'] = f'invalid: {error_msg}'
    
    # Test Anthropic
    if data.get('anthropic_key'):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=data['anthropic_key'])
            
            # Test with a simple call
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            if response and response.content:
                results['anthropic'] = 'valid'
            else:
                results['anthropic'] = 'invalid: No response from API'
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
                results['anthropic'] = 'invalid: Invalid API key'
            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                results['anthropic'] = 'invalid: Rate limit or quota exceeded'
            else:
                results['anthropic'] = f'invalid: {error_msg}'
    
    # Test Google (try both Gemini Pro and Flash)
    if data.get('google_key'):
        try:
            import google.generativeai as genai
            
            # Configure with the provided key
            genai.configure(api_key=data['google_key'])
            
            # Try Gemini Flash first (free tier)
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content("Hello", request_options={"timeout": 10})
                
                if response and response.text:
                    results['google'] = 'valid (Gemini 1.5 Flash)'
                    print(f"✅ Gemini Flash API test successful: {response.text[:50]}...")
                else:
                    # Fallback to Gemini Pro
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content("Hello", request_options={"timeout": 10})
                    
                    if response and response.text:
                        results['google'] = 'valid (Gemini Pro)'
                        print(f"✅ Gemini Pro API test successful: {response.text[:50]}...")
                    else:
                        results['google'] = 'invalid: No response from API'
                        print("❌ Gemini API test failed: No response")
            except Exception as flash_error:
                print(f"⚠️ Gemini Flash failed, trying Pro: {flash_error}")
                # Fallback to Gemini Pro
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content("Hello", request_options={"timeout": 10})
                
                if response and response.text:
                    results['google'] = 'valid (Gemini Pro - Flash unavailable)'
                    print(f"✅ Gemini Pro API test successful: {response.text[:50]}...")
                else:
                    results['google'] = 'invalid: No response from API'
                    print("❌ Gemini API test failed: No response")
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Gemini API test failed: {error_msg}")
            
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                results['google'] = 'invalid: Invalid API key'
            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                results['google'] = 'invalid: Rate limit or quota exceeded'
            elif "permission" in error_msg.lower():
                results['google'] = 'invalid: Permission denied - check API access'
            elif "not found" in error_msg.lower():
                results['google'] = 'invalid: API not enabled or model not found'
            else:
                results['google'] = f'invalid: {error_msg}'
    
    print(f"🔍 API test results: {results}")
    return jsonify(results)

@app.route('/execute')
def execute_page():
    """Test execution page."""
    if not session.get('api_keys_configured') and not any(app_state.api_keys.values()):
        return redirect(url_for('setup'))
    return render_template('execute.html')

@app.route('/api/start-test', methods=['POST'])
def start_test():
    """Start a QA test execution."""
    if app_state.test_running:
        return jsonify({'error': 'Test already running'}), 400
    
    data = request.get_json()
    task = data.get('task', 'Test default functionality')
    android_task = data.get('android_task', 'settings_wifi')
    use_real_env = data.get('use_real_env', False)
    
    # Check if API keys are available
    if not any(app_state.api_keys.values()):
        load_api_keys()  # Try to load from file
        
    # Log which API keys are available
    available_keys = [k for k, v in app_state.api_keys.items() if v]
    print(f"🔑 Available API keys: {available_keys}")
    print(f"🔑 Environment variables set: {[k for k in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'] if os.environ.get(k)]}")
    
    # Reset state
    app_state.test_running = True
    app_state.current_logs = []
    app_state.current_episode_id = str(uuid.uuid4())
    
    for agent in app_state.agent_status:
        app_state.agent_status[agent] = {'status': 'ready', 'last_action': None, 'progress': 0}
    
    # Start test in background thread
    def run_test():
        try:
            # Create orchestrator with real or mock engine based on API keys
            use_mock = not any(app_state.api_keys.values())
            if use_mock:
                print("🤖 Using mock LLM responses (no API keys available)")
            else:
                print("🚀 Using real LLM engines with available API keys")
                
            # Create status update callback
            def status_update_callback(agent_name: str, status: str, progress: int = 0, message: str = "", decision: str = ""):
                """Callback to update agent status in real-time."""
                if agent_name in app_state.agent_status:
                    app_state.agent_status[agent_name]['status'] = status
                    app_state.agent_status[agent_name]['progress'] = progress
                    app_state.agent_status[agent_name]['last_action'] = message[:100] if message else ""
                    app_state.agent_status[agent_name]['decision'] = decision[:150] if decision else ""
                    
                    # Emit to frontend
                    socketio.emit('agent_status_update', {
                        'agent': agent_name,
                        'status': app_state.agent_status[agent_name],
                        'timestamp': time.time()
                    })
                    
                    # Emit detailed agent activity for dialog box
                    socketio.emit('agent_activity', {
                        'agent': agent_name,
                        'status': status,
                        'progress': progress,
                        'message': message,
                        'decision': decision,
                        'timestamp': time.time()
                    })
            
            config = QASystemConfig(
                engine_params={'mock': use_mock},
                android_env=None,
                max_execution_time=300.0,
                enable_visual_trace=data.get('enable_screenshots', True),
                enable_recovery=data.get('enable_recovery', True),
                verification_strictness='balanced'
            )
            
            orchestrator = MultiAgentQAOrchestrator(config, status_callback=status_update_callback)
            app_state.current_orchestrator = orchestrator
            
            # Add logging handler to capture logs
            log_handler = WebLogHandler()
            logging.getLogger().addHandler(log_handler)
            
            # Run test
            results = orchestrator.run_qa_test(task)
            
            # Get the episode_id from results
            episode_id = results.episode_id if hasattr(results, 'episode_id') else app_state.current_episode_id
            
            # Save to history
            results_dict = safe_dataclass_to_dict({
                'episode_id': episode_id,
                'success': results.overall_success if hasattr(results, 'overall_success') else False,
                'final_score': results.final_score if hasattr(results, 'final_score') else 0.0,
                'duration': (results.end_time - results.start_time) if hasattr(results, 'end_time') and hasattr(results, 'start_time') else 0.0,
                'executions': getattr(results, 'execution_results', []),
                'verifications': getattr(results, 'verification_results', []),
                'bugs_detected': getattr(results, 'bugs_detected', []),
                'plan': getattr(results, 'plan', {}),
                'supervisor_analysis': getattr(results, 'supervisor_analysis', {})
            })
            
            history_folder = save_execution_to_history(
                episode_id, task, results_dict
            )
            
            # Update state
            app_state.test_running = False
            
            # Notify frontend
            socketio.emit('test_completed', {
                'success': True,
                'results': results_dict,  # Use the serialized version
                'history_folder': history_folder
            })
            
        except Exception as e:
            app_state.test_running = False
            logging.error(f"Test execution failed: {e}")
            socketio.emit('test_completed', {
                'success': False,
                'error': str(e)
            })
    
    thread = threading.Thread(target=run_test)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'episode_id': app_state.current_episode_id})

@app.route('/api/stop-test', methods=['POST'])
def stop_test():
    """Stop current test execution."""
    app_state.test_running = False
    
    # Reset agent status
    for agent in app_state.agent_status:
        app_state.agent_status[agent]['status'] = 'stopped'
    
    socketio.emit('test_stopped')
    return jsonify({'success': True})

@app.route('/api/agent-status')
def get_agent_status():
    """Get current agent status."""
    return jsonify(app_state.agent_status)

@app.route('/api/logs')
def get_logs():
    """Get current execution logs."""
    return jsonify(app_state.current_logs)

@app.route('/history')
def history_page():
    """Execution history page."""
    history = load_execution_history()
    return render_template('history.html', history=history)

@app.route('/history/<folder>/screenshots/<filename>')
def serve_screenshot(folder, filename):
    """Serve screenshot files from history folders."""
    import os
    from flask import send_from_directory
    
    screenshot_path = os.path.join("history", folder, "screenshots")
    if os.path.exists(os.path.join(screenshot_path, filename)):
        return send_from_directory(screenshot_path, filename)
    else:
        return "Screenshot not found", 404

@app.route('/api/history/<folder>/details')
def get_history_details(folder):
    """Get detailed history for a specific execution."""
    try:
        history_path = os.path.join("history", folder)
        
        if not os.path.exists(history_path):
            return jsonify({'error': 'History not found'}), 404
        
        details = {}
        
        # Load summary
        summary_file = os.path.join(history_path, "summary.json")
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    details['summary'] = json.load(f)
            except Exception as e:
                details['summary'] = {'error': f'Failed to load summary: {str(e)}'}
        
        # Load results
        results_file = os.path.join(history_path, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    # Convert any remaining non-serializable objects
                    details['results'] = safe_dataclass_to_dict(results_data)
            except Exception as e:
                details['results'] = {'error': f'Failed to load results: {str(e)}'}
        
        # Load logs
        logs_file = os.path.join(history_path, "logs.json")
        if os.path.exists(logs_file):
            try:
                with open(logs_file, 'r') as f:
                    details['logs'] = json.load(f)
            except Exception as e:
                details['logs'] = {'error': f'Failed to load logs: {str(e)}'}
        
        # List screenshots
        screenshots_dir = os.path.join(history_path, "screenshots")
        if os.path.exists(screenshots_dir):
            try:
                details['screenshots'] = [
                    f for f in os.listdir(screenshots_dir) 
                    if f.endswith('.png') or f.endswith('.jpg')
                ]
            except Exception as e:
                details['screenshots'] = []
        
        return jsonify(details)
        
    except Exception as e:
        logger.error(f"Error in get_history_details: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
    
    return jsonify(details)

@app.route('/api/history/<folder>/delete', methods=['DELETE'])
def delete_history(folder):
    """Delete a specific history execution."""
    import shutil
    
    history_path = os.path.join("history", folder)
    
    if not os.path.exists(history_path):
        return jsonify({'error': 'History not found'}), 404
    
    try:
        # Remove the entire history folder
        shutil.rmtree(history_path)
        return jsonify({'success': True, 'message': f'History {folder} deleted successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to delete history: {str(e)}'}), 500

@app.route('/api/history/clear-all', methods=['DELETE'])
def clear_all_history():
    """Clear all execution history."""
    import shutil
    
    history_dir = "history"
    
    if not os.path.exists(history_dir):
        return jsonify({'success': True, 'message': 'No history to clear'})
    
    try:
        # Remove all history folders but keep the history directory
        for item in os.listdir(history_dir):
            item_path = os.path.join(history_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        return jsonify({'success': True, 'message': 'All history cleared successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to clear history: {str(e)}'}), 500

class WebLogHandler(logging.Handler):
    """Custom log handler that sends logs to the web interface."""
    
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'agent': self.extract_agent_name(record.name)
        }
        
        app_state.current_logs.append(log_entry)
        
        # Update agent status based on log
        self.update_agent_status(log_entry)
        
        # Emit to frontend
        socketio.emit('log_update', log_entry)
    
    def extract_agent_name(self, logger_name: str) -> str:
        """Extract agent name from logger name."""
        if 'planner' in logger_name:
            return 'planner'
        elif 'executor' in logger_name:
            return 'executor'
        elif 'verifier' in logger_name:
            return 'verifier'
        elif 'supervisor' in logger_name:
            return 'supervisor'
        return 'system'
    
    def update_agent_status(self, log_entry: Dict):
        """Update agent status based on log entry."""
        agent = log_entry['agent']
        message = log_entry['message'].lower()
        
        if agent in app_state.agent_status:
            if 'creating plan' in message or 'planning' in message:
                app_state.agent_status[agent]['status'] = 'planning'
                app_state.agent_status[agent]['progress'] = 10
            elif 'executing' in message:
                app_state.agent_status[agent]['status'] = 'executing'
                app_state.agent_status[agent]['progress'] = 50
            elif 'verifying' in message:
                app_state.agent_status[agent]['status'] = 'verifying'
                app_state.agent_status[agent]['progress'] = 75
            elif 'analyzing' in message or 'analysis' in message:
                app_state.agent_status[agent]['status'] = 'analyzing'
                app_state.agent_status[agent]['progress'] = 80
            elif 'completed' in message or 'success' in message:
                app_state.agent_status[agent]['status'] = 'completed'
                app_state.agent_status[agent]['progress'] = 100
            elif 'error' in message or 'failed' in message:
                app_state.agent_status[agent]['status'] = 'error'
            
            app_state.agent_status[agent]['last_action'] = log_entry['message'][:100]
            
            # Emit status update
            socketio.emit('agent_status_update', {
                'agent': agent,
                'status': app_state.agent_status[agent]
            })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'status': 'Connected to QA System'})

@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client."""
    emit('agent_status_update', app_state.agent_status)

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('history', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load existing history
    app_state.execution_history = load_execution_history()
    
    print("🚀 Starting Multi-Agent QA System Web Interface...")
    print("📱 Access the dashboard at: http://localhost:5000")
    print("🔧 Setup API keys at: http://localhost:5000/setup")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
