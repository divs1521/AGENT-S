"""
Web Interface Extension for Bonus Features
Adds bonus feature management to the existing web interface
without disrupting core functionality.
"""

import os
import json
import logging
from flask import Blueprint, render_template, request, jsonify, current_app
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Create blueprint for bonus features
bonus_bp = Blueprint('bonus', __name__, url_prefix='/bonus')

@bonus_bp.route('/')
def bonus_dashboard():
    """Bonus features dashboard page."""
    return render_template('bonus/dashboard.html')

@bonus_bp.route('/android-in-the-wild')
def android_in_the_wild_page():
    """Android in the wild integration page."""
    return render_template('bonus/android_in_the_wild.html')

@bonus_bp.route('/enhanced-training')
def enhanced_training_page():
    """Enhanced training page."""
    return render_template('bonus/enhanced_training.html')

@bonus_bp.route('/api/status')
def get_bonus_status():
    """Get status of bonus features."""
    try:
        # Check if bonus manager is available
        bonus_manager = getattr(current_app, 'bonus_manager', None)
        
        if not bonus_manager:
            return jsonify({
                'available': False,
                'message': 'Bonus features not initialized'
            })
        
        status = {
            'available': True,
            'features_enabled': bonus_manager.features_enabled,
            'processing_status': {
                'videos_processed': len(bonus_manager.bonus_results.get('processed_videos', [])),
                'reproductions_completed': len(bonus_manager.bonus_results.get('reproduction_results', [])),
                'training_completed': bool(bonus_manager.bonus_results.get('training_results'))
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting bonus status: {e}")
        return jsonify({'error': str(e)}), 500

@bonus_bp.route('/api/start-integration', methods=['POST'])
def start_android_integration():
    """Start android_in_the_wild integration."""
    try:
        data = request.get_json()
        video_count = data.get('video_count', 5)
        
        bonus_manager = getattr(current_app, 'bonus_manager', None)
        if not bonus_manager:
            return jsonify({'error': 'Bonus features not available'}), 400
        
        # Start integration in background
        def run_integration():
            try:
                results = bonus_manager.run_android_in_the_wild_integration(video_count)
                # Emit results via websocket if available
                if hasattr(current_app, 'socketio'):
                    current_app.socketio.emit('bonus_integration_complete', results)
            except Exception as e:
                logger.error(f"Integration failed: {e}")
                if hasattr(current_app, 'socketio'):
                    current_app.socketio.emit('bonus_integration_error', {'error': str(e)})
        
        import threading
        thread = threading.Thread(target=run_integration)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Integration started'})
        
    except Exception as e:
        logger.error(f"Error starting integration: {e}")
        return jsonify({'error': str(e)}), 500

@bonus_bp.route('/api/start-training', methods=['POST'])
def start_enhanced_training():
    """Start enhanced training pipeline."""
    try:
        bonus_manager = getattr(current_app, 'bonus_manager', None)
        if not bonus_manager:
            return jsonify({'error': 'Bonus features not available'}), 400
        
        # Start training in background
        def run_training():
            try:
                results = bonus_manager.run_enhanced_training_pipeline()
                # Emit results via websocket if available
                if hasattr(current_app, 'socketio'):
                    current_app.socketio.emit('bonus_training_complete', results)
            except Exception as e:
                logger.error(f"Training failed: {e}")
                if hasattr(current_app, 'socketio'):
                    current_app.socketio.emit('bonus_training_error', {'error': str(e)})
        
        import threading
        thread = threading.Thread(target=run_training)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Training started'})
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({'error': str(e)}), 500

@bonus_bp.route('/api/results')
def get_bonus_results():
    """Get all bonus feature results."""
    try:
        bonus_manager = getattr(current_app, 'bonus_manager', None)
        if not bonus_manager:
            return jsonify({'error': 'Bonus features not available'}), 400
        
        # Generate comprehensive report
        report = bonus_manager.generate_bonus_report()
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Error getting bonus results: {e}")
        return jsonify({'error': str(e)}), 500

@bonus_bp.route('/api/download-results')
def download_bonus_results():
    """Download bonus results as JSON file."""
    try:
        bonus_manager = getattr(current_app, 'bonus_manager', None)
        if not bonus_manager:
            return jsonify({'error': 'Bonus features not available'}), 400
        
        from flask import Response
        import json
        
        # Get comprehensive results
        results = bonus_manager.generate_bonus_report()
        
        # Create JSON response
        json_data = json.dumps(results, indent=2)
        
        return Response(
            json_data,
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment; filename=bonus_features_results.json'}
        )
        
    except Exception as e:
        logger.error(f"Error downloading results: {e}")
        return jsonify({'error': str(e)}), 500

def register_bonus_features(app, bonus_manager=None):
    """Register bonus features with the Flask app."""
    try:
        # Register blueprint
        app.register_blueprint(bonus_bp)
        
        # Store bonus manager in app context
        if bonus_manager:
            app.bonus_manager = bonus_manager
            logger.info("✅ Bonus features registered with web interface")
        else:
            logger.info("⚠️ Bonus features registered but manager not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to register bonus features: {e}")
        return False
