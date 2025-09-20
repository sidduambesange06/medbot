"""
AUTO-GREETING CHAT SYSTEM v3.0
Automatically starts conversation without user input
Real-time integration with ultra-fast indexing system
"""

import asyncio
import json
import time
import threading
import websockets
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AutoGreetingSystem:
    """Automatically greets users and handles initial conversation flow"""
    
    def __init__(self):
        self.greeting_templates = [
            {
                "greeting": "👋 Welcome to MedAI Pro! I'm your advanced medical AI assistant.",
                "follow_up": "I can help with medical questions, drug interactions, diagnoses, and treatment options. What would you like to know?",
                "suggestions": [
                    "🩺 Ask about symptoms or conditions",
                    "💊 Check drug interactions", 
                    "🔬 Research medical procedures",
                    "📚 Get treatment guidelines"
                ]
            },
            {
                "greeting": "🔬 Hello! I'm your medical knowledge companion powered by extensive medical literature.",
                "follow_up": "I have access to comprehensive medical databases and can provide evidence-based information. How can I assist you today?",
                "suggestions": [
                    "🏥 Emergency medicine guidance",
                    "🧬 Genetic disorders information",
                    "⚕️ Clinical practice guidelines", 
                    "📖 Medical research insights"
                ]
            },
            {
                "greeting": "⚡ Welcome! I'm MedAI Pro - your ultra-fast medical information system.",
                "follow_up": "I can instantly search through thousands of medical documents to provide accurate, evidence-based answers. What's your medical question?",
                "suggestions": [
                    "🎯 Specific medical queries",
                    "📊 Compare treatment options",
                    "🔍 Symptom analysis",
                    "💡 Medical explanations"
                ]
            }
        ]
        
        self.current_template_index = 0
        self.session_data = {}
        
    def get_auto_greeting(self, session_id: str = None) -> Dict:
        """Get automatic greeting based on time and context"""
        
        # Rotate greetings to avoid repetition
        template = self.greeting_templates[self.current_template_index]
        self.current_template_index = (self.current_template_index + 1) % len(self.greeting_templates)
        
        # Add time-based context
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            time_context = "Good morning! ☀️"
        elif 12 <= current_hour < 17:
            time_context = "Good afternoon! ⛅"
        elif 17 <= current_hour < 21:
            time_context = "Good evening! 🌇"
        else:
            time_context = "Good night! 🌙"
        
        # Combine greeting with time context
        full_greeting = f"{time_context} {template['greeting']}"
        
        return {
            "type": "auto_greeting",
            "greeting": full_greeting,
            "follow_up": template['follow_up'],
            "suggestions": template['suggestions'],
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or f"session_{int(time.time())}"
        }
    
    def get_context_aware_greeting(self, indexing_status: Dict) -> Dict:
        """Get greeting that acknowledges current system status"""
        
        base_greeting = self.get_auto_greeting()
        
        # Add system status context
        if indexing_status.get('is_indexing', False):
            current_book = indexing_status.get('current_book', '')
            progress = indexing_status.get('progress_percent', 0)
            
            status_message = f"📚 Currently indexing medical books ({progress:.1f}% complete). "
            if current_book:
                status_message += f"Processing: {current_book}"
            
            base_greeting['system_status'] = status_message
            base_greeting['additional_info'] = "The knowledge base is being updated in real-time for even better responses!"
        
        elif indexing_status.get('chunks_uploaded', 0) > 0:
            total_vectors = indexing_status.get('chunks_uploaded', 0)
            base_greeting['system_status'] = f"✅ Ready with {total_vectors:,} medical knowledge chunks indexed!"
            base_greeting['additional_info'] = "Ask me anything - I have access to the latest medical information!"
        
        return base_greeting

class RealTimeStatusMonitor:
    """Monitor indexing status for real-time updates"""
    
    def __init__(self):
        self.status_file = Path("admin_status.json")
        self.last_modified = 0
        self.cached_status = {}
        
    def get_current_status(self) -> Dict:
        """Get current indexing status with caching"""
        
        if not self.status_file.exists():
            return {
                'is_indexing': False,
                'chunks_uploaded': 0,
                'system_ready': True
            }
        
        try:
            # Check if file was modified
            current_modified = self.status_file.stat().st_mtime
            if current_modified > self.last_modified:
                with open(self.status_file, 'r') as f:
                    self.cached_status = json.load(f)
                self.last_modified = current_modified
            
            return self.cached_status
            
        except Exception as e:
            logger.warning(f"Error reading status file: {e}")
            return {'is_indexing': False, 'system_ready': True}

class EnhancedChatInterface:
    """Enhanced chat interface with auto-greeting and real-time updates"""
    
    def __init__(self):
        self.greeting_system = AutoGreetingSystem()
        self.status_monitor = RealTimeStatusMonitor()
        self.active_sessions = {}
        
    def initialize_session(self, session_id: str) -> Dict:
        """Initialize new chat session with auto-greeting"""
        
        # Get current system status
        status = self.status_monitor.get_current_status()
        
        # Generate context-aware greeting
        greeting_data = self.greeting_system.get_context_aware_greeting(status)
        greeting_data['session_id'] = session_id
        
        # Store session data
        self.active_sessions[session_id] = {
            'start_time': datetime.now().isoformat(),
            'greeting_sent': True,
            'message_count': 0,
            'last_activity': datetime.now().isoformat()
        }
        
        return greeting_data
    
    def get_chat_response(self, session_id: str, user_message: str) -> Dict:
        """Get chat response with session tracking"""
        
        if session_id not in self.active_sessions:
            # Auto-initialize session if not exists
            self.initialize_session(session_id)
        
        # Update session activity
        self.active_sessions[session_id]['last_activity'] = datetime.now().isoformat()
        self.active_sessions[session_id]['message_count'] += 1
        
        # Here you would integrate with your main chat system
        # For now, return a placeholder response
        return {
            "type": "chat_response",
            "message": f"Processing your query: {user_message}",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_welcome_page_data(self) -> Dict:
        """Get data for welcome page with auto-greeting"""
        
        # Generate a new session ID
        session_id = f"session_{int(time.time())}_{hash(str(time.time())) % 10000}"
        
        # Get auto-greeting
        greeting_data = self.initialize_session(session_id)
        
        # Add quick start suggestions
        greeting_data['quick_start'] = {
            "medical_emergencies": [
                "What are the signs of a heart attack?",
                "How to treat severe allergic reactions?",
                "Stroke symptoms and immediate care"
            ],
            "common_conditions": [
                "Diabetes management guidelines",
                "Hypertension treatment options", 
                "Migraine prevention strategies"
            ],
            "drug_information": [
                "Common drug interactions to avoid",
                "Antibiotic resistance concerns",
                "Pain medication safety"
            ]
        }
        
        return greeting_data

# Flask route integration
def integrate_auto_greeting_routes(app):
    """Integrate auto-greeting routes with Flask app"""
    
    chat_interface = EnhancedChatInterface()
    
    @app.route('/api/auto-greeting')
    def get_auto_greeting():
        """API endpoint for auto-greeting"""
        try:
            session_id = request.args.get('session_id')
            if not session_id:
                session_id = f"session_{int(time.time())}"
            
            greeting_data = chat_interface.initialize_session(session_id)
            return jsonify(greeting_data)
            
        except Exception as e:
            logger.error(f"Auto-greeting error: {e}")
            return jsonify({
                "error": "Failed to generate greeting",
                "type": "error"
            }), 500
    
    @app.route('/api/welcome-data')
    def get_welcome_data():
        """API endpoint for welcome page data"""
        try:
            welcome_data = chat_interface.get_welcome_page_data()
            return jsonify(welcome_data)
            
        except Exception as e:
            logger.error(f"Welcome data error: {e}")
            return jsonify({
                "error": "Failed to load welcome data", 
                "type": "error"
            }), 500
    
    @app.route('/api/system-status')
    def get_system_status():
        """API endpoint for real-time system status"""
        try:
            status = chat_interface.status_monitor.get_current_status()
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Status error: {e}")
            return jsonify({
                "error": "Failed to get system status",
                "type": "error"
            }), 500

# JavaScript for auto-greeting integration
AUTO_GREETING_JS = """
class AutoGreetingChat {
    constructor() {
        this.sessionId = null;
        this.isGreetingSent = false;
        this.statusUpdateInterval = null;
        this.init();
    }
    
    async init() {
        try {
            // Auto-load greeting when page loads
            await this.loadAutoGreeting();
            
            // Start real-time status monitoring
            this.startStatusMonitoring();
            
            // Setup message input handlers
            this.setupMessageHandlers();
            
        } catch (error) {
            console.error('Auto-greeting initialization failed:', error);
        }
    }
    
    async loadAutoGreeting() {
        try {
            const response = await fetch('/api/welcome-data');
            const data = await response.json();
            
            if (data.type === 'auto_greeting') {
                this.sessionId = data.session_id;
                this.displayAutoGreeting(data);
                this.isGreetingSent = true;
            }
        } catch (error) {
            console.error('Failed to load auto-greeting:', error);
        }
    }
    
    displayAutoGreeting(data) {
        const chatContainer = document.getElementById('chat-messages');
        if (!chatContainer) return;
        
        // Create greeting message
        const greetingDiv = document.createElement('div');
        greetingDiv.className = 'message ai-message auto-greeting';
        
        greetingDiv.innerHTML = `
            <div class="message-content">
                <div class="greeting-text">${data.greeting}</div>
                <div class="follow-up-text">${data.follow_up}</div>
                
                ${data.system_status ? `
                    <div class="system-status">
                        <i class="fas fa-info-circle"></i>
                        ${data.system_status}
                    </div>
                ` : ''}
                
                <div class="suggestions">
                    <h4>Quick Start Suggestions:</h4>
                    <div class="suggestion-buttons">
                        ${data.suggestions.map(suggestion => `
                            <button class="suggestion-btn" onclick="autoGreeting.sendSuggestion('${suggestion}')">
                                ${suggestion}
                            </button>
                        `).join('')}
                    </div>
                </div>
                
                ${data.quick_start ? `
                    <div class="quick-start-categories">
                        <div class="category">
                            <h5>🚨 Medical Emergencies</h5>
                            ${data.quick_start.medical_emergencies.map(item => `
                                <button class="quick-start-btn" onclick="autoGreeting.sendMessage('${item}')">
                                    ${item}
                                </button>
                            `).join('')}
                        </div>
                        
                        <div class="category">
                            <h5>🏥 Common Conditions</h5>
                            ${data.quick_start.common_conditions.map(item => `
                                <button class="quick-start-btn" onclick="autoGreeting.sendMessage('${item}')">
                                    ${item}
                                </button>
                            `).join('')}
                        </div>
                        
                        <div class="category">
                            <h5>💊 Drug Information</h5>
                            ${data.quick_start.drug_information.map(item => `
                                <button class="quick-start-btn" onclick="autoGreeting.sendMessage('${item}')">
                                    ${item}
                                </button>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
            <div class="message-time">${new Date().toLocaleTimeString()}</div>
        `;
        
        chatContainer.appendChild(greetingDiv);
        
        // Auto-scroll to greeting
        setTimeout(() => {
            greetingDiv.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }
    
    startStatusMonitoring() {
        this.statusUpdateInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/system-status');
                const status = await response.json();
                this.updateSystemStatus(status);
            } catch (error) {
                console.warn('Status update failed:', error);
            }
        }, 2000); // Update every 2 seconds
    }
    
    updateSystemStatus(status) {
        const statusElement = document.querySelector('.system-status');
        if (!statusElement) return;
        
        if (status.is_indexing) {
            const progress = status.progress_percent || 0;
            const currentBook = status.current_book || 'Processing...';
            
            statusElement.innerHTML = `
                <i class="fas fa-spinner fa-spin"></i>
                📚 Indexing: ${currentBook} (${progress.toFixed(1)}% complete)
            `;
            statusElement.className = 'system-status indexing';
        } else {
            statusElement.innerHTML = `
                <i class="fas fa-check-circle"></i>
                ✅ System ready with ${status.chunks_uploaded || 0} knowledge chunks!
            `;
            statusElement.className = 'system-status ready';
        }
    }
    
    setupMessageHandlers() {
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        
        if (messageInput) {
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage(messageInput.value);
                }
            });
        }
        
        if (sendButton) {
            sendButton.addEventListener('click', () => {
                const message = messageInput.value;
                if (message.trim()) {
                    this.sendMessage(message);
                }
            });
        }
    }
    
    sendSuggestion(suggestion) {
        // Remove emoji prefix for cleaner query
        const cleanSuggestion = suggestion.replace(/^[\\u{1f300}-\\u{1f9ff}]\\s*/, '');
        this.sendMessage(cleanSuggestion);
    }
    
    sendMessage(message) {
        const messageInput = document.getElementById('message-input');
        if (messageInput) {
            messageInput.value = message;
        }
        
        // Trigger your existing chat send function
        if (typeof sendMessage === 'function') {
            sendMessage();
        } else if (typeof submitMessage === 'function') {
            submitMessage();
        } else {
            console.warn('No send message function found');
        }
    }
    
    destroy() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
        }
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.autoGreeting = new AutoGreetingChat();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.autoGreeting) {
        window.autoGreeting.destroy();
    }
});
"""

# CSS for auto-greeting styling
AUTO_GREETING_CSS = """
.auto-greeting {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    margin: 20px 0;
    animation: slideInUp 0.6s ease-out;
}

.auto-greeting .message-content {
    padding: 25px;
}

.greeting-text {
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 15px;
    color: white;
}

.follow-up-text {
    font-size: 1em;
    margin-bottom: 20px;
    color: rgba(255, 255, 255, 0.9);
    line-height: 1.5;
}

.system-status {
    padding: 12px 16px;
    border-radius: 8px;
    margin: 15px 0;
    font-size: 0.9em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.system-status.indexing {
    background: rgba(255, 193, 7, 0.2);
    border: 1px solid rgba(255, 193, 7, 0.3);
    color: #ffc107;
}

.system-status.ready {
    background: rgba(40, 167, 69, 0.2);
    border: 1px solid rgba(40, 167, 69, 0.3);
    color: #28a745;
}

.suggestions h4 {
    color: white;
    margin-bottom: 15px;
    font-size: 1.1em;
}

.suggestion-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 20px;
}

.suggestion-btn {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 0.9em;
}

.suggestion-btn:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

.quick-start-categories {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.category h5 {
    color: white;
    margin-bottom: 10px;
    font-size: 1em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.quick-start-btn {
    display: block;
    width: 100%;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 8px;
    transition: all 0.3s ease;
    text-align: left;
    font-size: 0.85em;
}

.quick-start-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateX(5px);
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 768px) {
    .quick-start-categories {
        grid-template-columns: 1fr;
    }
    
    .suggestion-buttons {
        flex-direction: column;
    }
    
    .suggestion-btn {
        width: 100%;
    }
}
"""

if __name__ == "__main__":
    # Test the auto-greeting system
    greeting_system = AutoGreetingSystem()
    test_greeting = greeting_system.get_auto_greeting("test_session")
    print(json.dumps(test_greeting, indent=2))