"""
ENHANCED APP INTEGRATION v3.0
Integrates ultra-fast indexing, auto-greeting, and admin panel with main Flask app
"""

import os
import sys
import time
import asyncio
import threading
import logging
from pathlib import Path
from flask import Flask
from auto_greeting_chat import integrate_auto_greeting_routes, AUTO_GREETING_JS, AUTO_GREETING_CSS
from admin_panel_integration import create_admin_blueprint, ADMIN_DASHBOARD_HTML

logger = logging.getLogger(__name__)

def enhance_main_app(app: Flask):
    """Enhance the main Flask app with all new features"""
    
    # 1. Integrate auto-greeting chat system
    logger.info("🚀 Integrating auto-greeting chat system...")
    integrate_auto_greeting_routes(app)
    
    # 2. Register admin panel blueprint
    logger.info("🚀 Integrating admin panel...")
    admin_bp = create_admin_blueprint()
    app.register_blueprint(admin_bp)
    
    # 3. Add enhanced chat route with auto-greeting (using /enhanced-chat to avoid conflicts)
    @app.route('/enhanced-chat')
    def enhanced_chat():
        """Enhanced chat page with auto-greeting"""
        return render_enhanced_chat_template()
    
    # 4. Add auto-greeting assets
    @app.route('/static/js/auto-greeting.js')
    def auto_greeting_js():
        """Serve auto-greeting JavaScript"""
        from flask import Response
        return Response(AUTO_GREETING_JS, mimetype='application/javascript')
    
    @app.route('/static/css/auto-greeting.css')
    def auto_greeting_css():
        """Serve auto-greeting CSS"""
        from flask import Response
        return Response(AUTO_GREETING_CSS, mimetype='text/css')
    
    # 5. Enhanced admin dashboard template
    @app.route('/admin/templates/dashboard.html')
    def admin_dashboard_template():
        """Serve admin dashboard template"""
        from flask import Response
        return Response(ADMIN_DASHBOARD_HTML, mimetype='text/html')
    
    logger.info("✅ App enhancement completed!")
    return app

def render_enhanced_chat_template():
    """Render enhanced chat template with auto-greeting"""
    template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedAI Pro - Advanced Medical Intelligence</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="/static/css/auto-greeting.css" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --medical-gradient: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --dark-bg: #0a0e1a;
            --card-bg: rgba(255, 255, 255, 0.02);
            --glass-bg: rgba(255, 255, 255, 0.1);
            --text-primary: #ffffff;
            --text-secondary: #a0aec0;
            --border-color: rgba(255, 255, 255, 0.1);
            --shadow-glow: 0 8px 32px rgba(31, 38, 135, 0.37);
            --shadow-intense: 0 20px 60px rgba(0, 0, 0, 0.4);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--dark-bg);
            color: var(--text-primary);
            overflow-x: hidden;
            position: relative;
        }

        /* Animated Background */
        .background-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: var(--dark-bg);
        }

        .background-animation::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(118, 75, 162, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 80%, rgba(0, 198, 255, 0.05) 0%, transparent 50%);
            animation: backgroundFloat 20s ease-in-out infinite;
        }

        @keyframes backgroundFloat {
            0%, 100% { transform: scale(1) rotate(0deg); }
            50% { transform: scale(1.1) rotate(180deg); }
        }

        /* Header */
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-color);
            padding: 15px 30px;
            z-index: 1000;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.5em;
            font-weight: 700;
            background: var(--medical-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-actions {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--card-bg);
            border-radius: 20px;
            border: 1px solid var(--border-color);
            font-size: 0.9em;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success-gradient);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        /* Main Container */
        .main-container {
            margin-top: 80px;
            height: calc(100vh - 80px);
            display: flex;
            flex-direction: column;
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            width: 100%;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            scroll-behavior: smooth;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            animation: messageSlideIn 0.4s ease-out;
        }

        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user-message {
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            font-weight: bold;
            flex-shrink: 0;
        }

        .ai-avatar {
            background: var(--medical-gradient);
            color: white;
        }

        .user-avatar {
            background: var(--primary-gradient);
            color: white;
        }

        .message-content {
            max-width: 70%;
            padding: 16px 20px;
            border-radius: 18px;
            position: relative;
        }

        .ai-message .message-content {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-bottom-left-radius: 6px;
        }

        .user-message .message-content {
            background: var(--primary-gradient);
            border-bottom-right-radius: 6px;
        }

        .message-time {
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        /* Input Area */
        .input-container {
            padding: 20px;
            background: var(--glass-bg);
            border-top: 1px solid var(--border-color);
            backdrop-filter: blur(20px);
        }

        .input-wrapper {
            display: flex;
            gap: 12px;
            align-items: flex-end;
            max-width: 1200px;
            margin: 0 auto;
        }

        .message-input {
            flex: 1;
            min-height: 50px;
            max-height: 120px;
            padding: 12px 20px;
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 25px;
            color: var(--text-primary);
            font-size: 1em;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        .message-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .message-input::placeholder {
            color: var(--text-secondary);
        }

        .send-button {
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            background: var(--medical-gradient);
            color: white;
            font-size: 1.2em;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-glow);
        }

        .send-button:hover {
            transform: scale(1.05);
            box-shadow: var(--shadow-intense);
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Typing Indicator */
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 8px;
            margin: 10px 0;
            color: var(--text-secondary);
        }

        .typing-dots {
            display: inline-block;
        }

        .typing-dots span {
            display: inline-block;
            width: 4px;
            height: 4px;
            background: var(--text-secondary);
            border-radius: 50%;
            margin: 0 1px;
            animation: typingBounce 1.4s infinite;
        }

        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typingBounce {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }

        /* Quick Actions */
        .quick-actions {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .quick-action-btn {
            padding: 8px 16px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9em;
        }

        .quick-action-btn:hover {
            background: var(--primary-gradient);
            transform: translateY(-2px);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .header {
                padding: 12px 20px;
            }

            .logo {
                font-size: 1.3em;
            }

            .chat-container {
                padding: 15px;
            }

            .message-content {
                max-width: 85%;
            }

            .input-wrapper {
                gap: 8px;
            }

            .quick-actions {
                flex-direction: column;
            }

            .quick-action-btn {
                width: 100%;
                text-align: center;
            }
        }

        /* Loading States */
        .loading {
            opacity: 0.7;
            pointer-events: none;
        }

        .shimmer {
            background: linear-gradient(
                90deg,
                rgba(255, 255, 255, 0.1) 25%,
                rgba(255, 255, 255, 0.2) 50%,
                rgba(255, 255, 255, 0.1) 75%
            );
            background-size: 200% 100%;
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% {
                background-position: -200% 0;
            }
            100% {
                background-position: 200% 0;
            }
        }
    </style>
</head>
<body>
    <div class="background-animation"></div>
    
    <header class="header">
        <div class="logo">
            <i class="fas fa-user-md"></i>
            MedAI Pro
        </div>
        <div class="header-actions">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span id="system-status-text">System Ready</span>
            </div>
            <a href="/admin/dashboard" class="quick-action-btn">
                <i class="fas fa-cog"></i> Admin
            </a>
        </div>
    </header>

    <main class="main-container">
        <div class="chat-container">
            <div class="chat-messages" id="chat-messages">
                <!-- Auto-greeting will be inserted here -->
            </div>
            
            <div class="typing-indicator" id="typing-indicator">
                <i class="fas fa-user-md"></i>
                <span>MedAI is thinking</span>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            
            <div class="input-container">
                <div class="quick-actions" id="quick-actions">
                    <!-- Quick action buttons will be populated by auto-greeting -->
                </div>
                
                <div class="input-wrapper">
                    <textarea 
                        id="message-input" 
                        class="message-input" 
                        placeholder="Ask me any medical question..."
                        rows="1"
                    ></textarea>
                    <button id="send-button" class="send-button">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>
    </main>

    <!-- Auto-greeting and enhanced chat functionality -->
    <script src="/static/js/auto-greeting.js"></script>
    
    <script>
        // Enhanced chat functionality
        class EnhancedMedicalChat {
            constructor() {
                this.messageInput = document.getElementById('message-input');
                this.sendButton = document.getElementById('send-button');
                this.chatMessages = document.getElementById('chat-messages');
                this.typingIndicator = document.getElementById('typing-indicator');
                this.quickActions = document.getElementById('quick-actions');
                this.systemStatusText = document.getElementById('system-status-text');
                
                this.isLoading = false;
                this.messageCount = 0;
                
                this.initializeChat();
                this.setupEventListeners();
                this.startSystemStatusMonitoring();
            }
            
            initializeChat() {
                // Auto-resize textarea
                this.messageInput.addEventListener('input', () => {
                    this.messageInput.style.height = 'auto';
                    this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
                });
            }
            
            setupEventListeners() {
                // Send message on Enter (but not Shift+Enter)
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
                
                // Send button click
                this.sendButton.addEventListener('click', () => {
                    this.sendMessage();
                });
            }
            
            async sendMessage(messageText = null) {
                const message = messageText || this.messageInput.value.trim();
                if (!message || this.isLoading) return;
                
                // Clear input
                this.messageInput.value = '';
                this.messageInput.style.height = 'auto';
                
                // Add user message to chat
                this.addMessage(message, 'user');
                
                // Show typing indicator
                this.showTypingIndicator();
                
                try {
                    this.isLoading = true;
                    this.sendButton.disabled = true;
                    
                    // Send to backend
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: window.autoGreeting ? window.autoGreeting.sessionId : null
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        // Add AI response to chat
                        this.addMessage(data.response, 'ai', data.medical_content);
                    } else {
                        this.addMessage('Sorry, I encountered an error. Please try again.', 'ai');
                    }
                    
                } catch (error) {
                    console.error('Chat error:', error);
                    this.addMessage('Sorry, I encountered a network error. Please try again.', 'ai');
                } finally {
                    this.hideTypingIndicator();
                    this.isLoading = false;
                    this.sendButton.disabled = false;
                    this.messageCount++;
                }
            }
            
            addMessage(content, type, medicalContent = null) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                
                const avatarDiv = document.createElement('div');
                avatarDiv.className = `message-avatar ${type}-avatar`;
                avatarDiv.innerHTML = type === 'ai' ? '<i class="fas fa-user-md"></i>' : '<i class="fas fa-user"></i>';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                if (type === 'ai' && medicalContent) {
                    // Enhanced AI message with medical context
                    contentDiv.innerHTML = `
                        <div class="ai-response">${content}</div>
                        ${medicalContent ? `
                            <div class="medical-context">
                                <details>
                                    <summary>📚 Medical Reference</summary>
                                    <div class="medical-content">${medicalContent.substring(0, 500)}...</div>
                                </details>
                            </div>
                        ` : ''}
                    `;
                } else {
                    contentDiv.textContent = content;
                }
                
                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time';
                timeDiv.textContent = new Date().toLocaleTimeString();
                
                messageDiv.appendChild(avatarDiv);
                messageDiv.appendChild(contentDiv);
                contentDiv.appendChild(timeDiv);
                
                this.chatMessages.appendChild(messageDiv);
                
                // Auto-scroll to bottom
                setTimeout(() => {
                    messageDiv.scrollIntoView({ behavior: 'smooth' });
                }, 100);
            }
            
            showTypingIndicator() {
                this.typingIndicator.style.display = 'flex';
                this.typingIndicator.scrollIntoView({ behavior: 'smooth' });
            }
            
            hideTypingIndicator() {
                this.typingIndicator.style.display = 'none';
            }
            
            async startSystemStatusMonitoring() {
                setInterval(async () => {
                    try {
                        const response = await fetch('/api/system-status');
                        const status = await response.json();
                        
                        if (status.is_indexing) {
                            this.systemStatusText.textContent = `Indexing: ${status.progress_percent?.toFixed(1) || 0}%`;
                        } else {
                            this.systemStatusText.textContent = 'System Ready';
                        }
                    } catch (error) {
                        console.warn('Status update failed:', error);
                    }
                }, 3000);
            }
        }
        
        // Initialize enhanced chat when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            window.enhancedChat = new EnhancedMedicalChat();
        });
        
        // Global function for auto-greeting integration
        window.sendMessage = function(message) {
            if (window.enhancedChat) {
                window.enhancedChat.sendMessage(message);
            }
        };
    </script>
</body>
</html>
    '''
    
    return template

def create_optimized_chunk_processor():
    """Create optimized chunk processing system"""
    
    class OptimizedChunkProcessor:
        def __init__(self):
            self.chunk_cache = {}
            self.processing_stats = {
                'chunks_processed': 0,
                'cache_hits': 0,
                'processing_time': 0
            }
        
        def process_chunks_ultra_fast(self, chunks, batch_size=500):
            """Process chunks with maximum optimization"""
            import time
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            start_time = time.time()
            processed_chunks = []
            
            # Use maximum parallelism
            max_workers = min(32, len(chunks) // 10 + 1)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit chunks in batches
                futures = []
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    future = executor.submit(self._process_chunk_batch, batch)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        batch_results = future.result(timeout=60)
                        processed_chunks.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Chunk processing error: {e}")
            
            processing_time = time.time() - start_time
            self.processing_stats['chunks_processed'] += len(processed_chunks)
            self.processing_stats['processing_time'] += processing_time
            
            logger.info(f"⚡ Processed {len(processed_chunks)} chunks in {processing_time:.2f}s")
            return processed_chunks
        
        def _process_chunk_batch(self, chunk_batch):
            """Process a batch of chunks"""
            processed = []
            for chunk in chunk_batch:
                # Simple processing - could be enhanced based on needs
                processed_chunk = {
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'processed_at': time.time()
                }
                processed.append(processed_chunk)
            return processed
    
    return OptimizedChunkProcessor()

def run_integration_tests():
    """Run integration tests for all components"""
    
    logger.info("🧪 Running integration tests...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test auto-greeting system
    try:
        from auto_greeting_chat import AutoGreetingSystem
        greeting_system = AutoGreetingSystem()
        test_greeting = greeting_system.get_auto_greeting("test_session")
        assert 'greeting' in test_greeting
        assert 'suggestions' in test_greeting
        tests_passed += 1
        logger.info("✅ Auto-greeting system test passed")
    except Exception as e:
        logger.error(f"❌ Auto-greeting test failed: {e}")
    finally:
        tests_total += 1
    
    # Test admin panel integration
    try:
        from admin_panel_integration import RealTimeBookManager
        book_manager = RealTimeBookManager()
        books_status = book_manager.get_all_books_status()
        system_status = book_manager.get_system_status()
        assert isinstance(books_status, list)
        assert hasattr(system_status, 'is_indexing')
        tests_passed += 1
        logger.info("✅ Admin panel integration test passed")
    except Exception as e:
        logger.error(f"❌ Admin panel test failed: {e}")
    finally:
        tests_total += 1
    
    # Test ultra-fast indexer
    try:
        from store_index import UltimateConfig, UltimateMetrics
        config = UltimateConfig()
        metrics = UltimateMetrics()
        assert hasattr(config, 'BATCH_SIZE')
        assert hasattr(metrics, 'chunks_processed')
        tests_passed += 1
        logger.info("✅ Ultra-fast indexer test passed")
    except Exception as e:
        logger.error(f"❌ Ultra-fast indexer test failed: {e}")
    finally:
        tests_total += 1
    
    # Test chunk processor
    try:
        chunk_processor = create_optimized_chunk_processor()
        assert hasattr(chunk_processor, 'process_chunks_ultra_fast')
        tests_passed += 1
        logger.info("✅ Chunk processor test passed")
    except Exception as e:
        logger.error(f"❌ Chunk processor test failed: {e}")
    finally:
        tests_total += 1
    
    success_rate = (tests_passed / tests_total) * 100
    logger.info(f"🧪 Integration tests completed: {tests_passed}/{tests_total} passed ({success_rate:.1f}%)")
    
    return success_rate > 75

if __name__ == "__main__":
    # Run integration tests
    success = run_integration_tests()
    
    if success:
        logger.info("🎉 All integration tests passed! System ready for deployment.")
    else:
        logger.warning("⚠️ Some integration tests failed. Please review before deployment.")