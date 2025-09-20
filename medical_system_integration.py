"""
🏥 MEDICAL SYSTEM INTEGRATION HUB
=================================
Complete integration of all medical system components with fallback support
Handles authentication, conversation, book processing, and admin functions
"""

import os
import sys
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalSystemIntegration:
    """
    🚀 ULTIMATE MEDICAL SYSTEM INTEGRATION
    
    Unified interface for all medical system components:
    ✅ Authentication with fallback
    ✅ Conversation engine with fallback
    ✅ Book processing with fallback
    ✅ Admin panel with fallback
    ✅ Error recovery and graceful degradation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.components = {}
        self.fallback_mode = False
        self.system_status = {
            'auth_manager': 'unknown',
            'conversation_engine': 'unknown',
            'book_processor': 'unknown',
            'admin_dashboard': 'unknown'
        }
        
        logger.info("🏥 Initializing Medical System Integration Hub...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components with fallback support"""
        
        # Initialize Auth Manager
        try:
            from auth_manager import get_auth_manager
            self.components['auth_manager'] = get_auth_manager()
            self.system_status['auth_manager'] = 'active'
            logger.info("✅ Auth Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Auth Manager failed to initialize: {e}")
            self.system_status['auth_manager'] = 'fallback'
            self.components['auth_manager'] = None
        
        # Initialize Conversation Engine
        try:
            from enhanced_conversational_engine import create_conversational_engine
            engine_config = {
                'pinecone_api_key': self.config.get('pinecone_api_key', 'fallback_key'),
                'groq_api_key': self.config.get('groq_api_key', 'fallback_key'),
                'pinecone_index_name': self.config.get('pinecone_index_name', 'medical-books-ultimate')
            }
            self.components['conversation_engine'] = create_conversational_engine(engine_config)
            self.system_status['conversation_engine'] = 'active'
            logger.info("✅ Conversation Engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Conversation Engine failed to initialize: {e}")
            self.system_status['conversation_engine'] = 'fallback'
            self.components['conversation_engine'] = self._create_fallback_conversation_engine()
        
        # Initialize Book Processor
        try:
            from advanced_book_processor import UltimateMedicalBookProcessor
            self.components['book_processor'] = UltimateMedicalBookProcessor()
            self.system_status['book_processor'] = 'active'
            logger.info("✅ Book Processor initialized")
        except Exception as e:
            logger.warning(f"⚠️ Book Processor failed to initialize: {e}")
            self.system_status['book_processor'] = 'fallback'
            self.components['book_processor'] = None
        
        # Initialize Admin Dashboard
        try:
            from admin_panel_integration import EnhancedMedicalAdminDashboard
            self.components['admin_dashboard'] = EnhancedMedicalAdminDashboard()
            self.system_status['admin_dashboard'] = 'active'
            logger.info("✅ Admin Dashboard initialized")
        except Exception as e:
            logger.warning(f"⚠️ Admin Dashboard failed to initialize: {e}")
            self.system_status['admin_dashboard'] = 'fallback'
            self.components['admin_dashboard'] = self._create_fallback_admin_dashboard()
        
        # Check overall system health
        active_components = sum(1 for status in self.system_status.values() if status == 'active')
        total_components = len(self.system_status)
        
        if active_components == total_components:
            logger.info("🎉 All components initialized successfully!")
        elif active_components >= total_components * 0.5:
            logger.info(f"⚠️ System operational with {active_components}/{total_components} components active")
        else:
            logger.warning(f"❌ System in degraded mode with only {active_components}/{total_components} components active")
            self.fallback_mode = True
    
    def _create_fallback_conversation_engine(self):
        """Create fallback conversation engine"""
        class FallbackConversationEngine:
            def __init__(self):
                self.medical_responses = {
                    'fever': "For fever, monitor temperature, stay hydrated, and rest. Seek medical care if fever is high or persistent.",
                    'headache': "For headaches, try rest, hydration, and stress reduction. See a healthcare provider if severe or frequent.",
                    'nausea': "For nausea, try small frequent meals, stay hydrated, and rest. Contact a doctor if symptoms persist.",
                    'pain': "For pain management, follow healthcare provider instructions and avoid self-medication. Seek medical care for severe pain.",
                    'cough': "For coughs, stay hydrated, use throat lozenges, and consider a humidifier. See a doctor if symptoms worsen."
                }
            
            async def process_conversation(self, user_query: str, user_id: str, 
                                         session_id: str = None, user_profile: Dict = None,
                                         auth_context: Dict = None) -> Tuple[str, Dict]:
                """Fallback conversation processing"""
                query_lower = user_query.lower()
                
                # Check for emergency
                emergency_keywords = ['chest pain', 'difficulty breathing', 'unconscious', 'emergency', 'severe pain']
                if any(keyword in query_lower for keyword in emergency_keywords):
                    return """🚨 **MEDICAL EMERGENCY** 🚨
                    
This appears to be an emergency. Please:
• Call emergency services (911) immediately
• Go to the nearest emergency room
• Do not delay seeking professional medical care""", {
                        'type': 'emergency',
                        'fallback_mode': True
                    }
                
                # Look for medical topics
                for topic, response in self.medical_responses.items():
                    if topic in query_lower:
                        return f"**{topic.title()} Information:**\n{response}\n\n⚠️ **Important:** This is general information only. Please consult a healthcare professional for proper diagnosis and treatment.", {
                            'type': 'medical_info',
                            'topic': topic,
                            'fallback_mode': True
                        }
                
                # Generic response
                return """I understand you have a health question. While I can provide general information, I recommend:

**For medical concerns:**
• Contact your healthcare provider
• Describe your specific symptoms
• Mention duration and severity
• Share relevant medical history

**For emergencies:**
• Call 911 or emergency services
• Go to the nearest emergency room

**Remember:** This system provides educational information only and cannot replace professional medical care.""", {
                    'type': 'general_health',
                    'fallback_mode': True
                }
        
        return FallbackConversationEngine()
    
    def _create_fallback_admin_dashboard(self):
        """Create fallback admin dashboard"""
        class FallbackAdminDashboard:
            def __init__(self):
                self.start_time = datetime.now()
            
            def get_admin_dashboard_metrics(self, admin_email: str = None) -> Dict[str, Any]:
                """Fallback admin metrics"""
                return {
                    'timestamp': datetime.now().isoformat(),
                    'admin_requesting': admin_email,
                    'system_status': 'degraded_mode',
                    'fallback_mode': True,
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                    'message': 'System running in fallback mode. Limited functionality available.',
                    'quick_stats': {
                        'total_users': 0,
                        'active_sessions': 0,
                        'admin_users': 0,
                        'system_health': 'degraded'
                    }
                }
            
            def get_realtime_metrics(self) -> Dict:
                """Fallback realtime metrics"""
                return {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'fallback_mode',
                    'message': 'Limited metrics available in fallback mode'
                }
        
        return FallbackAdminDashboard()
    
    async def authenticate_user(self, email: str, password: str, provider: str = 'email') -> Dict[str, Any]:
        """Unified user authentication"""
        try:
            if self.components.get('auth_manager'):
                result = self.components['auth_manager'].login_user(email, password, provider)
                return {
                    'success': result.success,
                    'message': result.message,
                    'user_data': result.user_data,
                    'session_data': result.session_data,
                    'fallback_mode': False
                }
            else:
                # Fallback authentication (basic validation)
                if '@' in email and len(password) >= 6:
                    return {
                        'success': True,
                        'message': 'Authenticated in fallback mode',
                        'user_data': {'email': email, 'name': 'User'},
                        'fallback_mode': True
                    }
                else:
                    return {
                        'success': False,
                        'message': 'Invalid credentials',
                        'fallback_mode': True
                    }
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {
                'success': False,
                'message': f'Authentication error: {str(e)}',
                'fallback_mode': True
            }
    
    async def process_medical_conversation(self, user_query: str, user_id: str, 
                                         session_id: str = None, auth_context: Dict = None) -> Tuple[str, Dict]:
        """Unified medical conversation processing"""
        try:
            engine = self.components.get('conversation_engine')
            if engine:
                return await engine.process_conversation(
                    user_query, user_id, session_id, None, auth_context
                )
            else:
                logger.warning("Using fallback conversation engine")
                return await self.components['conversation_engine'].process_conversation(
                    user_query, user_id, session_id, None, auth_context
                )
        except Exception as e:
            logger.error(f"Conversation processing failed: {e}")
            return f"""I apologize, but I'm experiencing technical difficulties. 

For medical questions, please:
• Contact your healthcare provider
• Call your doctor's office  
• Visit a medical professional
• Use official medical resources

Your health and safety are important.""", {
                'error': str(e),
                'fallback_mode': True,
                'type': 'system_error'
            }
    
    async def upload_medical_book(self, file_path: str, filename: str, 
                                uploaded_by: str, auth_context: Dict = None) -> Dict[str, Any]:
        """Unified medical book upload"""
        try:
            if not auth_context or not auth_context.get('is_admin', False):
                return {
                    'success': False,
                    'message': 'Admin privileges required for book upload',
                    'error_code': 'INSUFFICIENT_PRIVILEGES'
                }
            
            processor = self.components.get('book_processor')
            if processor:
                return await processor.process_uploaded_file(
                    file_path, filename, uploaded_by, auth_context
                )
            else:
                return {
                    'success': False,
                    'message': 'Book processing system not available',
                    'fallback_mode': True,
                    'error_code': 'SERVICE_UNAVAILABLE'
                }
        except Exception as e:
            logger.error(f"Book upload failed: {e}")
            return {
                'success': False,
                'message': f'Book upload failed: {str(e)}',
                'error_code': 'SYSTEM_ERROR'
            }
    
    def get_admin_dashboard_metrics(self, admin_email: str = None) -> Dict[str, Any]:
        """Unified admin dashboard metrics"""
        try:
            dashboard = self.components.get('admin_dashboard')
            if dashboard:
                return dashboard.get_admin_dashboard_metrics(admin_email)
            else:
                return self.components['admin_dashboard'].get_admin_dashboard_metrics(admin_email)
        except Exception as e:
            logger.error(f"Admin dashboard metrics failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'fallback_mode': True,
                'system_status': 'error'
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'degraded' if self.fallback_mode else 'healthy',
            'components': self.system_status,
            'fallback_mode': self.fallback_mode,
            'active_components': sum(1 for status in self.system_status.values() if status == 'active'),
            'total_components': len(self.system_status)
        }

# Global system integration instance
_medical_system = None

def get_medical_system(config: Dict[str, Any] = None) -> MedicalSystemIntegration:
    """Get global medical system integration instance"""
    global _medical_system
    if _medical_system is None:
        _medical_system = MedicalSystemIntegration(config)
    return _medical_system

def initialize_medical_system(config: Dict[str, Any] = None) -> MedicalSystemIntegration:
    """Initialize the medical system with configuration"""
    global _medical_system
    _medical_system = MedicalSystemIntegration(config)
    return _medical_system

if __name__ == "__main__":
    # Test the medical system integration
    async def test_system():
        print("🧪 Testing Medical System Integration...")
        
        # Initialize system
        system = initialize_medical_system({
            'pinecone_api_key': 'test_key',
            'groq_api_key': 'test_key'
        })
        
        # Test authentication
        auth_result = await system.authenticate_user('test@example.com', 'password123')
        print(f"Auth test: {'✅' if auth_result['success'] else '❌'} {auth_result['message']}")
        
        # Test conversation
        conv_result = await system.process_medical_conversation(
            'I have a headache', 
            'user_123',
            auth_context={'is_admin': False}
        )
        print(f"Conversation test: ✅ Response generated")
        
        # Test system health
        health = system.get_system_health()
        print(f"System health: {health['overall_status']} ({health['active_components']}/{health['total_components']} components)")
        
        print("🎉 Medical System Integration test completed!")
    
    asyncio.run(test_system())