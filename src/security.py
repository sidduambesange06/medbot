"""
Security Manager for Medical Chatbot
Handles authentication, rate limiting, input sanitization, and security validations
"""

import jwt
import re
import hashlib
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, g
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.rate_limit_storage = {}  # In production, use Redis
        self.blocked_ips = set()
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'eval\s*\(',
            r'document\.cookie',
        ]
        
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        try:
            # Generate salt
            salt = os.urandom(32)
            # Hash password with salt
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            # Store salt and hash together
            return salt.hex() + pwd_hash.hex()
        except Exception as e:
            logger.error(f"Password hashing error: {e}")
            raise Exception("Failed to hash password")
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            # Extract salt and hash
            salt = bytes.fromhex(stored_hash[:64])
            stored_pwd_hash = stored_hash[64:]
            # Hash the provided password with the stored salt
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return pwd_hash.hex() == stored_pwd_hash
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
        
    def verify_token(self, token: str) -> Dict:
        """Verify JWT token with proper validation."""
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=["HS256"],
                options={
                    "require_exp": True,
                    "verify_signature": True,
                    "verify_exp": True
                }
            )
            
            # Additional validation
            if not payload.get('sub'):
                raise jwt.InvalidTokenError("Missing subject")
                
            if not payload.get('iat'):
                raise jwt.InvalidTokenError("Missing issued at")
            
            logger.info(f"Token verified for user: {payload.get('sub')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise jwt.ExpiredSignatureError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise jwt.InvalidTokenError(f"Token verification failed: {str(e)}")
    
    def generate_token(self, user_data: Dict, expires_in: int = 3600) -> str:
        """Generate a new JWT token."""
        try:
            payload = {
                'sub': user_data.get('id') or user_data.get('sub'),
                'email': user_data.get('email'),
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(seconds=expires_in),
                'iss': 'medical-chatbot',
                'aud': 'medical-chatbot-users'
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            logger.info(f"Token generated for user: {payload['sub']}")
            return token
            
        except Exception as e:
            logger.error(f"Token generation error: {e}")
            raise Exception(f"Failed to generate token: {str(e)}")
    
    def rate_limit_check(self, user_id: str, action: str = "chat", limit: int = 100, window: int = 3600) -> bool:
        """Implement rate limiting per user."""
        try:
            current_time = time.time()
            key = f"{user_id}:{action}"
            
            if key not in self.rate_limit_storage:
                self.rate_limit_storage[key] = []
            
            # Clean old requests outside the window
            self.rate_limit_storage[key] = [
                timestamp for timestamp in self.rate_limit_storage[key]
                if current_time - timestamp < window
            ]
            
            # Check if limit exceeded
            if len(self.rate_limit_storage[key]) >= limit:
                logger.warning(f"Rate limit exceeded for user {user_id}, action {action}")
                return False
            
            # Add current request
            self.rate_limit_storage[key].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error to avoid blocking legitimate users
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent XSS and injection attacks."""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Remove suspicious patterns
            sanitized = text
            for pattern in self.suspicious_patterns:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            # Remove null bytes
            sanitized = sanitized.replace('\x00', '')
            
            # Limit length
            if len(sanitized) > 10000:  # 10KB limit
                sanitized = sanitized[:10000]
                logger.warning("Input truncated due to length")
            
            # Remove excessive whitespace
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Input sanitization error: {e}")
            return text[:1000]  # Return truncated original on error
    
    def validate_medical_query(self, query: str) -> Dict[str, bool]:
        """Validate if the query is appropriate for a medical chatbot."""
        validation_result = {
            'is_appropriate': True,
            'is_medical_related': False,
            'contains_personal_info': False,
            'is_emergency': False,
            'needs_disclaimer': False
        }
        
        try:
            query_lower = query.lower()
            
            # Check for medical keywords
            medical_keywords = [
                'symptom', 'pain', 'ache', 'fever', 'doctor', 'medicine', 'treatment',
                'diagnosis', 'health', 'medical', 'hospital', 'prescription', 'drug',
                'medication', 'disease', 'condition', 'therapy', 'cure', 'heal'
            ]
            
            validation_result['is_medical_related'] = any(
                keyword in query_lower for keyword in medical_keywords
            )
            
            # Check for personal information (basic patterns)
            personal_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr)\b'  # Address
            ]
            
            validation_result['contains_personal_info'] = any(
                re.search(pattern, query, re.IGNORECASE) for pattern in personal_patterns
            )
            
            # Check for emergency keywords
            emergency_keywords = [
                'emergency', 'urgent', 'critical', 'dying', 'suicide', 'overdose',
                'chest pain', 'can\'t breathe', 'bleeding heavily', 'unconscious',
                'heart attack', 'stroke', 'poisoning', 'severe pain'
            ]
            
            validation_result['is_emergency'] = any(
                keyword in query_lower for keyword in emergency_keywords
            )
            
            # Check if disclaimer needed
            disclaimer_triggers = [
                'diagnosis', 'treatment', 'medication', 'should i take', 'is it safe',
                'dosage', 'side effects', 'stop taking', 'replace my doctor'
            ]
            
            validation_result['needs_disclaimer'] = any(
                trigger in query_lower for trigger in disclaimer_triggers
            )
            
            # Check for inappropriate content
            inappropriate_keywords = [
                'illegal drug', 'recreational drug', 'how to get high', 'fake prescription',
                'doctor shopping', 'drug dealing', 'abuse medication'
            ]
            
            if any(keyword in query_lower for keyword in inappropriate_keywords):
                validation_result['is_appropriate'] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Query validation error: {e}")
            return validation_result
    
    def check_ip_reputation(self, ip_address: str) -> bool:
        """Check if IP address is blocked or suspicious."""
        try:
            if ip_address in self.blocked_ips:
                logger.warning(f"Blocked IP attempted access: {ip_address}")
                return False
            
            # Add more sophisticated IP reputation checking here
            # e.g., check against known malicious IP databases
            
            return True
            
        except Exception as e:
            logger.error(f"IP reputation check error: {e}")
            return True  # Allow on error
    
    def log_security_event(self, event_type: str, details: Dict, severity: str = "INFO"):
        """Log security events for monitoring."""
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'severity': severity,
                'details': details,
                'ip_address': request.remote_addr if request else 'unknown'
            }
            
            if severity == "WARNING":
                logger.warning(f"Security Event: {event_type} - {details}")
            elif severity == "ERROR":
                logger.error(f"Security Alert: {event_type} - {details}")
            else:
                logger.info(f"Security Log: {event_type} - {details}")
                
        except Exception as e:
            logger.error(f"Security logging error: {e}")

# Decorators for Flask routes
def require_auth(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'error': 'No authorization token provided'}), 401
            
            security_manager = SecurityManager()
            payload = security_manager.verify_token(auth_header)
            g.current_user = payload
            
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
    
    return decorated_function

def rate_limit(limit: int = 100, window: int = 3600, action: str = "general"):
    """Decorator to apply rate limiting to routes."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                user_id = g.get('current_user', {}).get('sub', request.remote_addr)
                
                security_manager = SecurityManager()
                
                if not security_manager.rate_limit_check(user_id, action, limit, window):
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'message': f'Too many requests. Limit: {limit} per {window} seconds'
                    }), 429
                
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Rate limiting error: {e}")
                return f(*args, **kwargs)  # Allow on error
        
        return decorated_function
    return decorator

def sanitize_inputs(fields: List[str]):
    """Decorator to sanitize specified input fields."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                security_manager = SecurityManager()
                
                if request.is_json:
                    data = request.get_json()
                    if data:
                        for field in fields:
                            if field in data and isinstance(data[field], str):
                                data[field] = security_manager.sanitize_input(data[field])
                        request._cached_json = data
                
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Input sanitization error: {e}")
                return f(*args, **kwargs)  # Continue on error
        
        return decorated_function
    return decorator