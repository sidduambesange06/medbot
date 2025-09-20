"""
Medical Chatbot Core - Rebuilt from Working Backup
Enhanced Medical AI with Context-Aware RAG System
"""

import os
import re
import requests
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from functools import wraps
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.status import HTTP_302_FOUND
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dataclasses import dataclass
import asyncio
from sentence_transformers import SentenceTransformer
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# API Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ Missing API keys in .env file")

# Medical Keywords for Query Classification
MEDICAL_KEYWORDS = [
    'disease', 'cancer', 'tumor', 'diabetes', 'hypertension', 'asthma', 'pneumonia',
    'tuberculosis', 'malaria', 'dengue', 'covid', 'flu', 'fever', 'infection',
    'hepatitis', 'cirrhosis', 'arthritis', 'osteoporosis', 'anemia', 'leukemia',
    'stroke', 'heart attack', 'angina', 'depression', 'anxiety', 'migraine',
    'alzheimer', 'parkinson', 'epilepsy', 'schizophrenia', 'bipolar',
    'pain', 'headache', 'nausea', 'vomiting', 'diarrhea', 'constipation',
    'cough', 'cold', 'sore throat', 'fatigue', 'weakness', 'dizziness',
    'shortness of breath', 'chest pain', 'abdominal pain', 'back pain',
    'joint pain', 'swelling', 'rash', 'itching', 'bleeding', 'bruising',
    'diagnosis', 'treatment', 'therapy', 'cure', 'medicine', 'medication',
    'drug', 'antibiotic', 'vaccine', 'surgery', 'operation', 'procedure',
    'examination', 'test', 'lab', 'x-ray', 'ct scan', 'mri', 'ultrasound',
    'biopsy', 'endoscopy', 'chemotherapy', 'radiotherapy', 'immunotherapy',
    'heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine',
    'blood', 'bone', 'muscle', 'nerve', 'skin', 'eye', 'ear', 'nose',
    'throat', 'respiratory', 'cardiovascular', 'gastrointestinal',
    'neurological', 'endocrine', 'immune', 'reproductive',
    'patient', 'doctor', 'physician', 'hospital', 'clinic', 'pharmacy',
    'prescription', 'dosage', 'side effect', 'allergy', 'symptom',
    'prevention', 'precaution', 'risk factor', 'complication',
    'what is', 'how to treat', 'cure for', 'symptoms of', 'causes of',
    'medicine for', 'drug for', 'prevention of', 'risk of'
]

@dataclass
class ChatMessage:
    role: str
    message: str
    timestamp: datetime
    metadata: Dict = None
    relevance_score: float = 1.0

@dataclass
class ConversationSummary:
    summary: str
    key_topics: List[str]
    message_count: int
    timespan: str

class OptimizedMedicalRetriever:
    """Optimized medical knowledge retriever with production-ready responses"""
    
    def __init__(self):
        logger.info("🏥 Initializing Medical Knowledge Retriever...")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.vectorstore = PineconeVectorStore.from_existing_index(
                index_name="medical-chatbot-v2",
                embedding=self.embeddings
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
            
            logger.info("✅ Medical Knowledge Retriever ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize medical retriever: {e}")
            self.retriever = None
    
    def is_medical_query(self, query: str) -> bool:
        """Enhanced medical query detection"""
        query_lower = query.lower()
        
        # Direct keyword matching
        for keyword in MEDICAL_KEYWORDS:
            if keyword in query_lower:
                return True
        
        # Medical question patterns
        medical_patterns = [
            r'\b(what|how|why|when|where)\s+.*\b(disease|symptom|treatment|medicine|drug|cure|diagnosis)\b',
            r'\b(symptoms?|causes?|treatment|cure|medicine|drug)\s+(of|for)\b',
            r'\b(how\s+to\s+(treat|cure|prevent|diagnose))\b',
            r'\b(side\s+effects?|dosage|prescription)\b',
            r'\b(medical|health|clinical|therapeutic)\b'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def retrieve_medical_knowledge(self, query: str) -> Dict:
        """Retrieve and organize medical knowledge for production responses"""
        if not self.retriever:
            return {'has_knowledge': False, 'content': '', 'sources': [], 'chunks_found': 0}
            
        try:
            logger.info(f"🔍 Searching medical textbooks for: {query}")
            
            docs = self.retriever.invoke(query)
            
            if not docs:
                return {
                    'has_knowledge': False,
                    'content': '',
                    'sources': [],
                    'chunks_found': 0
                }
            
            medical_info = self._process_medical_content(docs, query)
            logger.info(f"✅ Retrieved {medical_info['chunks_found']} relevant medical chunks")
            
            return medical_info
            
        except Exception as e:
            logger.error(f"❌ Error retrieving medical knowledge: {e}")
            return {
                'has_knowledge': False,
                'content': '',
                'sources': [],
                'chunks_found': 0,
                'error': str(e)
            }
    
    def _process_medical_content(self, docs, query: str) -> Dict:
        """Process medical chunks for optimized, production-ready content"""
        
        relevant_chunks = []
        sources = []
        unique_books = set()
        
        for doc in docs:
            if not hasattr(doc, 'page_content') or len(doc.page_content.strip()) < 100:
                continue
            
            content = doc.page_content.strip()
            metadata = doc.metadata
            
            book_name = metadata.get('filename', metadata.get('source', 'Medical Textbook'))
            if book_name.endswith('.pdf'):
                book_name = book_name.replace('.pdf', '').replace('_', ' ').title()
            
            page_num = metadata.get('page', 'N/A')
            
            chunk_info = {
                'content': content,
                'book_name': book_name,
                'page': page_num,
                'relevance': len([kw for kw in MEDICAL_KEYWORDS if kw in content.lower()])
            }
            
            relevant_chunks.append(chunk_info)
            
            source_key = f"{book_name}_{page_num}"
            if source_key not in [s['key'] for s in sources]:
                sources.append({
                    'book_name': book_name,
                    'page': page_num,
                    'key': source_key
                })
                unique_books.add(book_name)
        
        if not relevant_chunks:
            return {
                'has_knowledge': False,
                'content': '',
                'sources': [],
                'chunks_found': 0
            }
        
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x['relevance'], reverse=True)
        optimized_content = self._create_optimized_content(sorted_chunks[:4], query)
        
        return {
            'has_knowledge': True,
            'content': optimized_content,
            'sources': sources[:5],
            'chunks_found': len(relevant_chunks),
            'unique_books': len(unique_books),
            'top_chunks': sorted_chunks[:3]
        }
    
    def _create_optimized_content(self, chunks: List[Dict], query: str) -> str:
        """Create optimized medical content with clear source attribution"""
        
        content_sections = []
        
        for chunk in chunks:
            sentences = chunk['content'].split('. ')
            relevant_sentences = []
            
            query_terms = query.lower().split()
            for sentence in sentences:
                sentence_lower = sentence.lower()
                relevance_score = sum(1 for term in query_terms if term in sentence_lower)
                if relevance_score > 0 or len(relevant_sentences) < 2:
                    relevant_sentences.append(sentence.strip())
                if len(relevant_sentences) >= 3:
                    break
            
            if relevant_sentences:
                section = '. '.join(relevant_sentences)
                if not section.endswith('.'):
                    section += '.'
                section += f" [{chunk['book_name']}, p.{chunk['page']}]"
                content_sections.append(section)
        
        return '\n\n'.join(content_sections)

class ProductionGroqInterface:
    """Production-optimized Groq API interface with refined prompts"""
    
    def __init__(self):
        self.available_models = [
            "llama-3.1-8b-instant",       # Updated working model (Jan 2025)
            "llama-3.1-70b-versatile",    # High-performance model
            "mixtral-8x7b-32768", 
            "gemma2-9b-it"                # Backup model
        ]
    
    def call_groq_api(self, system_prompt: str, user_query: str, temperature: float = 0.1) -> Optional[str]:
        """Call Groq API with error handling"""
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
    
        for model in self.available_models:
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    "model": model,
                    "max_tokens": 400,
                    "temperature": temperature,
                    "top_p": 0.9,
                }
                
                response = requests.post(
                    GROQ_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0]["message"]["content"]
                        if content and len(content.strip()) > 10:
                            return content.strip()
                
                elif response.status_code == 429:
                    continue
                    
            except Exception as e:
                logger.warning(f"❌ {model} failed: {e}")
                continue
        
        return None
    
    def get_production_medical_prompt(self, medical_content: str, sources: List[Dict], query: str) -> str:
        """Production-optimized system prompt for medical queries"""
        
        return f"""You are a professional Medical AI Assistant providing evidence-based medical information from authoritative medical textbooks.

QUERY: {query}

MEDICAL TEXTBOOK CONTENT:
{medical_content}

RESPONSE GUIDELINES:
1. Provide a concise, professional medical response (150-300 words maximum)
2. Focus on directly answering the user's question
3. Use clear, accessible medical language
4. Organize information logically (definition → symptoms → causes → treatment → prevention)
5. Include only the most relevant and important information
6. Maintain professional medical tone
7. All citations are already included in the content - do not add additional ones

STRUCTURE YOUR RESPONSE:
- Brief, clear answer to the user's specific question
- Key medical facts organized logically
- Essential information only (avoid lengthy explanations)
- Professional medical terminology with brief explanations when needed

Remember: This is a production medical assistant - responses should be concise, authoritative, and directly helpful to healthcare queries.

Always end with: "⚠️ This information is from medical textbooks for educational purposes. Always consult healthcare professionals for medical advice."
"""

class EnhancedChatService:
    def __init__(self):
        self.chat_history = {}  # In-memory storage
        self.conversation_summaries = {}
        self.user_preferences = {}
    
    async def store_message_with_context(self, user_id: str, session_id: str, role: str, message: str, metadata: Dict = None):
        """Store message with context"""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        
        chat_message = ChatMessage(
            role=role,
            message=message,
            timestamp=datetime.now(),
            metadata={
                'session_id': session_id,
                **(metadata or {})
            }
        )
        
        self.chat_history[user_id].append(chat_message)
    
    async def get_recent_history(self, user_id: str, session_id: str = None, limit: int = 10) -> List[ChatMessage]:
        """Get recent chat history"""
        user_history = self.chat_history.get(user_id, [])
        
        if session_id:
            session_history = [msg for msg in user_history if msg.metadata and msg.metadata.get('session_id') == session_id]
            return session_history[-limit:] if session_history else []
        
        return user_history[-limit:] if user_history else []

class ProductionMedicalChatbot:
    """Production-ready Medical Chatbot with enhanced context awareness"""
    
    def __init__(self):
        self.medical_retriever = OptimizedMedicalRetriever()
        self.groq_interface = ProductionGroqInterface()
        self.chat_service = EnhancedChatService()
        
        self.identity_response = """
I am **Med-Ai**, your advanced medical AI assistant powered by evidence-based medical textbook knowledge.

**Core Identity:**
• **Medical AI** specialized in healthcare information
• **Knowledge Source**: Authoritative medical textbooks & clinical literature
• **Features**: Context-aware memory, session tracking, personalized responses
• **Authentication**: OAuth + Guest sessions
• **Technology**: RAG with Pinecone vector database

⚠️ **Important**: I provide educational medical information — always consult healthcare professionals for medical advice.
"""
        logger.info("🏥 Production Medical Chatbot initialized")

    async def process_query_with_context(self, query: str, user_context: Dict = None, session_id: str = None) -> str:
        """Advanced medical-only query processor with identity handling and context awareness"""

        query_lower = query.lower().strip()
        user_id = user_context.get('id', 'anonymous') if user_context else 'anonymous'

        # Identity Question Handling
        identity_triggers = [
            "who are you", "what are you", "tell me about yourself",
            "introduce yourself", "what is med-ai", "explain yourself"
        ]
        if any(trigger in query_lower for trigger in identity_triggers):
            logger.info("🎯 Identity question detected")
            await self.chat_service.store_message_with_context(
                user_id, session_id or str(uuid.uuid4()), "user", query
            )
            await self.chat_service.store_message_with_context(
                user_id, session_id or str(uuid.uuid4()), "assistant", self.identity_response
            )
            return self.identity_response

        # Medical Query Validation
        classification = self.medical_retriever.is_medical_query(query)
        if not classification:
            logger.info("🚫 Non-medical query blocked")
            warning = (
                "❌ This is a **medical chatbot**. I only respond to **medical or health-related questions**.\n\n"
                "Examples:\n"
                "- What are the symptoms of diabetes?\n"
                "- How to treat high blood pressure?\n"
                "- What causes migraines?\n\n"
                "Please ask a **medical or health-related question**."
            )
            await self.chat_service.store_message_with_context(
                user_id, session_id or str(uuid.uuid4()), "user", query
            )
            await self.chat_service.store_message_with_context(
                user_id, session_id or str(uuid.uuid4()), "assistant", warning
            )
            return warning

        # Medical Query Processing
        logger.info("🏥 Medical query detected")
        
        medical_knowledge = self.medical_retriever.retrieve_medical_knowledge(query)

        if medical_knowledge['has_knowledge']:
            logger.info(f"✅ Found medical knowledge from {medical_knowledge['unique_books']} textbooks")

            system_prompt = self.groq_interface.get_production_medical_prompt(
                medical_knowledge['content'],
                medical_knowledge['sources'],
                query
            )

            response = self.groq_interface.call_groq_api(system_prompt, query, temperature=0.1)

            if response:
                if medical_knowledge['sources']:
                    source_info = "\n\n📚 **Sources:** " + ", ".join([
                        f"{s['book_name']} (p.{s['page']})"
                        for s in medical_knowledge['sources'][:3]
                    ])
                    response += source_info
            else:
                response = self._get_fallback_response(query, medical_knowledge)
        else:
            logger.info("❌ No textbook knowledge found")
            response = (
                f"I couldn't find specific information about **{query}** in the medical textbook database.\n\n"
                "For accurate, personalized medical information, I recommend:\n"
                "• Consulting a licensed healthcare provider\n"
                "• Speaking with a medical specialist\n"
                "• Referring to current clinical guidelines\n\n"
                "⚠️ For medical concerns, always consult qualified healthcare professionals."
            )

        # Store conversation
        await self.chat_service.store_message_with_context(
            user_id, session_id or str(uuid.uuid4()), "user", query
        )
        await self.chat_service.store_message_with_context(
            user_id, session_id or str(uuid.uuid4()), "assistant", response
        )

        return response
    
    def _get_fallback_response(self, query: str, medical_knowledge: Dict) -> str:
        """Generate fallback response when API fails but we have medical sources"""
        
        sources_text = ""
        if medical_knowledge['sources']:
            sources_text = "\n\n📚 **Relevant Sources Found:**\n" + "\n".join([
                f"• {source['book_name']}, Page {source['page']}"
                for source in medical_knowledge['sources'][:3]
            ])
        
        return f"""I found relevant medical information in the textbook database but encountered a technical issue generating the response.

Please refer to the medical textbook sources below for information about "{query}".
{sources_text}

⚠️ This information is from medical textbooks for educational purposes. Always consult healthcare professionals for medical advice."""

# FastAPI Authentication utilities
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from starlette.requests import Request

security = HTTPBearer()

async def get_current_user(request: Request):
    """Get current user from session/token"""
    # FastAPI session handling would be implemented here
    # For now, return a mock user for compatibility
    return {
        'id': 'user_123',
        'email': 'user@medai.pro',
        'name': 'FastAPI User',
        'role': 'user',
        'avatar': None,
        'is_guest': False
    }

async def get_user_context(request: Request):
    """Get current user context for logging and personalization"""
    try:
        user = await get_current_user(request)
        return user
    except:
        return {
            'id': 'guest_user',
            'email': 'guest@medai.pro',
            'name': 'Guest User',
            'role': 'guest',
            'avatar': None,
            'is_guest': True
        }