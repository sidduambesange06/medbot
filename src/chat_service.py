"""
Enhanced Chat Service - Context-Aware Medical Chatbot Backend
Handles intelligent context retrieval, conversation management, and vector operations
"""

import asyncio
import uuid
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class EnhancedPromptBuilder:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        
    def build_contextual_prompt(
        self,
        user_input: str,
        recent_history: List[ChatMessage],
        summaries: List[ConversationSummary] = None,
        semantic_matches: List[ChatMessage] = None,
        user_preferences: Dict = None,
        medical_content: str = ""
    ) -> str:
        """Build a comprehensive contextual prompt with multiple context sources."""
        
        prompt_parts = []
        
        # System prompt with user context
        system_prompt = self._build_system_prompt(user_preferences)
        prompt_parts.append(system_prompt)
        
        # Add medical content from RAG
        if medical_content:
            prompt_parts.append(f"\n=== Medical Knowledge Base ===\n{medical_content}")
        
        # Add conversation summaries for long-term context
        if summaries:
            summary_context = self._build_summary_context(summaries)
            prompt_parts.append(summary_context)
        
        # Add semantically relevant messages
        if semantic_matches:
            semantic_context = self._build_semantic_context(semantic_matches)
            prompt_parts.append(semantic_context)
        
        # Add recent conversation history
        if recent_history:
            recent_context = self._build_recent_context(recent_history)
            prompt_parts.append(recent_context)
        
        # Add current user input
        prompt_parts.append(f"\nCurrent User Query: {user_input}")
        prompt_parts.append("\nPlease provide a comprehensive medical response based on the above context and knowledge:")
        
        # Combine and ensure token limit
        full_prompt = "\n".join(prompt_parts)
        return self._truncate_to_token_limit(full_prompt)
    
    def _build_system_prompt(self, user_preferences: Dict = None) -> str:
        base_prompt = """You are MedBot, an intelligent medical assistant with access to medical textbooks and conversation history.

CORE CAPABILITIES:
- Provide evidence-based medical information from textbooks
- Remember previous conversations and build upon them
- Ask clarifying questions when context is missing
- Maintain conversation continuity across sessions
- Cite sources from medical textbooks when available

IMPORTANT GUIDELINES:
- Always prioritize patient safety
- Recommend consulting healthcare professionals for serious concerns  
- Be clear about limitations of AI medical advice
- Use conversation context to provide personalized responses"""

        if user_preferences:
            if user_preferences.get('communication_style'):
                base_prompt += f"\nCommunication style: {user_preferences['communication_style']}"
            if user_preferences.get('medical_background'):
                base_prompt += f"\nUser's medical background: {user_preferences['medical_background']}"
        
        return base_prompt
    
    def _build_summary_context(self, summaries: List[ConversationSummary]) -> str:
        if not summaries:
            return ""
        
        context = "\n=== Previous Conversation Summaries ==="
        for summary in summaries[:3]:  # Limit to 3 most relevant summaries
            context += f"\n[{summary.timespan}] {summary.summary}"
            if summary.key_topics:
                context += f" (Topics: {', '.join(summary.key_topics)})"
        
        return context + "\n"
    
    def _build_semantic_context(self, semantic_matches: List[ChatMessage]) -> str:
        if not semantic_matches:
            return ""
        
        context = "\n=== Relevant Previous Messages ==="
        for msg in semantic_matches[:5]:  # Limit to top 5 matches
            days_ago = (datetime.now() - msg.timestamp).days
            time_ref = f"{days_ago} days ago" if days_ago > 0 else "today"
            context += f"\n[{time_ref}] {msg.role.capitalize()}: {msg.message}"
        
        return context + "\n"
    
    def _build_recent_context(self, recent_history: List[ChatMessage]) -> str:
        if not recent_history:
            return ""
        
        context = "\n=== Recent Conversation ==="
        for msg in recent_history:
            context += f"\n{msg.role.capitalize()}: {msg.message}"
        
        return context
    
    def _truncate_to_token_limit(self, prompt: str) -> str:
        # Simple token estimation (4 chars ≈ 1 token)
        estimated_tokens = len(prompt) // 4
        if estimated_tokens <= self.max_tokens:
            return prompt
        
        # Truncate from the middle, keeping system prompt and recent context
        lines = prompt.split('\n')
        system_lines = []
        recent_lines = []
        middle_lines = []
        
        in_recent = False
        for line in lines:
            if "Recent Conversation" in line:
                in_recent = True
            
            if not middle_lines and not in_recent:
                system_lines.append(line)
            elif in_recent:
                recent_lines.append(line)
            else:
                middle_lines.append(line)
        
        # Keep system and recent, truncate middle
        truncated = system_lines + ["...[previous context truncated]..."] + recent_lines[-10:]
        return '\n'.join(truncated)


class EnhancedChatService:
    def __init__(self):
        # Initialize Supabase client
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("Missing Supabase credentials in environment variables")
            raise ValueError("Missing Supabase credentials")
            
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        self.prompt_builder = EnhancedPromptBuilder()
        logger.info("🚀 Enhanced Chat Service initialized successfully")
    
    async def get_user_context_preferences(self, user_id: str) -> Dict:
        """Get user's context preferences or create defaults."""
        try:
            result = self.supabase.table("user_context_preferences")\
                .select("*")\
                .eq("user_id", user_id)\
                .execute()
            
            if result.data:
                return result.data[0]
            
            # Create default preferences
            default_prefs = {
                "user_id": user_id,
                "max_context_messages": 10,
                "context_strategy": "mixed",
                "include_summaries": True,
                "auto_summarize_threshold": 50,
                "preferences": {}
            }
            
            self.supabase.table("user_context_preferences")\
                .insert(default_prefs)\
                .execute()
            
            logger.info(f"✅ Created default preferences for user {user_id}")
            return default_prefs
            
        except Exception as e:
            logger.error(f"❌ Error getting user preferences: {e}")
            return {
                "max_context_messages": 10,
                "context_strategy": "mixed",
                "include_summaries": True,
                "auto_summarize_threshold": 50,
                "preferences": {}
            }
    
    async def get_intelligent_context(
        self, 
        user_id: str, 
        current_input: str,
        session_id: str = None
    ) -> Tuple[List[ChatMessage], List[ConversationSummary], List[ChatMessage]]:
        """Get intelligent context using multiple strategies."""
        
        try:
            preferences = await self.get_user_context_preferences(user_id)
            strategy = preferences.get("context_strategy", "mixed")
            max_messages = preferences.get("max_context_messages", 10)
            
            # Get recent history
            recent_history = await self._get_recent_history(user_id, session_id, max_messages)
            
            # Get conversation summaries if enabled
            summaries = []
            if preferences.get("include_summaries", True):
                summaries = await self._get_relevant_summaries(user_id, current_input)
            
            # Get semantically similar messages
            semantic_matches = []
            if strategy in ["semantic", "mixed"]:
                semantic_matches = await self._get_semantic_matches(user_id, current_input)
            
            return recent_history, summaries, semantic_matches
            
        except Exception as e:
            logger.error(f"❌ Error getting intelligent context: {e}")
            return [], [], []
    
    async def _get_recent_history(
        self, 
        user_id: str, 
        session_id: str = None, 
        limit: int = 10
    ) -> List[ChatMessage]:
        """Get recent chat history."""
        try:
            query = self.supabase.table("chat_history")\
                .select("role,message,timestamp,metadata")\
                .eq("user_id", user_id)\
                .order("timestamp", desc=True)\
                .limit(limit)
            
            if session_id:
                query = query.eq("session_id", session_id)
            
            result = query.execute()
            
            messages = []
            for item in reversed(result.data):
                # Handle timestamp parsing
                timestamp_str = item["timestamp"]
                if isinstance(timestamp_str, str):
                    if timestamp_str.endswith('Z'):
                        timestamp_str = timestamp_str.replace('Z', '+00:00')
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.now()
                
                messages.append(ChatMessage(
                    role=item["role"],
                    message=item["message"],
                    timestamp=timestamp,
                    metadata=item.get("metadata", {})
                ))
            
            return messages
            
        except Exception as e:
            logger.error(f"❌ Error getting recent history: {e}")
            return []
    
    async def _get_semantic_matches(
        self, 
        user_id: str, 
        query_text: str, 
        limit: int = 5
    ) -> List[ChatMessage]:
        """Get semantically similar messages using vector search."""
        try:
            # Generate embedding for current input
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # For now, get recent messages and compute similarity in Python
            # In production, you'd use Supabase vector search
            result = self.supabase.table("chat_history")\
                .select("role,message,timestamp,embedding")\
                .eq("user_id", user_id)\
                .limit(50)\
                .execute()
            
            messages = []
            for item in result.data:
                if item.get("embedding"):
                    # Calculate cosine similarity
                    item_embedding = np.array(item["embedding"])
                    query_emb = np.array(query_embedding)
                    similarity = np.dot(item_embedding, query_emb) / (
                        np.linalg.norm(item_embedding) * np.linalg.norm(query_emb)
                    )
                    
                    if similarity > 0.7:  # Threshold for relevance
                        timestamp_str = item["timestamp"]
                        if isinstance(timestamp_str, str):
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.now()
                        
                        messages.append(ChatMessage(
                            role=item["role"],
                            message=item["message"],
                            timestamp=timestamp,
                            relevance_score=float(similarity)
                        ))
            
            # Sort by similarity and return top matches
            messages.sort(key=lambda x: x.relevance_score, reverse=True)
            return messages[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error getting semantic matches: {e}")
            return []
    
    async def _get_relevant_summaries(
        self, 
        user_id: str, 
        query_text: str
    ) -> List[ConversationSummary]:
        """Get relevant conversation summaries."""
        try:
            result = self.supabase.table("conversation_summaries")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("end_time", desc=True)\
                .limit(5)\
                .execute()
            
            summaries = []
            for item in result.data:
                start_time = item.get('start_time', '')[:10] if item.get('start_time') else 'unknown'
                end_time = item.get('end_time', '')[:10] if item.get('end_time') else 'unknown'
                timespan = f"{start_time} to {end_time}"
                
                summaries.append(ConversationSummary(
                    summary=item.get("summary", ""),
                    key_topics=item.get("key_topics", []),
                    message_count=item.get("message_count", 0),
                    timespan=timespan
                ))
            
            return summaries
            
        except Exception as e:
            logger.error(f"❌ Error getting relevant summaries: {e}")
            return []
    
    async def store_message_with_embedding(
        self,
        user_id: str,
        session_id: str,
        role: str,
        message: str,
        conversation_id: str = None,
        metadata: Dict = None
    ):
        """Store message with vector embedding."""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(message).tolist()
            
            # Store in database
            data = {
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "message": message,
                "embedding": embedding,
                "metadata": metadata or {},
                "message_tokens": int(len(message.split()) * 1.3)  # Rough estimate
            }
            
            if conversation_id:
                data["conversation_id"] = conversation_id
            
            result = self.supabase.table("chat_history").insert(data).execute()
            
            if result.data:
                logger.info(f"✅ Stored message for user {user_id}")
            else:
                logger.warning(f"⚠️ No data returned when storing message for user {user_id}")
                
        except Exception as e:
            logger.error(f"❌ Error storing message: {e}")
    
    async def should_summarize_conversation(self, user_id: str, session_id: str) -> bool:
        """Check if conversation should be summarized."""
        try:
            preferences = await self.get_user_context_preferences(user_id)
            threshold = preferences.get("auto_summarize_threshold", 50)
            
            result = self.supabase.table("chat_history")\
                .select("id", count="exact")\
                .eq("user_id", user_id)\
                .eq("session_id", session_id)\
                .execute()
            
            return result.count >= threshold
            
        except Exception as e:
            logger.error(f"❌ Error checking summarization threshold: {e}")
            return False
    
    async def create_conversation_summary(self, user_id: str, session_id: str):
        """Create a summary of the conversation."""
        try:
            # Get all messages in session
            messages = await self._get_recent_history(user_id, session_id, limit=1000)
            
            if len(messages) < 5:  # Not enough for summary
                return
            
            # Create summary using simple extraction for now
            conversation_text = "\n".join([f"{msg.role}: {msg.message}" for msg in messages])
            
            # Simple summary - in production, use LLM
            summary = f"Medical conversation with {len(messages)} messages covering various health topics."
            
            # Extract key topics (simple keyword extraction)
            key_topics = self._extract_key_topics(conversation_text)
            
            # Generate embedding for summary
            summary_embedding = self.embedding_model.encode(summary).tolist()
            
            # Store summary
            self.supabase.table("conversation_summaries").insert({
                "user_id": user_id,
                "conversation_id": session_id,
                "summary": summary,
                "key_topics": key_topics,
                "message_count": len(messages),
                "start_time": messages[0].timestamp.isoformat() if messages else datetime.now().isoformat(),
                "end_time": messages[-1].timestamp.isoformat() if messages else datetime.now().isoformat(),
                "embedding": summary_embedding
            }).execute()
            
            logger.info(f"✅ Created conversation summary for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Error creating conversation summary: {e}")
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Simple keyword extraction for topics."""
        medical_keywords = [
            "headache", "fever", "pain", "medication", "symptoms", "diagnosis",
            "treatment", "doctor", "hospital", "blood pressure", "diabetes",
            "heart", "lung", "stomach", "back", "joint", "muscle", "skin",
            "infection", "allergy", "prescription", "dosage", "side effects"
        ]
        
        text_lower = text.lower()
        found_topics = [keyword for keyword in medical_keywords if keyword in text_lower]
        return list(set(found_topics))