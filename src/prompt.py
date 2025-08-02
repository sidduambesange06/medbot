"""
Enhanced Medical Prompt System - Production Ready with Context Awareness
Advanced prompt engineering for medical queries with context-aware response generation
"""

import re
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def get_production_medical_prompt(medical_content: str, sources: list, query: str, context_info: Dict = None) -> str:
    """
    Enhanced production-optimized system prompt for medical queries with context awareness
    Generates concise, professional responses suitable for clinical use with conversation context
    """
    
    context_section = ""
    if context_info and context_info.get('has_context'):
        context_section = f"""
CONVERSATION CONTEXT:
{context_info.get('context_summary', 'Previous conversation context available')}

CONTEXT INTEGRATION:
- Build upon previous discussions when relevant
- Reference earlier topics if they relate to current query
- Maintain conversation continuity
- Acknowledge user's ongoing medical concerns
"""

    return f"""You are a Medical AI Assistant with access to medical textbooks and general medical knowledge, enhanced with conversation context awareness.

PATIENT QUERY: {query}

TEXTBOOK INFORMATION AVAILABLE:
{medical_content}
{context_section}

ENHANCED INSTRUCTIONS:
1. PRIMARY: Use the textbook information above as your main source
2. SUPPLEMENT: If textbook info is incomplete, add essential medical knowledge
3. CONTEXT INTEGRATION: Consider previous conversation when relevant
4. CLEARLY DISTINGUISH: Mark textbook info vs. general medical knowledge
5. MAINTAIN QUALITY: Provide comprehensive yet concise medical response

RESPONSE STRUCTURE:
📚 **From Medical Textbooks:** [Use textbook content with citations]
🏥 **Additional Medical Information:** [Add essential medical knowledge if needed]
🔄 **Context Integration:** [Connect with previous discussions if relevant]

ENHANCED GUIDELINES:
• Keep response focused and clinically relevant (200-400 words)
• Prioritize textbook information when available
• Add general medical knowledge only to complete the answer
• Use professional medical terminology
• Organize information logically
• Reference previous conversation when applicable
• Build upon established medical context

End with: "⚠️ Information combines medical textbook evidence with established medical knowledge. Always consult healthcare professionals for medical advice."
"""

def get_medical_no_textbook_prompt(query: str, context_info: Dict = None) -> str:
    """
    Enhanced prompt for medical queries when no textbook information is available
    Provides helpful medical guidance while emphasizing the need for professional consultation
    """
    
    context_awareness = ""
    if context_info and context_info.get('has_context'):
        context_awareness = f"""
CONVERSATION CONTEXT: Building on our previous medical discussions about {context_info.get('previous_topics', 'your health concerns')}.
"""

    return f"""You are a Medical AI Assistant with conversation context awareness. The user asked about: "{query}"

SITUATION: No specific information found in the medical textbook database.
{context_awareness}

YOUR ENHANCED RESPONSE SHOULD:
1. Briefly acknowledge the lack of textbook information
2. Reference previous conversation context if relevant
3. Provide general medical guidance (3-4 sentences maximum)
4. Emphasize the importance of professional medical consultation
5. Suggest appropriate healthcare resources
6. Maintain conversation continuity

RESPONSE FORMAT:
"I couldn't find specific information about '{query}' in our medical textbook database.

{context_awareness}

[2-3 sentences of general medical guidance, considering previous context]

For accurate, personalized medical information about {query}, I strongly recommend:
• Consulting with your healthcare provider
• Speaking with a medical specialist if needed
• Referring to current medical literature
• Following up on previous medical discussions we've had

⚠️ For accurate medical information, please consult qualified healthcare professionals."

Keep the response helpful but conservative, acknowledging limitations while providing contextual value.
"""

def get_general_knowledge_prompt(context_info: Dict = None) -> str:
    """
    Enhanced prompt for non-medical queries with conversation awareness
    Provides helpful, focused responses using general AI knowledge
    """
    
    context_section = ""
    if context_info and context_info.get('has_context'):
        context_section = """
CONVERSATION CONTEXT: Consider our previous discussion when providing your response.

"""

    return f"""You are a helpful AI assistant responding to a general (non-medical) question with conversation awareness.

{context_section}RESPONSE GUIDELINES:
• Provide clear, accurate, and helpful information
• Keep responses focused and well-organized
• Use appropriate tone for the question type
• Include relevant examples or context when helpful
• Be conversational yet informative
• Reference previous conversation when relevant
• Maintain conversation continuity

Since this is not a medical query, you can use your full knowledge base to provide a comprehensive answer. No medical disclaimers are needed unless the topic touches on health matters.
"""

def get_greeting_introduction_prompt(context_info: Dict = None) -> str:
    """
    Enhanced specialized prompt for greetings and introduction queries with context awareness
    """
    
    returning_user = context_info and context_info.get('is_returning_user', False)
    
    if returning_user:
        return """You are a specialized Medical AI Assistant welcoming back a returning user.

RETURNING USER GREETING:
🏥 **Welcome Back!** I remember our previous medical discussions and I'm here to continue helping with your health questions.

CONTEXTUAL INTRODUCTION:
📚 **Continuous Support:** I provide evidence-based medical information from authoritative medical textbooks
🔄 **Conversation Memory:** I can reference our previous discussions to provide better context
🔍 **How I Work:** I search medical literature and consider our conversation history for relevant information
📝 **Enhanced Citations:** All medical responses include textbook sources and page references
🎯 **Personalized Care:** I build upon our previous conversations for more relevant responses
⚠️ **Important:** I provide educational information - always consult healthcare professionals for medical advice

RESPONSE STYLE:
• Warm and welcoming tone acknowledging previous interactions
• Reference ability to continue previous medical discussions
• Mention enhanced conversation continuity
• Keep welcome concise (3-4 sentences)
• Invite continuation of medical conversation

Example: "Welcome back! I remember our previous discussions about [health topics] and I'm ready to continue helping with your medical questions. What would you like to explore today?"
"""
    
    else:
        return """You are a specialized Medical AI Assistant powered by medical textbook knowledge and conversation awareness.

INTRODUCE YOURSELF WITH:
🏥 **Purpose:** I provide evidence-based medical information from authoritative medical textbooks
📚 **Knowledge Source:** Medical textbooks stored in a specialized database
🔍 **How I Work:** I search medical literature to find relevant information for your health questions
🔄 **Conversation Memory:** I remember our discussions to provide better context in future responses
📝 **Citations:** All medical responses include textbook sources and page references
🎯 **Personalized Experience:** I build context over time for more relevant medical guidance
⚠️ **Important:** I provide educational information - always consult healthcare professionals for medical advice

RESPONSE STYLE:
• Professional yet approachable
• Briefly explain your capabilities including context awareness
• Mention the medical textbook database
• Highlight conversation continuity features
• Keep introduction concise (3-4 sentences)
• Invite medical questions

Example: "Hello! I'm a Medical AI Assistant that provides evidence-based medical information from authoritative medical textbooks. I remember our conversations to provide better context and continuity. What medical topic would you like to learn about today?"
"""

def optimize_medical_content_extraction(chunks: list, query: str, context_info: Dict = None, max_sentences: int = 8) -> str:
    """
    Enhanced extract and optimize medical content for concise responses with context awareness
    Focus on most relevant information while maintaining medical accuracy and context continuity
    """
    
    query_terms = set(query.lower().split())
    
    # Add context terms if available
    if context_info and context_info.get('previous_topics'):
        context_terms = set()
        for topic in context_info['previous_topics']:
            context_terms.update(topic.lower().split())
        query_terms.update(context_terms)
    
    optimized_sections = []
    
    for chunk in chunks[:3]:  # Focus on top 3 most relevant chunks
        content = chunk.get('content', '')
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        # Score sentences by relevance to query and context
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Base relevance to current query
            relevance_score = sum(1 for term in query_terms if term in sentence_lower)
            
            # Boost score for medical keywords
            medical_boost = sum(1 for keyword in ['treatment', 'symptom', 'cause', 'diagnosis', 'medication', 'therapy', 'condition'] 
                            if keyword in sentence_lower)
            
            # Context boost for returning users
            context_boost = 0
            if context_info and context_info.get('previous_topics'):
                for topic in context_info['previous_topics']:
                    if any(word in sentence_lower for word in topic.lower().split()):
                        context_boost += 0.5
            
            total_score = relevance_score + medical_boost + context_boost
            scored_sentences.append((sentence, total_score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = [s[0] for s in scored_sentences[:3] if s[1] > 0]
        
        if selected_sentences:
            section = '. '.join(selected_sentences)
            if not section.endswith('.'):
                section += '.'
            
            # Add citation
            book_name = chunk.get('book_name', 'Medical Textbook')
            page = chunk.get('page', 'N/A')
            section += f" [{book_name}, p.{page}]"
            
            optimized_sections.append(section)
    
    return '\n\n'.join(optimized_sections)

def format_medical_sources_concisely(sources: list, max_sources: int = 3) -> str:
    """
    Enhanced format medical sources in a concise, professional manner
    """
    
    if not sources:
        return ""
    
    formatted_sources = []
    for source in sources[:max_sources]:
        book_name = source.get('book_name', 'Medical Textbook')
        page = source.get('page', 'N/A')
        formatted_sources.append(f"{book_name} (p.{page})")
    
    return "📚 **Sources:** " + ", ".join(formatted_sources)

def create_context_summary(recent_messages: List[Dict], max_length: int = 200) -> str:
    """
    Create a concise summary of recent conversation context
    """
    if not recent_messages:
        return ""
    
    # Extract key topics and medical terms from recent messages
    medical_terms = []
    topics = []
    
    for msg in recent_messages[-5:]:  # Last 5 messages
        content = msg.get('message', '').lower()
        
        # Extract medical keywords
        medical_keywords = ['symptom', 'treatment', 'diagnosis', 'medication', 'condition', 'disease', 'therapy', 'pain', 'health']
        found_terms = [term for term in medical_keywords if term in content]
        medical_terms.extend(found_terms)
        
        # Extract potential topics (nouns)
        words = content.split()
        potential_topics = [word for word in words if len(word) > 4 and not word.startswith(('what', 'how', 'why', 'when', 'where'))]
        topics.extend(potential_topics[:3])  # Top 3 topics per message
    
    # Create summary
    unique_terms = list(set(medical_terms))
    unique_topics = list(set(topics))
    
    if unique_terms or unique_topics:
        summary_parts = []
        if unique_terms:
            summary_parts.append(f"Medical focus: {', '.join(unique_terms[:3])}")
        if unique_topics:
            summary_parts.append(f"Topics discussed: {', '.join(unique_topics[:3])}")
        
        summary = "Previous conversation covered: " + "; ".join(summary_parts)
        return summary[:max_length] + "..." if len(summary) > max_length else summary
    
    return "Continuing medical discussion"

def create_medical_response_template(query_type: str, **kwargs) -> str:
    """
    Enhanced main function to create appropriate response templates with context awareness
    """
    
    context_info = kwargs.get('context_info', {})
    
    if query_type == "medical_with_textbook":
        return get_production_medical_prompt(
            kwargs.get('medical_content', ''),
            kwargs.get('sources', []),
            kwargs.get('query', ''),
            context_info
        )
    
    elif query_type == "medical_hybrid":
        return get_medical_hybrid_prompt(
            kwargs.get('medical_content', ''),
            kwargs.get('sources', []),
            kwargs.get('query', ''),
            context_info
        )
    
    elif query_type == "medical_no_textbook":
        return get_medical_no_textbook_prompt(
            kwargs.get('query', ''),
            context_info
        )
    
    elif query_type == "general":
        return get_general_knowledge_prompt(context_info)
    
    elif query_type == "greeting":
        return get_greeting_introduction_prompt(context_info)
    
    else:
        return get_general_knowledge_prompt(context_info)

def validate_medical_response_quality(response: str, query: str, context_info: Dict = None) -> dict:
    """
    Enhanced quality validation for medical responses with context awareness
    Ensures responses meet production standards and proper context integration
    """
    
    quality_metrics = {
        'length_appropriate': 100 <= len(response) <= 700,  # Slightly increased for context
        'has_medical_disclaimer': '⚠️' in response and 'healthcare professionals' in response.lower(),
        'addresses_query': any(term in response.lower() for term in query.lower().split()),
        'professional_tone': not any(word in response.lower() for word in ['awesome', 'cool', 'wow', 'amazing']),
        'has_citations': '[' in response and ']' in response,
        'concise': response.count('\n\n') <= 5,  # Slightly more flexible for context
        'no_repetition': len(set(response.split('. '))) > len(response.split('. ')) * 0.75,
        'context_integration': True  # Default to true, enhanced below
    }
    
    # Enhanced context integration check
    if context_info and context_info.get('has_context'):
        context_integration_indicators = [
            'previous', 'earlier', 'discussed', 'mentioned', 'continue', 'building',
            'following up', 'as we talked', 'from our conversation'
        ]
        quality_metrics['context_integration'] = any(
            indicator in response.lower() for indicator in context_integration_indicators
        )
    
    # Medical terminology check
    medical_terms = [
        'diagnosis', 'treatment', 'symptoms', 'condition', 'medication', 'therapy',
        'clinical', 'medical', 'health', 'patient', 'disease'
    ]
    medical_term_count = sum(1 for term in medical_terms if term in response.lower())
    quality_metrics['adequate_medical_terminology'] = medical_term_count >= 2
    
    quality_score = sum(quality_metrics.values()) / len(quality_metrics) * 100
    
    return {
        'quality_score': round(quality_score, 1),
        'metrics': quality_metrics,
        'recommendation': 'approved' if quality_score >= 75 else 'needs_improvement',  # Adjusted threshold
        'word_count': len(response.split()),
        'medical_term_count': medical_term_count,
        'context_aware': quality_metrics['context_integration']
    }

def enhance_medical_keywords_detection() -> list:
    """
    Enhanced medical keywords for better query classification with context categories
    Focused on common medical queries and symptoms with categorical organization
    """
    
    return {
        'core_medical': [
            'medical', 'health', 'healthcare', 'clinical', 'diagnosis', 'treatment',
            'therapy', 'cure', 'healing', 'recovery', 'medicine', 'medication',
            'drug', 'pharmaceutical', 'prescription', 'dosage', 'side effect'
        ],
        'diseases_conditions': [
            'diabetes', 'hypertension', 'cancer', 'tumor', 'heart disease', 'stroke',
            'asthma', 'copd', 'pneumonia', 'bronchitis', 'tuberculosis', 'covid',
            'flu', 'influenza', 'cold', 'fever', 'infection', 'virus', 'bacteria',
            'arthritis', 'osteoporosis', 'alzheimer', 'parkinson', 'epilepsy',
            'depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd',
            'hepatitis', 'cirrhosis', 'kidney disease', 'liver disease',
            'anemia', 'leukemia', 'lymphoma', 'migraine', 'headache'
        ],
        'symptoms': [
            'pain', 'ache', 'hurt', 'discomfort', 'soreness', 'burning',
            'headache', 'migraine', 'dizziness', 'vertigo', 'nausea',
            'vomiting', 'diarrhea', 'constipation', 'bloating', 'cramps',
            'cough', 'wheeze', 'shortness of breath', 'chest pain',
            'back pain', 'neck pain', 'joint pain', 'muscle pain',
            'fatigue', 'weakness', 'tiredness', 'exhaustion',
            'fever', 'chills', 'sweating', 'rash', 'itching', 'swelling',
            'bleeding', 'bruising', 'numbness', 'tingling',
            'insomnia', 'sleep disorder', 'appetite loss', 'weight loss'
        ],
        'anatomy': [
            'heart', 'cardiac', 'blood pressure', 'circulation',
            'lung', 'respiratory', 'breathing', 'airway',
            'brain', 'neurological', 'nervous system', 'spine',
            'liver', 'kidney', 'stomach', 'intestine', 'digestive',
            'blood', 'bone', 'muscle', 'skin', 'eye', 'ear'
        ],
        'procedures_tests': [
            'surgery', 'operation', 'biopsy', 'x-ray', 'ct scan',
            'mri', 'ultrasound', 'blood test', 'urine test',
            'vaccination', 'vaccine', 'immunization',
            'physical exam', 'checkup', 'screening'
        ],
        'question_patterns': [
            'what is', 'what are', 'what causes', 'how to treat',
            'symptoms of', 'signs of', 'treatment for', 'cure for',
            'medicine for', 'prevention of', 'risk factors'
        ]
    }

def create_smart_medical_classifier(query: str, context_info: Dict = None) -> dict:
    """
    Enhanced smart medical query classifier with confidence scoring and context awareness
    """
    
    query_lower = query.lower()
    medical_keywords = enhance_medical_keywords_detection()
    
    # Flatten keywords for matching
    all_keywords = []
    category_matches = {}
    
    for category, keywords in medical_keywords.items():
        category_matches[category] = []
        for keyword in keywords:
            all_keywords.append(keyword)
            if keyword in query_lower:
                category_matches[category].append(keyword)
    
    # Keyword matching with category weights
    keyword_score = 0
    for category, matches in category_matches.items():
        if matches:
            # Different weights for different categories
            weights = {
                'core_medical': 3,
                'diseases_conditions': 4,
                'symptoms': 3,
                'anatomy': 2,
                'procedures_tests': 2,
                'question_patterns': 1
            }
            keyword_score += len(matches) * weights.get(category, 1)
    
    # Pattern matching
    medical_patterns = [
        r'\b(what|how|why)\s+.*\b(disease|symptom|treatment|medicine|cure)\b',
        r'\b(symptoms?|treatment|cure|medicine)\s+(of|for)\b',
        r'\b(pain|ache|hurt)\s+(in|of)\b',
        r'\b(how\s+to\s+(treat|cure|prevent))\b',
        r'\b(side\s+effects?|dosage)\b'
    ]
    
    pattern_matches = sum(1 for pattern in medical_patterns 
                        if re.search(pattern, query_lower))
    
    # Context boost for returning users with medical history
    context_boost = 0
    if context_info and context_info.get('has_medical_history'):
        context_boost = 5  # Boost for users with medical conversation history
    
    # Calculate confidence
    total_score = keyword_score + (pattern_matches * 3) + context_boost
    confidence = min(100, total_score * 8)  # Adjusted multiplier
    
    is_medical = confidence >= 25  # Adjusted threshold for context-aware classification
    
    return {
        'is_medical': is_medical,
        'confidence': confidence,
        'category_matches': {k: v for k, v in category_matches.items() if v},
        'pattern_matches': pattern_matches,
        'context_boost': context_boost,
        'classification': 'medical' if is_medical else 'general',
        'primary_categories': [cat for cat, matches in category_matches.items() if matches][:3]
    }

def generate_response_summary(response: str, context_aware: bool = False) -> dict:
    """
    Enhanced generate summary statistics for response optimization with context awareness
    """
    
    words = response.split()
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    
    # Enhanced metrics for context-aware responses
    summary = {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'has_citations': '[' in response and ']' in response,
        'has_disclaimer': '⚠️' in response,
        'medical_terms_count': sum(1 for word in words 
                                if word.lower() in [kw for cat in enhance_medical_keywords_detection().values() for kw in cat]),
        'readability_score': 100 - (len(words) / len(sentences) * 2) if sentences else 0,
        'context_aware': context_aware
    }
    
    # Context-specific metrics
    if context_aware:
        context_indicators = ['previous', 'earlier', 'discussed', 'mentioned', 'continue', 'building']
        summary['context_integration_score'] = sum(1 for indicator in context_indicators 
                                                if indicator in response.lower())
        summary['context_integration_percentage'] = (summary['context_integration_score'] / len(context_indicators)) * 100
    
    return summary

# Example usage and testing functions
def test_enhanced_prompt_optimization():
    """
    Enhanced test function to validate prompt optimization with context awareness
    """
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "How to treat high blood pressure?",
        "What causes heart attack?",
        "Side effects of aspirin",
        "Hello, how are you?",
        "What's the weather today?"
    ]
    
    # Mock context info for testing
    mock_context = {
        'has_context': True,
        'has_medical_history': True,
        'previous_topics': ['diabetes', 'blood pressure'],
        'is_returning_user': True
    }
    
    results = {}
    
    for query in test_queries:
        classification = create_smart_medical_classifier(query, mock_context)
        
        if classification['is_medical']:
            prompt_type = "medical_with_textbook"
            prompt = get_production_medical_prompt("Sample medical content", [], query, mock_context)
        else:
            prompt_type = "general"
            prompt = get_general_knowledge_prompt(mock_context)
        
        results[query] = {
            'classification': classification,
            'prompt_type': prompt_type,
            'prompt_length': len(prompt.split()),
            'context_aware': True
        }
    
    return results

def optimize_for_enhanced_production():
    """
    Enhanced production optimization settings and recommendations with context features
    """
    
    return {
        'response_settings': {
            'max_tokens': 700,  # Increased for context integration
            'temperature': 0.1,
            'top_p': 0.9,
            'target_word_count': '150-350 words',  # Adjusted for context
            'max_sources_displayed': 3,
            'context_integration': True
        },
        'quality_thresholds': {
            'minimum_quality_score': 75,  # Adjusted for context features
            'maximum_response_length': 700,
            'minimum_response_length': 100,
            'medical_confidence_threshold': 25,  # Adjusted for context boost
            'context_integration_threshold': 60
        },
        'context_features': {
            'max_context_messages': 10,
            'context_summary_length': 200,
            'semantic_similarity_threshold': 0.7,
            'conversation_memory_days': 30
        },
        'performance_targets': {
            'response_time': '<2.5 seconds',  # Slightly adjusted for context processing
            'accuracy': '>90%',
            'user_satisfaction': '>85%',
            'context_relevance': '>80%'
        },
        'monitoring_metrics': [
            'average_response_length',
            'medical_classification_accuracy',
            'citation_inclusion_rate',
            'disclaimer_compliance',
            'response_quality_score',
            'context_integration_success',
            'conversation_continuity_score',
            'user_context_satisfaction'
        ]
    }

# Main enhanced prompt selection function
def select_optimal_prompt(query: str, has_textbook_knowledge: bool = False, 
                        medical_content: str = "", sources: list = [], 
                        context_info: Dict = None) -> str:
    """
    Enhanced intelligently select the optimal prompt based on query, available knowledge, and context
    """
    
    classification = create_smart_medical_classifier(query, context_info)
    
    if classification['is_medical']:
        if has_textbook_knowledge and medical_content:
            # Full medical response with textbook evidence and context
            return get_production_medical_prompt(medical_content, sources, query, context_info)
        elif has_textbook_knowledge and len(medical_content) < 200:
            # Hybrid response - limited textbook + LLM knowledge + context
            return get_medical_hybrid_prompt(medical_content, sources, query, context_info)
        else:
            # No textbook knowledge available but maintain context
            return get_medical_no_textbook_prompt(query, context_info)
    else:
        # Check if it's a greeting/introduction
        greeting_keywords = ['hello', 'hi', 'hey', 'who are you', 'what are you', 'introduce']
        if any(keyword in query.lower() for keyword in greeting_keywords):
            return get_greeting_introduction_prompt(context_info)
        else:
            return get_general_knowledge_prompt(context_info)

# Export key functions for main application
__all__ = [
    'get_production_medical_prompt',
    'get_medical_hybrid_prompt', 
    'get_medical_no_textbook_prompt',
    'get_general_knowledge_prompt',
    'get_greeting_introduction_prompt',
    'create_smart_medical_classifier',
    'select_optimal_prompt',
    'validate_medical_response_quality',
    'optimize_medical_content_extraction',
    'format_medical_sources_concisely',
    'create_context_summary',
    'create_medical_response_template',
    'enhance_medical_keywords_detection',
    'generate_response_summary',
    'test_enhanced_prompt_optimization',
    'optimize_for_enhanced_production'
]

def get_medical_hybrid_prompt(medical_content: str, sources: list, query: str, context_info: Dict = None) -> str:
    """
    Enhanced hybrid prompt combining textbook knowledge with LLM medical knowledge
    Used when textbook information is limited but medical expertise is needed
    """
    
    context_section = ""
    if context_info and context_info.get('has_context'):
        context_section = f"""
CONVERSATION HISTORY:
{context_info.get('context_summary', 'Building on previous medical discussions')}
"""

    return f"""You are a Professional Medical AI Assistant providing evidence-based medical information from authoritative medical textbooks with conversation context awareness.

CURRENT QUERY: {query}

MEDICAL TEXTBOOK EVIDENCE:
{medical_content}
{context_section}

ENHANCED RESPONSE GUIDELINES:
1. CONTEXTUAL AWARENESS: Consider previous conversation when relevant
2. CONCISE & PROFESSIONAL: 150-300 words maximum
3. DIRECT ANSWER: Address the user's specific question immediately
4. CLINICAL ACCURACY: Use precise medical terminology with brief explanations
5. LOGICAL STRUCTURE: Definition → Key Facts → Clinical Significance → Context Integration
6. EVIDENCE-BASED: Use only the textbook information provided above
7. CONVERSATION CONTINUITY: Reference previous discussions when applicable

RESPONSE FORMAT:
1. Direct answer to the current query (2-3 sentences)
2. Key medical facts organized by importance
3. Integration with previous conversation context (if relevant)
4. Essential clinical information only
5. Professional medical tone throughout

CONTEXT INTEGRATION RULES:
- If user asked about related topics before, acknowledge the connection
- Build upon previously discussed symptoms, conditions, or treatments
- Maintain conversation flow while providing new information
- Reference user's ongoing medical journey when appropriate

AVOID:
❌ Lengthy explanations or excessive detail
❌ Repetitive information from previous responses
❌ Adding information not in the textbook content
❌ Multiple disclaimers (one at the end is sufficient)
❌ Ignoring conversation context when relevant

The citations are already included in the content - focus on delivering clear, contextual, actionable medical information.

Always end with: "⚠️ This information is from medical textbooks for educational purposes. Always consult healthcare professionals for medical advice."
"""