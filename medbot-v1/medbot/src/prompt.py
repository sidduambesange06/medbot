"""
Optimized Medical Prompt System - Production Ready
Concise, professional medical responses with focused output generation
"""

import re

def get_production_medical_prompt(medical_content: str, sources: list, query: str) -> str:
    """
    Production-optimized system prompt for medical queries
    Generates concise, professional responses suitable for clinical use
    """
    
    return f"""You are a Professional Medical AI Assistant providing evidence-based medical information from authoritative medical textbooks.

PATIENT QUERY: {query}

MEDICAL TEXTBOOK EVIDENCE:
{medical_content}

RESPONSE REQUIREMENTS:
✅ CONCISE & PROFESSIONAL: 150-300 words maximum
✅ DIRECT ANSWER: Address the user's specific question immediately
✅ CLINICAL ACCURACY: Use precise medical terminology with brief explanations
✅ LOGICAL STRUCTURE: Definition → Key Facts → Clinical Significance
✅ EVIDENCE-BASED: Use only the textbook information provided above

RESPONSE FORMAT:
1. Direct answer to the query (2-3 sentences)
2. Key medical facts organized by importance
3. Essential clinical information only
4. Professional medical tone throughout

AVOID:
❌ Lengthy explanations or excessive detail
❌ Repetitive information
❌ Adding information not in the textbook content
❌ Multiple disclaimers (one at the end is sufficient)

The citations are already included in the content - focus on delivering clear, actionable medical information.

End with: "⚠️ This information is from medical textbooks for educational purposes. Always consult healthcare professionals for medical advice."
"""

def get_medical_hybrid_prompt(medical_content: str, sources: list, query: str) -> str:
    """
    Hybrid prompt combining textbook knowledge with LLM medical knowledge
    Used when textbook information is limited but medical expertise is needed
    """
    
    return f"""You are a Medical AI Assistant with access to medical textbooks and general medical knowledge.

PATIENT QUERY: {query}

TEXTBOOK INFORMATION AVAILABLE:
{medical_content}

INSTRUCTIONS:
1. PRIMARY: Use the textbook information above as your main source
2. SUPPLEMENT: If textbook info is incomplete, add essential medical knowledge
3. CLEARLY DISTINGUISH: Mark textbook info vs. general medical knowledge
4. MAINTAIN QUALITY: Provide comprehensive yet concise medical response

RESPONSE STRUCTURE:
📚 **From Medical Textbooks:** [Use textbook content with citations]
🏥 **Additional Medical Information:** [Add essential medical knowledge if needed]

GUIDELINES:
• Keep response focused and clinically relevant (200-400 words)
• Prioritize textbook information when available
• Add general medical knowledge only to complete the answer
• Use professional medical terminology
• Organize information logically

End with: "⚠️ Information combines medical textbook evidence with established medical knowledge. Always consult healthcare professionals for medical advice."
"""

def get_medical_no_textbook_prompt(query: str) -> str:
    """
    Prompt for medical queries when no textbook information is available
    Provides helpful medical guidance while emphasizing the need for professional consultation
    """
    
    return f"""You are a Medical AI Assistant. The user asked about: "{query}"

SITUATION: No specific information found in the medical textbook database.

YOUR RESPONSE SHOULD:
1. Briefly acknowledge the lack of textbook information
2. Provide general medical guidance (3-4 sentences maximum)
3. Emphasize the importance of professional medical consultation
4. Suggest appropriate healthcare resources

RESPONSE FORMAT:
"I couldn't find specific information about '{query}' in our medical textbook database.

[2-3 sentences of general medical guidance]

For accurate, personalized medical information about {query}, I strongly recommend:
• Consulting with your healthcare provider
• Speaking with a medical specialist if needed
• Referring to current medical literature

⚠️ For accurate medical information, please consult qualified healthcare professionals."

Keep the response helpful but conservative, acknowledging the limitations while providing value.
"""

def get_general_knowledge_prompt() -> str:
    """
    Optimized prompt for non-medical queries
    Provides helpful, focused responses using general AI knowledge
    """
    
    return """You are a helpful AI assistant responding to a general (non-medical) question.

RESPONSE GUIDELINES:
• Provide clear, accurate, and helpful information
• Keep responses focused and well-organized
• Use appropriate tone for the question type
• Include relevant examples or context when helpful
• Be conversational yet informative

Since this is not a medical query, you can use your full knowledge base to provide a comprehensive answer. No medical disclaimers are needed.
"""

def get_greeting_introduction_prompt() -> str:
    """
    Specialized prompt for greetings and introduction queries
    """
    
    return """You are a specialized Medical AI Assistant powered by medical textbook knowledge.

INTRODUCE YOURSELF WITH:
🏥 **Purpose:** I provide evidence-based medical information from authoritative medical textbooks
📚 **Knowledge Source:** Medical textbooks stored in a specialized database
🔍 **How I Work:** I search medical literature to find relevant information for your health questions
📝 **Citations:** All medical responses include textbook sources and page references
⚠️ **Important:** I provide educational information - always consult healthcare professionals for medical advice

RESPONSE STYLE:
• Professional yet approachable
• Briefly explain your capabilities
• Mention the medical textbook database
• Keep introduction concise (3-4 sentences)
• Invite medical questions

Example: "Hello! I'm a Medical AI Assistant that provides evidence-based medical information from authoritative medical textbooks. I can help answer your health and medical questions using verified medical literature, complete with citations. What medical topic would you like to learn about today?"
"""

def optimize_medical_content_extraction(chunks: list, query: str, max_sentences: int = 8) -> str:
    """
    Extract and optimize medical content for concise responses
    Focus on most relevant information while maintaining medical accuracy
    """
    
    query_terms = set(query.lower().split())
    optimized_sections = []
    
    for chunk in chunks[:3]:  # Focus on top 3 most relevant chunks
        content = chunk.get('content', '')
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        # Score sentences by relevance to query
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            relevance_score = sum(1 for term in query_terms if term in sentence_lower)
            
            # Boost score for medical keywords
            medical_boost = sum(1 for keyword in ['treatment', 'symptom', 'cause', 'diagnosis', 'medication'] 
                              if keyword in sentence_lower)
            
            scored_sentences.append((sentence, relevance_score + medical_boost))
        
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
    Format medical sources in a concise, professional manner
    """
    
    if not sources:
        return ""
    
    formatted_sources = []
    for source in sources[:max_sources]:
        book_name = source.get('book_name', 'Medical Textbook')
        page = source.get('page', 'N/A')
        formatted_sources.append(f"{book_name} (p.{page})")
    
    return "📚 **Sources:** " + ", ".join(formatted_sources)

def create_medical_response_template(query_type: str, **kwargs) -> str:
    """
    Main function to create appropriate response templates
    """
    
    if query_type == "medical_with_textbook":
        return get_production_medical_prompt(
            kwargs.get('medical_content', ''),
            kwargs.get('sources', []),
            kwargs.get('query', '')
        )
    
    elif query_type == "medical_hybrid":
        return get_medical_hybrid_prompt(
            kwargs.get('medical_content', ''),
            kwargs.get('sources', []),
            kwargs.get('query', '')
        )
    
    elif query_type == "medical_no_textbook":
        return get_medical_no_textbook_prompt(kwargs.get('query', ''))
    
    elif query_type == "general":
        return get_general_knowledge_prompt()
    
    elif query_type == "greeting":
        return get_greeting_introduction_prompt()
    
    else:
        return get_general_knowledge_prompt()

def validate_medical_response_quality(response: str, query: str) -> dict:
    """
    Quality validation for medical responses
    Ensures responses meet production standards
    """
    
    quality_metrics = {
        'length_appropriate': 100 <= len(response) <= 600,
        'has_medical_disclaimer': '⚠️' in response and 'healthcare professionals' in response.lower(),
        'addresses_query': any(term in response.lower() for term in query.lower().split()),
        'professional_tone': not any(word in response.lower() for word in ['awesome', 'cool', 'wow']),
        'has_citations': '[' in response and ']' in response,
        'concise': response.count('\n\n') <= 4,  # Not too many paragraph breaks
        'no_repetition': len(set(response.split('. '))) > len(response.split('. ')) * 0.8
    }
    
    quality_score = sum(quality_metrics.values()) / len(quality_metrics) * 100
    
    return {
        'quality_score': round(quality_score, 1),
        'metrics': quality_metrics,
        'recommendation': 'approved' if quality_score >= 80 else 'needs_improvement',
        'word_count': len(response.split())
    }

def enhance_medical_keywords_detection() -> list:
    """
    Enhanced medical keywords for better query classification
    Focused on common medical queries and symptoms
    """
    
    return [
        # Core Medical Terms
        'medical', 'health', 'healthcare', 'clinical', 'diagnosis', 'treatment',
        'therapy', 'cure', 'healing', 'recovery', 'medicine', 'medication',
        'drug', 'pharmaceutical', 'prescription', 'dosage', 'side effect',
        
        # Diseases & Conditions (Most Common)
        'diabetes', 'hypertension', 'cancer', 'tumor', 'heart disease', 'stroke',
        'asthma', 'copd', 'pneumonia', 'bronchitis', 'tuberculosis', 'covid',
        'flu', 'influenza', 'cold', 'fever', 'infection', 'virus', 'bacteria',
        'arthritis', 'osteoporosis', 'alzheimer', 'parkinson', 'epilepsy',
        'depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd',
        'hepatitis', 'cirrhosis', 'kidney disease', 'liver disease',
        'anemia', 'leukemia', 'lymphoma', 'migraine', 'headache',
        
        # Symptoms (Most Searched)
        'pain', 'ache', 'hurt', 'discomfort', 'soreness', 'burning',
        'headache', 'migraine', 'dizziness', 'vertigo', 'nausea',
        'vomiting', 'diarrhea', 'constipation', 'bloating', 'cramps',
        'cough', 'wheeze', 'shortness of breath', 'chest pain',
        'back pain', 'neck pain', 'joint pain', 'muscle pain',
        'fatigue', 'weakness', 'tiredness', 'exhaustion',
        'fever', 'chills', 'sweating', 'rash', 'itching', 'swelling',
        'bleeding', 'bruising', 'numbness', 'tingling',
        'insomnia', 'sleep disorder', 'appetite loss', 'weight loss',
        
        # Body Systems & Anatomy
        'heart', 'cardiac', 'blood pressure', 'circulation',
        'lung', 'respiratory', 'breathing', 'airway',
        'brain', 'neurological', 'nervous system', 'spine',
        'liver', 'kidney', 'stomach', 'intestine', 'digestive',
        'blood', 'bone', 'muscle', 'skin', 'eye', 'ear',
        
        # Medical Procedures & Tests
        'surgery', 'operation', 'biopsy', 'x-ray', 'ct scan',
        'mri', 'ultrasound', 'blood test', 'urine test',
        'vaccination', 'vaccine', 'immunization',
        'physical exam', 'checkup', 'screening',
        
        # Question Patterns
        'what is', 'what are', 'what causes', 'how to treat',
        'symptoms of', 'signs of', 'treatment for', 'cure for',
        'medicine for', 'prevention of', 'risk factors'
    ]

def create_smart_medical_classifier(query: str) -> dict:
    """
    Smart medical query classifier with confidence scoring
    """
    
    query_lower = query.lower()
    medical_keywords = enhance_medical_keywords_detection()
    
    # Keyword matching with weights
    keyword_matches = []
    for keyword in medical_keywords:
        if keyword in query_lower:
            # Weight based on keyword importance
            weight = 3 if keyword in ['diabetes', 'cancer', 'heart', 'pain'] else 1
            keyword_matches.append((keyword, weight))
    
    keyword_score = sum(weight for _, weight in keyword_matches)
    
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
    
    # Calculate confidence
    total_score = keyword_score + (pattern_matches * 2)
    confidence = min(100, total_score * 10)  # Cap at 100%
    
    is_medical = confidence >= 30  # Threshold for medical classification
    
    return {
        'is_medical': is_medical,
        'confidence': confidence,
        'keyword_matches': [kw for kw, _ in keyword_matches],
        'pattern_matches': pattern_matches,
        'classification': 'medical' if is_medical else 'general'
    }

def generate_response_summary(response: str) -> dict:
    """
    Generate summary statistics for response optimization
    """
    
    words = response.split()
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'has_citations': '[' in response and ']' in response,
        'has_disclaimer': '⚠️' in response,
        'medical_terms_count': sum(1 for word in words 
                                 if word.lower() in enhance_medical_keywords_detection()),
        'readability_score': 100 - (len(words) / len(sentences) * 2) if sentences else 0
    }

# Example usage and testing functions
def test_prompt_optimization():
    """
    Test function to validate prompt optimization
    """
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "How to treat high blood pressure?",
        "What causes heart attack?",
        "Side effects of aspirin",
        "Hello, how are you?",
        "What's the weather today?"
    ]
    
    results = {}
    
    for query in test_queries:
        classification = create_smart_medical_classifier(query)
        
        if classification['is_medical']:
            prompt_type = "medical_with_textbook"
            prompt = get_production_medical_prompt("Sample medical content", [], query)
        else:
            prompt_type = "general"
            prompt = get_general_knowledge_prompt()
        
        results[query] = {
            'classification': classification,
            'prompt_type': prompt_type,
            'prompt_length': len(prompt.split())
        }
    
    return results

def optimize_for_production():
    """
    Production optimization settings and recommendations
    """
    
    return {
        'response_settings': {
            'max_tokens': 600,
            'temperature': 0.1,
            'top_p': 0.9,
            'target_word_count': '150-300 words',
            'max_sources_displayed': 3
        },
        'quality_thresholds': {
            'minimum_quality_score': 80,
            'maximum_response_length': 600,
            'minimum_response_length': 100,
            'medical_confidence_threshold': 30
        },
        'performance_targets': {
            'response_time': '<2 seconds',
            'accuracy': '>90%',
            'user_satisfaction': '>85%'
        },
        'monitoring_metrics': [
            'average_response_length',
            'medical_classification_accuracy',
            'citation_inclusion_rate',
            'disclaimer_compliance',
            'response_quality_score'
        ]
    }

# Main prompt selection function
def select_optimal_prompt(query: str, has_textbook_knowledge: bool = False, 
                         medical_content: str = "", sources: list = []) -> str:
    """
    Intelligently select the optimal prompt based on query and available knowledge
    """
    
    classification = create_smart_medical_classifier(query)
    
    if classification['is_medical']:
        if has_textbook_knowledge and medical_content:
            # Full medical response with textbook evidence
            return get_production_medical_prompt(medical_content, sources, query)
        elif has_textbook_knowledge and len(medical_content) < 200:
            # Hybrid response - limited textbook + LLM knowledge
            return get_medical_hybrid_prompt(medical_content, sources, query)
        else:
            # No textbook knowledge available
            return get_medical_no_textbook_prompt(query)
    else:
        # Check if it's a greeting/introduction
        greeting_keywords = ['hello', 'hi', 'hey', 'who are you', 'what are you', 'introduce']
        if any(keyword in query.lower() for keyword in greeting_keywords):
            return get_greeting_introduction_prompt()
        else:
            return get_general_knowledge_prompt()

# Export key functions for main application
__all__ = [
    'get_production_medical_prompt',
    'get_medical_hybrid_prompt', 
    'get_medical_no_textbook_prompt',
    'get_general_knowledge_prompt',
    'create_smart_medical_classifier',
    'select_optimal_prompt',
    'validate_medical_response_quality',
    'optimize_medical_content_extraction',
    'format_medical_sources_concisely'
]