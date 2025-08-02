"""
ENHANCED Medical Helper Functions - Turbo Optimized with Context-Aware Processing
Ultra-fast PDF processing, parallel embedding generation, optimized chunking
Fixed version with proper error handling and dependencies
"""

import os
import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from pathlib import Path

# Core dependencies
try:
    from langchain.document_loaders import PyPDFDirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter  
    from langchain.schema import Document
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"Warning: LangChain imports failed: {e}")
    print("Install with: pip install langchain langchain-huggingface")

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Warning: ML dependencies failed: {e}")
    print("Install with: pip install numpy sentence-transformers")

# Optional performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Info: psutil not available - memory monitoring disabled")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalHelperError(Exception):
    """Custom exception for medical helper operations"""
    pass

def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """TURBO: Load embeddings with speed optimizations and error handling"""
    logger.info("⚡ TURBO: Loading embeddings...")
    
    try:
        # Faster, smaller model for speed - optimized parameters
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu', 'trust_remote_code': False},
            encode_kwargs={'normalize_embeddings': False, 'batch_size': 32}
        )
        
        # Test the embeddings with a simple query
        test_embedding = embeddings.embed_query("medical test")
        if not test_embedding or len(test_embedding) == 0:
            raise MedicalHelperError("Embedding model failed validation test")
            
        logger.info("⚡ Embeddings loaded successfully - TURBO MODE")
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise MedicalHelperError(f"Embedding model initialization failed: {e}")

def fast_clean_text(text: str) -> str:
    """Ultra-fast text cleaning - minimal processing for speed"""
    if not text or not isinstance(text, str) or len(text) < 20:
        return ""
    
    try:
        # Only essential cleaning for speed
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'^\d+\s*,?\s*', '', text, flags=re.MULTILINE)  # Remove page numbers
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/]', '', text)  # Keep medical chars
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        
        return text.strip()
    except Exception as e:
        logger.warning(f"Text cleaning failed: {e}")
        return text.strip() if isinstance(text, str) else ""

def enhanced_medical_text_filter(text: str) -> bool:
    """Enhanced medical content detection for better filtering"""
    if not text or not isinstance(text, str) or len(text) < 50:
        return False
    
    try:
        # Medical keywords for content validation
        medical_indicators = [
            'patient', 'diagnosis', 'treatment', 'symptom', 'disease', 'medical',
            'clinical', 'therapy', 'medication', 'pharmaceutical', 'pathology',
            'anatomy', 'physiology', 'surgery', 'hospital', 'doctor', 'physician',
            'health', 'medicine', 'therapeutic', 'diagnostic', 'procedure',
            'condition', 'syndrome', 'disorder', 'infection', 'virus', 'bacteria',
            'cancer', 'tumor', 'diabetes', 'hypertension', 'cardiovascular',
            'respiratory', 'neurological', 'gastrointestinal', 'dermatological',
            'pharmacology', 'immunology', 'oncology', 'cardiology', 'neurology'
        ]
        
        text_lower = text.lower()
        medical_score = sum(1 for keyword in medical_indicators if keyword in text_lower)
        
        # Also check for medical patterns
        medical_patterns = [
            r'\b\d+\s*mg\b',  # dosage
            r'\b\d+\s*ml\b',  # volume
            r'\bICD[-\s]?\d+\b',  # ICD codes
            r'\b[A-Z]{2,4}[-\s]?\d+\b',  # medical codes
        ]
        
        pattern_score = sum(1 for pattern in medical_patterns if re.search(pattern, text))
        
        # Require at least 2 medical terms or 1 medical pattern
        return medical_score >= 2 or pattern_score >= 1
        
    except Exception as e:
        logger.warning(f"Medical text filtering failed: {e}")
        return len(text) > 200  # Fallback to length-based filter

def minimal_metadata(doc: Document, filename: str) -> Dict[str, Any]:
    """Enhanced metadata extraction with medical context"""
    try:
        # Extract more useful metadata
        content_preview = doc.page_content[:200].replace('\n', ' ').strip()
        
        return {
            'filename': filename,
            'source': doc.metadata.get('source', filename),
            'page': doc.metadata.get('page', 0),
            'medical': True,
            'content_length': len(doc.page_content),
            'preview': content_preview,
            'medical_score': enhanced_medical_text_filter(doc.page_content),
            'word_count': len(doc.page_content.split()),
            'created_at': time.time()
        }
    except Exception as e:
        logger.warning(f"Metadata extraction failed: {e}")
        return {
            'filename': filename,
            'source': filename,
            'page': 0,
            'medical': False,
            'content_length': 0,
            'preview': '',
            'medical_score': False
        }

def process_pdf_fast(pdf_path: str) -> List[Document]:
    """Enhanced fast single PDF processing with better filtering"""
    processed_docs = []
    
    try:
        # Try PyMuPDFLoader first (faster)
        try:
            from langchain.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            loader_type = "PyMuPDF"
        except ImportError:
            # Fallback to PyPDFLoader
            try:
                from langchain.document_loaders import PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                loader_type = "PyPDF"
            except ImportError:
                raise MedicalHelperError("No PDF loader available. Install PyMuPDF or PyPDF2")
        
        filename = os.path.basename(pdf_path)
        
        for i, doc in enumerate(docs):
            try:
                cleaned_text = fast_clean_text(doc.page_content)
                
                # Enhanced filtering
                if len(cleaned_text) > 100 and enhanced_medical_text_filter(cleaned_text):
                    doc.page_content = cleaned_text
                    doc.metadata = minimal_metadata(doc, filename)
                    doc.metadata['page'] = i + 1  # Ensure page number is set
                    processed_docs.append(doc)
                    
            except Exception as e:
                logger.warning(f"Failed to process page {i+1} of {filename}: {e}")
                continue
        
        logger.info(f"⚡ {filename} ({loader_type}): {len(processed_docs)}/{len(docs)} pages passed medical filter")
        return processed_docs
        
    except Exception as e:
        logger.error(f"⚠️ Fast PDF processing failed for {pdf_path}: {e}")
        return []

def prepare_medical_knowledge_base(data_path: str, max_workers: int = 4) -> Tuple[List[Document], Dict[str, Any]]:
    """
    ENHANCED TURBO: Ultra-fast medical knowledge base preparation
    Parallel PDF processing, enhanced filtering, optimized chunking
    """
    logger.info(f"🚀 ENHANCED TURBO: Processing PDFs from {data_path}")
    
    # Quick validation
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDFs found in {data_path}")
    
    logger.info(f"⚡ Found {len(pdf_files)} PDFs - ENHANCED TURBO processing...")
    
    # PARALLEL PDF PROCESSING FOR MAXIMUM SPEED
    all_documents = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {
            executor.submit(process_pdf_fast, str(pdf_path)): pdf_path 
            for pdf_path in pdf_files
        }
        
        for future in as_completed(future_to_pdf, timeout=300):  # 5 minute timeout
            pdf_path = future_to_pdf[future]
            try:
                docs = future.result(timeout=60)  # 1 minute per PDF
                all_documents.extend(docs)
                logger.info(f"⚡ Processed {pdf_path.name}: {len(docs)} medical pages")
            except Exception as e:
                logger.warning(f"⚠️ Failed {pdf_path.name}: {e}")
    
    if not all_documents:
        raise ValueError("No medical content extracted from PDFs")
    
    process_time = time.time() - start_time
    logger.info(f"⚡ PDF processing completed: {len(all_documents)} pages in {process_time:.1f}s")
    
    # ENHANCED TEXT SPLITTING - Optimized for medical content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Slightly larger for medical context
        chunk_overlap=150,   # Better overlap for medical continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]  # Medical-friendly separators
    )
    
    logger.info("⚡ ENHANCED TURBO: Creating optimized medical chunks...")
    
    # PARALLEL CHUNKING with enhanced processing
    def chunk_documents_batch(docs_batch: List[Document]) -> List[Document]:
        try:
            chunks = text_splitter.split_documents(docs_batch)
            # Additional medical filtering at chunk level
            return [chunk for chunk in chunks if enhanced_medical_text_filter(chunk.page_content)]
        except Exception as e:
            logger.warning(f"Chunking batch failed: {e}")
            return []
    
    # Split documents into batches for parallel processing
    batch_size = max(1, len(all_documents) // max_workers)
    doc_batches = [all_documents[i:i + batch_size] for i in range(0, len(all_documents), batch_size)]
    
    all_chunks = []
    chunk_start = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_futures = [executor.submit(chunk_documents_batch, batch) for batch in doc_batches]
        
        for future in as_completed(chunk_futures, timeout=180):  # 3 minute timeout
            try:
                chunks = future.result(timeout=120)  # 2 minutes per batch
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Chunking batch failed: {e}")
    
    chunk_time = time.time() - chunk_start
    logger.info(f"⚡ Chunking completed: {len(all_chunks)} chunks in {chunk_time:.1f}s")
    
    # ENHANCED QUALITY FILTER with medical validation
    filtered_chunks = []
    total_words = 0
    unique_sources = set()
    medical_terms_found = set()
    
    # Medical terminology for validation
    medical_terms = [
        'diagnosis', 'treatment', 'symptoms', 'disease', 'condition', 'therapy',
        'medication', 'pharmaceutical', 'clinical', 'pathology', 'anatomy',
        'physiology', 'surgery', 'medical', 'patient', 'health', 'therapeutic',
        'diagnostic', 'procedure', 'syndrome', 'disorder', 'infection'
    ]
    
    for chunk in all_chunks:
        try:
            content = chunk.page_content
            if len(content) >= 150:  # Minimum chunk size for medical content
                # Count medical terms
                content_lower = content.lower()
                chunk_medical_terms = [term for term in medical_terms if term in content_lower]
                
                if len(chunk_medical_terms) >= 1:  # At least 1 medical term required
                    filtered_chunks.append(chunk)
                    total_words += len(content.split())
                    unique_sources.add(chunk.metadata.get('filename', 'unknown'))
                    medical_terms_found.update(chunk_medical_terms)
        except Exception as e:
            logger.warning(f"Chunk filtering failed: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # ENHANCED VALIDATION REPORT
    validation_report = {
        'total_chunks': len(filtered_chunks),
        'unique_books': len(unique_sources),
        'unique_sources': list(unique_sources),
        'total_words': total_words,
        'medical_terms_coverage': f"{len(medical_terms_found)}/{len(medical_terms)} terms found",
        'medical_terms_found': list(medical_terms_found),
        'quality_score': min(100, len(unique_sources) * 15 + len(medical_terms_found) * 2),
        'average_chunk_size': total_words // len(filtered_chunks) if filtered_chunks else 0,
        'processing_time': {
            'pdf_processing': f"{process_time:.1f}s",
            'chunking': f"{chunk_time:.1f}s",
            'total': f"{total_time:.1f}s"
        },
        'performance_metrics': {
            'pages_per_second': len(all_documents) / process_time if process_time > 0 else 0,
            'chunks_per_second': len(all_chunks) / chunk_time if chunk_time > 0 else 0,
            'words_per_second': total_words / total_time if total_time > 0 else 0
        }
    }
    
    logger.info("⚡ ENHANCED TURBO knowledge base ready:")
    logger.info(f"   📄 Total chunks: {len(filtered_chunks)}")
    logger.info(f"   📚 Sources: {len(unique_sources)}")
    logger.info(f"   🔤 Total words: {total_words:,}")
    logger.info(f"   📏 Avg chunk: {validation_report['average_chunk_size']} words")
    logger.info(f"   ⭐ Quality: {validation_report['quality_score']}/100")
    logger.info(f"   🏥 Medical terms: {len(medical_terms_found)}/{len(medical_terms)}")
    logger.info(f"   ⚡ Performance: {validation_report['performance_metrics']['pages_per_second']:.1f} pages/sec")
    
    if not filtered_chunks:
        raise ValueError("No valid medical chunks created")
    
    return filtered_chunks, validation_report

def batch_embed_texts(embeddings_model: HuggingFaceEmbeddings, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Enhanced ultra-fast batch embedding generation with error recovery"""
    if not texts:
        return []
    
    all_embeddings = []
    
    logger.info(f"⚡ Generating embeddings for {len(texts)} texts in batches of {batch_size}")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        try:
            # Use embed_documents for batch processing
            batch_embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"⚡ Embedding batch {batch_num}/{total_batches} completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Embedding batch {batch_num} failed: {e}")
            
            # Fallback: process individually with error handling
            for j, text in enumerate(batch):
                try:
                    embedding = embeddings_model.embed_query(text)
                    all_embeddings.append(embedding)
                except Exception as e2:
                    logger.warning(f"⚠️ Individual embedding failed for text {i+j+1}: {e2}")
                    # Create zero embedding as placeholder
                    all_embeddings.append([0.0] * 384)  # MiniLM dimension
    
    logger.info(f"⚡ Embedding generation completed: {len(all_embeddings)} embeddings")
    return all_embeddings

def optimize_chunk_metadata(chunk: Document) -> Dict[str, Any]:
    """Enhanced metadata optimization for Pinecone with medical context"""
    try:
        content = chunk.page_content
        
        # Extract medical keywords from content
        medical_keywords = [
            'diagnosis', 'treatment', 'symptoms', 'disease', 'condition', 'therapy',
            'medication', 'clinical', 'patient', 'medical', 'health', 'surgery'
        ]
        
        found_keywords = [kw for kw in medical_keywords if kw.lower() in content.lower()]
        
        return {
            'text': content[:800],  # Increased for better context
            'source': str(chunk.metadata.get('filename', 'unknown'))[:50],
            'page': str(chunk.metadata.get('page', 0)),
            'length': len(content),
            'medical_keywords': found_keywords[:5],  # Top 5 medical keywords
            'content_type': 'medical',
            'word_count': len(content.split()),
            'quality_score': len(found_keywords)
        }
    except Exception as e:
        logger.warning(f"Metadata optimization failed: {e}")
        return {
            'text': str(chunk.page_content)[:800],
            'source': 'unknown',
            'page': '0',
            'length': len(chunk.page_content),
            'medical_keywords': [],
            'content_type': 'medical',
            'word_count': 0,
            'quality_score': 0
        }

def prepare_pinecone_vectors_fast(chunks: List[Document], embeddings_model: HuggingFaceEmbeddings, start_id: int = 0) -> List[Dict[str, Any]]:
    """ENHANCED TURBO: Prepare vectors for direct Pinecone upsert with medical optimization"""
    if not chunks:
        return []
        
    logger.info(f"⚡ ENHANCED TURBO: Preparing {len(chunks)} vectors for upsert...")
    
    # Extract texts for batch embedding
    texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings in large batches for maximum speed
    start_time = time.time()
    embeddings = batch_embed_texts(embeddings_model, texts, batch_size=64)
    embedding_time = time.time() - start_time
    
    logger.info(f"⚡ Embeddings generated in {embedding_time:.1f}s ({len(embeddings)/embedding_time:.1f} embeddings/sec)")
    
    # Prepare vectors with enhanced metadata
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        try:
            vectors.append({
                'id': f"med_{start_id + i}",
                'values': embedding,
                'metadata': optimize_chunk_metadata(chunk)
            })
        except Exception as e:
            logger.warning(f"Vector preparation failed for chunk {i}: {e}")
            continue
    
    logger.info(f"⚡ {len(vectors)} vectors ready for ENHANCED TURBO upload")
    return vectors

# ENHANCED VALIDATION FUNCTIONS
def quick_validate_setup(data_path: str) -> Dict[str, Any]:
    """Enhanced setup validation with detailed analysis"""
    validation_results = {
        'data_path_exists': False,
        'pdf_files': [],
        'total_size_mb': 0,
        'estimated_processing_time': 0,
        'setup_valid': False,
        'error': None
    }
    
    try:
        data_path = Path(data_path)
        validation_results['data_path_exists'] = data_path.exists()
        
        if not validation_results['data_path_exists']:
            validation_results['error'] = f"Directory does not exist: {data_path}"
            return validation_results
        
        pdf_files = list(data_path.glob("*.pdf"))
        validation_results['pdf_files'] = [f.name for f in pdf_files]
        
        if pdf_files:
            total_size = sum(f.stat().st_size for f in pdf_files)
            validation_results['total_size_mb'] = total_size / (1024 * 1024)
            validation_results['estimated_processing_time'] = max(0.5, total_size / (1024 * 1024) / 3)  # ~3MB per minute
            validation_results['setup_valid'] = True
        else:
            validation_results['error'] = "No PDF files found in directory"
        
    except Exception as e:
        validation_results['error'] = str(e)
    
    return validation_results

def estimate_processing_time(data_path: str) -> Dict[str, Any]:
    """Enhanced processing time estimation with detailed breakdown"""
    try:
        data_path = Path(data_path)
        pdf_files = list(data_path.glob("*.pdf"))
        
        if not pdf_files:
            return {'error': 'No PDF files found'}
        
        total_size_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
        
        # Enhanced estimates based on processing stages
        estimates = {
            'pdf_processing': max(0.5, total_size_mb / 4),  # ~4MB per minute
            'chunking': max(0.2, total_size_mb / 10),       # ~10MB per minute  
            'embedding_generation': max(1.0, total_size_mb / 2),  # ~2MB per minute
            'pinecone_upload': max(0.5, total_size_mb / 8), # ~8MB per minute
        }
        
        estimates['total_estimated'] = sum(estimates.values())
        estimates['file_count'] = len(pdf_files)
        estimates['total_size_mb'] = total_size_mb
        
        return estimates
        
    except Exception as e:
        return {'error': str(e)}

def turbo_chunk_filter(chunks: List[Document], min_length: int = 120) -> List[Document]:
    """Enhanced ultra-fast chunk filtering with medical relevance"""
    if not chunks:
        return []
        
    # Multi-criteria filtering for speed and quality
    filtered_chunks = []
    
    for chunk in chunks:
        try:
            content = chunk.page_content
            
            # Basic length filter
            if len(content) < min_length:
                continue
            
            # Quick medical relevance check
            content_lower = content.lower()
            medical_indicators = ['medical', 'patient', 'treatment', 'diagnosis', 'health', 'clinical', 'therapy']
            medical_score = sum(1 for indicator in medical_indicators if indicator in content_lower)
            
            # Require at least some medical relevance
            if medical_score > 0:
                filtered_chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Chunk filtering failed: {e}")
            continue
    
    logger.info(f"⚡ Chunk filter: {len(filtered_chunks)}/{len(chunks)} chunks passed medical relevance filter")
    return filtered_chunks

# ENHANCED PERFORMANCE MONITORING
def log_performance_stats(operation: str, start_time: float, item_count: int, additional_metrics: Optional[Dict[str, Any]] = None):
    """Enhanced performance statistics logging"""
    try:
        elapsed = time.time() - start_time
        rate = item_count / elapsed if elapsed > 0 else 0
        
        log_message = f"⚡ {operation}: {elapsed:.1f}s | {rate:.1f} items/sec | {item_count} items"
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                if isinstance(value, float):
                    log_message += f" | {key}: {value:.2f}"
                else:
                    log_message += f" | {key}: {value}"
        
        logger.info(log_message)
    except Exception as e:
        logger.warning(f"Performance logging failed: {e}")

def log_memory_usage():
    """Enhanced memory usage logging"""
    if not PSUTIL_AVAILABLE:
        logger.info("⚡ Memory monitoring requires psutil package")
        return
        
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        logger.info(f"⚡ Memory: RSS {mem_info.rss / (1024 * 1024):.1f}MB | "
                f"VMS {mem_info.vms / (1024 * 1024):.1f}MB | "
                f"CPU {process.cpu_percent():.1f}%")
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e}")

def log_turbo_summary(chunks: List[Document], validation_report: Dict[str, Any]):
    """Enhanced concise summary of the TURBO processing"""
    try:
        logger.info("⚡ ENHANCED TURBO SUMMARY:")
        logger.info(f"   📄 Total chunks: {len(chunks)}")
        logger.info(f"   📚 Unique sources: {validation_report.get('unique_books', 0)}")
        logger.info(f"   🔤 Total words: {validation_report.get('total_words', 0):,}")
        logger.info(f"   📏 Avg chunk size: {validation_report.get('average_chunk_size', 0)} words")
        logger.info(f"   ⭐ Quality score: {validation_report.get('quality_score', 0)}/100")
        logger.info(f"   🏥 Medical coverage: {validation_report.get('medical_terms_coverage', 'N/A')}")
        
        perf_metrics = validation_report.get('performance_metrics', {})
        logger.info(f"   ⚡ Performance: {perf_metrics.get('pages_per_second', 0):.1f} pages/sec")
        
        if 'processing_time' in validation_report:
            times = validation_report['processing_time']
            logger.info(f"   ⏱️ Timing: PDF {times.get('pdf_processing', 'N/A')} | "
                    f"Chunk {times.get('chunking', 'N/A')} | Total {times.get('total', 'N/A')}")
    except Exception as e:
        logger.warning(f"Summary logging failed: {e}")

def validate_medical_content_quality(chunks: List[Document]) -> Dict[str, Any]:
    """Enhanced medical content quality validation"""
    if not chunks:
        return {'quality_score': 0, 'issues': ['No chunks provided']}
    
    quality_metrics = {
        'total_chunks': len(chunks),
        'medical_relevance': 0,
        'average_length': 0,
        'unique_sources': set(),
        'medical_terms_coverage': 0,
        'quality_issues': []
    }
    
    medical_terms = [
        'diagnosis', 'treatment', 'symptoms', 'disease', 'condition', 'therapy',
        'medication', 'clinical', 'patient', 'medical', 'health', 'surgery',
        'anatomy', 'physiology', 'pathology', 'pharmaceutical'
    ]
    
    try:
        total_length = 0
        medical_chunks = 0
        all_found_terms = set()
        
        for chunk in chunks:
            content = chunk.page_content
            content_lower = content.lower()
            
            total_length += len(content)
            quality_metrics['unique_sources'].add(chunk.metadata.get('filename', 'unknown'))
            
            # Check medical relevance
            found_terms = [term for term in medical_terms if term in content_lower]
            if found_terms:
                medical_chunks += 1
                all_found_terms.update(found_terms)
        
        quality_metrics['average_length'] = total_length // len(chunks)
        quality_metrics['medical_relevance'] = (medical_chunks / len(chunks)) * 100
        quality_metrics['medical_terms_coverage'] = (len(all_found_terms) / len(medical_terms)) * 100
        quality_metrics['unique_sources'] = len(quality_metrics['unique_sources'])
        
        # Calculate overall quality score
        quality_score = (
            min(100, quality_metrics['medical_relevance']) * 0.4 +
            min(100, quality_metrics['medical_terms_coverage']) * 0.3 +
            min(100, (quality_metrics['average_length'] / 500) * 100) * 0.2 +
            min(100, quality_metrics['unique_sources'] * 20) * 0.1
        )
        
        quality_metrics['quality_score'] = quality_score
        
        # Identify quality issues
        if quality_metrics['medical_relevance'] < 80:
            quality_metrics['quality_issues'].append('Low medical relevance')
        if quality_metrics['medical_terms_coverage'] < 50:
            quality_metrics['quality_issues'].append('Poor medical terminology coverage')
        if quality_metrics['average_length'] < 200:
            quality_metrics['quality_issues'].append('Chunks too short for meaningful content')
        if quality_metrics['unique_sources'] < 3:
            quality_metrics['quality_issues'].append('Limited source diversity')
        
    except Exception as e:
        logger.error(f"Quality validation failed: {e}")
        quality_metrics['quality_score'] = 0
        quality_metrics['quality_issues'] = [f'Validation error: {str(e)}']
    
    return quality_metrics

# MEDICAL KNOWLEDGE VALIDATION
def validate_medical_terminology_coverage(chunks: List[Document]) -> Dict[str, Any]:
    """Validate coverage of essential medical terminology"""
    
    # Essential medical categories and terms
    medical_categories = {
        'anatomy': ['heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine', 'bone', 'muscle'],
        'diseases': ['diabetes', 'cancer', 'hypertension', 'asthma', 'arthritis', 'infection', 'tumor'],
        'symptoms': ['pain', 'fever', 'cough', 'nausea', 'fatigue', 'headache', 'dizziness', 'swelling'],
        'treatments': ['surgery', 'medication', 'therapy', 'treatment', 'cure', 'prevention', 'rehabilitation'],
        'diagnostics': ['diagnosis', 'test', 'examination', 'scan', 'biopsy', 'blood test', 'x-ray'],
        'medical_professionals': ['doctor', 'physician', 'surgeon', 'nurse', 'specialist', 'therapist']
    }
    
    try:
        coverage_report = {}
        all_content = ' '.join([chunk.page_content.lower() for chunk in chunks])
        
        for category, terms in medical_categories.items():
            found_terms = [term for term in terms if term in all_content]
            coverage_report[category] = {
                'found_terms': found_terms,
                'coverage_percentage': (len(found_terms) / len(terms)) * 100,
                'missing_terms': [term for term in terms if term not in found_terms]
            }
        
        # Overall coverage score
        overall_coverage = sum(cat['coverage_percentage'] for cat in coverage_report.values()) / len(coverage_report)
        
        return {
            'overall_coverage': overall_coverage,
            'category_coverage': coverage_report,
            'recommendations': _generate_coverage_recommendations(coverage_report)
        }
        
    except Exception as e:
        logger.error(f"Terminology coverage validation failed: {e}")
        return {
            'overall_coverage': 0,
            'category_coverage': {},
            'recommendations': [f'Validation error: {str(e)}'],
            'error': str(e)
        }

def _generate_coverage_recommendations(coverage_report: Dict[str, Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on coverage analysis"""
    recommendations = []
    
    try:
        for category, data in coverage_report.items():
            if data['coverage_percentage'] < 60:
                recommendations.append(f"Consider adding more {category}-related content")
            if len(data['missing_terms']) > len(data['found_terms']):
                recommendations.append(f"Low {category} terminology coverage - only {len(data['found_terms'])} terms found")
        
        if not recommendations:
            recommendations.append("Excellent medical terminology coverage across all categories")
            
    except Exception as e:
        recommendations.append(f"Recommendation generation failed: {str(e)}")
    
    return recommendations

# ADVANCED HELPER FUNCTIONS
def create_medical_index_summary(chunks: List[Document]) -> Dict[str, Any]:
    """Create a comprehensive summary of the medical knowledge base"""
    try:
        if not chunks:
            return {'error': 'No chunks provided'}
        
        # Collect statistics
        total_chunks = len(chunks)
        total_words = sum(len(chunk.page_content.split()) for chunk in chunks)
        sources = set(chunk.metadata.get('filename', 'unknown') for chunk in chunks)
        
        # Medical term frequency analysis
        medical_terms = {}
        all_text = ' '.join(chunk.page_content.lower() for chunk in chunks)
        
        key_terms = [
            'diagnosis', 'treatment', 'symptoms', 'disease', 'condition', 'therapy',
            'medication', 'clinical', 'patient', 'medical', 'health', 'surgery',
            'anatomy', 'physiology', 'pathology', 'pharmaceutical', 'diagnostic'
        ]
        
        for term in key_terms:
            medical_terms[term] = all_text.count(term)
        
        # Top medical terms
        top_terms = sorted(medical_terms.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Source analysis
        source_stats = {}
        for chunk in chunks:
            source = chunk.metadata.get('filename', 'unknown')
            if source not in source_stats:
                source_stats[source] = {'chunks': 0, 'words': 0}
            source_stats[source]['chunks'] += 1
            source_stats[source]['words'] += len(chunk.page_content.split())
        
        return {
            'total_chunks': total_chunks,
            'total_words': total_words,
            'unique_sources': len(sources),
            'sources': list(sources),
            'average_chunk_size': total_words // total_chunks,
            'top_medical_terms': top_terms,
            'source_breakdown': source_stats,
            'content_quality': 'high' if total_words > 50000 else 'medium' if total_words > 10000 else 'low'
        }
        
    except Exception as e:
        logger.error(f"Index summary creation failed: {e}")
        return {'error': str(e)}

def optimize_chunks_for_retrieval(chunks: List[Document], target_size: int = 1000) -> List[Document]:
    """Optimize chunk sizes for better retrieval performance"""
    optimized_chunks = []
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_size,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "]
        )
        
        for chunk in chunks:
            if len(chunk.page_content) > target_size * 1.5:
                # Split large chunks
                sub_chunks = text_splitter.split_documents([chunk])
                optimized_chunks.extend(sub_chunks)
            elif len(chunk.page_content) < target_size * 0.3:
                # Skip very small chunks or combine with metadata
                if enhanced_medical_text_filter(chunk.page_content):
                    optimized_chunks.append(chunk)
            else:
                # Keep appropriately sized chunks
                optimized_chunks.append(chunk)
        
        logger.info(f"⚡ Chunk optimization: {len(chunks)} → {len(optimized_chunks)} chunks")
        return optimized_chunks
        
    except Exception as e:
        logger.error(f"Chunk optimization failed: {e}")
        return chunks

def export_knowledge_base_stats(chunks: List[Document], output_path: str) -> bool:
    """Export detailed statistics about the knowledge base to a file"""
    try:
        import json
        
        stats = {
            'generation_time': time.time(),
            'total_chunks': len(chunks),
            'summary': create_medical_index_summary(chunks),
            'quality_metrics': validate_medical_content_quality(chunks),
            'terminology_coverage': validate_medical_terminology_coverage(chunks)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"⚡ Knowledge base stats exported to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Stats export failed: {e}")
        return False

def validate_embeddings_model(embeddings_model: HuggingFaceEmbeddings) -> Dict[str, Any]:
    """Validate that the embeddings model is working correctly"""
    try:
        # Test with medical content
        test_texts = [
            "Patient presents with acute myocardial infarction symptoms",
            "Diagnosis and treatment of diabetes mellitus",
            "Surgical intervention for cardiac bypass"
        ]
        
        start_time = time.time()
        embeddings = embeddings_model.embed_documents(test_texts)
        embedding_time = time.time() - start_time
        
        # Validate embeddings
        if not embeddings or len(embeddings) != len(test_texts):
            return {'valid': False, 'error': 'Incorrect number of embeddings generated'}
        
        # Check embedding dimensions
        dimensions = len(embeddings[0]) if embeddings else 0
        if dimensions == 0:
            return {'valid': False, 'error': 'Empty embeddings generated'}
        
        # Check for valid values
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list) or len(emb) != dimensions:
                return {'valid': False, 'error': f'Invalid embedding structure at index {i}'}
            if all(v == 0 for v in emb):
                return {'valid': False, 'error': f'Zero embedding at index {i}'}
        
        return {
            'valid': True,
            'dimensions': dimensions,
            'test_embeddings_count': len(embeddings),
            'embedding_time': embedding_time,
            'average_time_per_text': embedding_time / len(test_texts)
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def cleanup_temp_files(temp_dir: str = "./temp_medical") -> bool:
    """Clean up temporary files created during processing"""
    try:
        import shutil
        temp_path = Path(temp_dir)
        
        if temp_path.exists():
            shutil.rmtree(temp_path)
            logger.info(f"⚡ Cleaned up temporary directory: {temp_dir}")
            return True
        return True
        
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")
        return False

# MAIN WORKFLOW FUNCTION
def create_medical_knowledge_base_complete(
    data_path: str,
    output_stats_path: Optional[str] = None,
    max_workers: int = 4,
    optimize_for_retrieval: bool = True
) -> Tuple[List[Document], HuggingFaceEmbeddings, Dict[str, Any]]:
    """
    Complete workflow to create a medical knowledge base with embeddings
    
    Args:
        data_path: Path to directory containing PDF files
        output_stats_path: Optional path to export statistics
        max_workers: Number of parallel workers for processing
        optimize_for_retrieval: Whether to optimize chunks for retrieval
    
    Returns:
        Tuple of (processed_chunks, embeddings_model, processing_report)
    """
    logger.info("🚀 Starting complete medical knowledge base creation...")
    workflow_start = time.time()
    
    try:
        # Step 1: Validate setup
        logger.info("📋 Step 1: Validating setup...")
        validation = quick_validate_setup(data_path)
        if not validation['setup_valid']:
            raise MedicalHelperError(f"Setup validation failed: {validation.get('error', 'Unknown error')}")
        
        # Step 2: Load embeddings model
        logger.info("🤖 Step 2: Loading embeddings model...")
        embeddings_model = download_hugging_face_embeddings()
        
        # Step 3: Validate embeddings model
        logger.info("🔍 Step 3: Validating embeddings model...")
        model_validation = validate_embeddings_model(embeddings_model)
        if not model_validation['valid']:
            raise MedicalHelperError(f"Embeddings model validation failed: {model_validation.get('error')}")
        
        # Step 4: Process PDFs and create knowledge base
        logger.info("📚 Step 4: Processing PDFs and creating knowledge base...")
        chunks, processing_report = prepare_medical_knowledge_base(data_path, max_workers=max_workers)
        
        # Step 5: Optimize chunks for retrieval (optional)
        if optimize_for_retrieval:
            logger.info("⚡ Step 5: Optimizing chunks for retrieval...")
            chunks = optimize_chunks_for_retrieval(chunks)
            processing_report['optimized_chunks'] = len(chunks)
        
        # Step 6: Final quality validation
        logger.info("✅ Step 6: Final quality validation...")
        quality_metrics = validate_medical_content_quality(chunks)
        terminology_coverage = validate_medical_terminology_coverage(chunks)
        
        # Compile final report
        workflow_time = time.time() - workflow_start
        final_report = {
            **processing_report,
            'workflow_time': workflow_time,
            'setup_validation': validation,
            'model_validation': model_validation,
            'final_quality_metrics': quality_metrics,
            'terminology_coverage': terminology_coverage,
            'workflow_status': 'completed_successfully'
        }
        
        # Step 7: Export statistics (optional)
        if output_stats_path:
            logger.info(f"📊 Step 7: Exporting statistics to {output_stats_path}...")
            export_knowledge_base_stats(chunks, output_stats_path)
        
        # Final summary
        logger.info("🎉 Medical knowledge base creation completed successfully!")
        log_turbo_summary(chunks, final_report)
        
        return chunks, embeddings_model, final_report
        
    except Exception as e:
        error_report = {
            'workflow_status': 'failed',
            'error': str(e),
            'workflow_time': time.time() - workflow_start,
            'traceback': traceback.format_exc()
        }
        logger.error(f"❌ Medical knowledge base creation failed: {e}")
        raise MedicalHelperError(f"Workflow failed: {e}") from e

# EXPORT FUNCTIONS
__all__ = [
    # Core functions
    'download_hugging_face_embeddings',
    'prepare_medical_knowledge_base',
    'batch_embed_texts',
    'prepare_pinecone_vectors_fast',
    'create_medical_knowledge_base_complete',
    
    # Processing functions
    'fast_clean_text',
    'enhanced_medical_text_filter',
    'process_pdf_fast',
    'optimize_chunk_metadata',
    'turbo_chunk_filter',
    'optimize_chunks_for_retrieval',
    
    # Validation functions
    'quick_validate_setup',
    'estimate_processing_time',
    'validate_medical_content_quality',
    'validate_medical_terminology_coverage',
    'validate_embeddings_model',
    
    # Monitoring and utilities
    'log_performance_stats',
    'log_memory_usage',
    'log_turbo_summary',
    'create_medical_index_summary',
    'export_knowledge_base_stats',
    'cleanup_temp_files',
    
    # Exception class
    'MedicalHelperError'
]

# Usage Example
if __name__ == "__main__":
    # Example usage
    try:
        # Create complete medical knowledge base
        chunks, embeddings, report = create_medical_knowledge_base_complete(
            data_path="./medical_pdfs",
            output_stats_path="./medical_kb_stats.json",
            max_workers=4,
            optimize_for_retrieval=True
        )
        
        print(f"✅ Successfully created knowledge base with {len(chunks)} chunks")
        print(f"📊 Quality score: {report['final_quality_metrics']['quality_score']:.1f}/100")
        
    except MedicalHelperError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")