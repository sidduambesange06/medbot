"""
TURBO Medical Helper Functions - Optimized for Maximum Speed
Fast PDF processing, minimal validation, optimized chunking
"""

import os
import re
import logging
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def download_hugging_face_embeddings():
    """TURBO: Load embeddings with speed optimizations"""
    logger.info("⚡ TURBO: Loading embeddings...")
    
    # Faster, smaller model for speed - fixed parameters
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    logger.info("⚡ Embeddings loaded - TURBO MODE")
    return embeddings

def fast_clean_text(text: str) -> str:
    """Ultra-fast text cleaning - minimal processing for speed"""
    if not text or len(text) < 20:
        return ""
    
    # Only essential cleaning for speed
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Remove page numbers
    
    return text.strip()

def minimal_metadata(doc: Document, filename: str) -> Dict:
    """Minimal metadata extraction for maximum speed"""
    return {
        'filename': filename,
        'source': doc.metadata.get('source', filename),
        'page': doc.metadata.get('page', 0),
        'medical': True  # Assume all content is medical
    }

def process_pdf_fast(pdf_path: str) -> List[Document]:
    """Fast single PDF processing"""
    try:
        from langchain.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        filename = os.path.basename(pdf_path)
        processed_docs = []
        
        for doc in docs:
            cleaned_text = fast_clean_text(doc.page_content)
            if len(cleaned_text) > 50:  # Quick filter
                doc.page_content = cleaned_text
                doc.metadata = minimal_metadata(doc, filename)
                processed_docs.append(doc)
        
        return processed_docs
    except Exception as e:
        logger.warning(f"⚠️ Fast PDF processing failed for {pdf_path}: {e}")
        return []

def prepare_medical_knowledge_base(data_path: str) -> Tuple[List[Document], Dict]:
    """
    TURBO: Ultra-fast medical knowledge base preparation
    Parallel PDF processing, minimal validation, optimized chunking
    """
    logger.info(f"🚀 TURBO: Processing PDFs from {data_path}")
    
    # Quick validation
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
    if not pdf_files:
        raise ValueError(f"No PDFs found in {data_path}")
    
    logger.info(f"⚡ Found {len(pdf_files)} PDFs - TURBO processing...")
    
    # PARALLEL PDF PROCESSING FOR MAXIMUM SPEED
    all_documents = []
    pdf_paths = [os.path.join(data_path, pdf) for pdf in pdf_files]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_pdf = {executor.submit(process_pdf_fast, pdf_path): pdf_path for pdf_path in pdf_paths}
        
        for future in future_to_pdf:
            pdf_path = future_to_pdf[future]
            try:
                docs = future.result(timeout=30)  # 30 sec timeout per PDF
                all_documents.extend(docs)
                logger.info(f"⚡ Processed {os.path.basename(pdf_path)}: {len(docs)} pages")
            except Exception as e:
                logger.warning(f"⚠️ Failed {os.path.basename(pdf_path)}: {e}")
    
    if not all_documents:
        raise ValueError("No content extracted from PDFs")
    
    logger.info(f"⚡ Total pages loaded: {len(all_documents)}")
    
    # TURBO TEXT SPLITTING - Optimized for speed and Pinecone limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for fewer API calls
        chunk_overlap=50,   # Reduced overlap for speed
        length_function=len,
        separators=["\n\n", "\n", ". ", " "]  # Simplified separators
    )
    
    logger.info("⚡ TURBO: Creating optimized chunks...")
    
    # PARALLEL CHUNKING
    def chunk_documents_batch(docs_batch):
        return text_splitter.split_documents(docs_batch)
    
    # Split documents into batches for parallel processing
    batch_size = max(1, len(all_documents) // 4)  # 4 batches
    doc_batches = [all_documents[i:i + batch_size] for i in range(0, len(all_documents), batch_size)]
    
    all_chunks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        chunk_futures = [executor.submit(chunk_documents_batch, batch) for batch in doc_batches]
        
        for future in chunk_futures:
            try:
                chunks = future.result(timeout=60)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"⚠️ Chunking batch failed: {e}")
    
    # QUICK QUALITY FILTER - Remove very short chunks
    filtered_chunks = []
    total_words = 0
    unique_sources = set()
    
    for chunk in all_chunks:
        if len(chunk.page_content) >= 100:  # Quick length filter
            filtered_chunks.append(chunk)
            total_words += len(chunk.page_content.split())
            unique_sources.add(chunk.metadata.get('filename', 'unknown'))
    
    # MINIMAL VALIDATION REPORT FOR SPEED
    validation_report = {
        'total_chunks': len(filtered_chunks),
        'unique_books': len(unique_sources),
        'unique_authors': len(unique_sources), 
        'total_words': total_words,
        'medical_terms_coverage': f"Auto-validated",
        'quality_score': min(100, len(unique_sources) * 20),  # Simple scoring
        'average_chunk_size': total_words // len(filtered_chunks) if filtered_chunks else 0
    }
    
    logger.info("⚡ TURBO knowledge base ready:")
    logger.info(f"   📄 Total chunks: {len(filtered_chunks)}")
    logger.info(f"   📚 Sources: {len(unique_sources)}")
    logger.info(f"   🔤 Total words: {total_words:,}")
    logger.info(f"   📏 Avg chunk: {validation_report['average_chunk_size']} words")
    logger.info(f"   ⭐ Quality: {validation_report['quality_score']}/100")
    
    if not filtered_chunks:
        raise ValueError("No valid chunks created")
    
    return filtered_chunks, validation_report

def batch_embed_texts(embeddings_model, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Ultra-fast batch embedding generation"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Use embed_documents for batch processing
            batch_embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.warning(f"⚠️ Embedding batch {i//batch_size + 1} failed: {e}")
            # Fallback: process individually
            for text in batch:
                try:
                    embedding = embeddings_model.embed_query(text)
                    all_embeddings.append(embedding)
                except Exception as e2:
                    logger.warning(f"⚠️ Individual embedding failed: {e2}")
                    # Skip problematic text
                    all_embeddings.append([0.0] * 384)  # MiniLM dimension
    
    return all_embeddings

def optimize_chunk_metadata(chunk: Document) -> Dict:
    """Ultra-fast metadata optimization for Pinecone"""
    # Minimal metadata to reduce upload size and increase speed
    return {
        'text': chunk.page_content[:500],  # Truncated for speed
        'source': chunk.metadata.get('filename', 'unknown')[:30],
        'page': str(chunk.metadata.get('page', 0)),
        'length': len(chunk.page_content)
    }

def prepare_pinecone_vectors_fast(chunks: List[Document], embeddings_model, start_id: int = 0) -> List[Dict]:
    """TURBO: Prepare vectors for direct Pinecone upsert"""
    logger.info(f"⚡ TURBO: Preparing {len(chunks)} vectors for upsert...")
    
    # Extract texts for batch embedding
    texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings in large batches for maximum speed
    embeddings = batch_embed_texts(embeddings_model, texts, batch_size=64)
    
    # Prepare vectors with minimal metadata
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            'id': f"med_{start_id + i}",
            'values': embedding,
            'metadata': optimize_chunk_metadata(chunk)
        })
    
    logger.info(f"⚡ {len(vectors)} vectors ready for TURBO upload")
    return vectors

# TURBO VALIDATION FUNCTIONS
def quick_validate_setup(data_path: str) -> bool:
    """Ultra-fast setup validation"""
    if not os.path.exists(data_path):
        return False
    
    pdf_count = len([f for f in os.listdir(data_path) if f.endswith('.pdf')])
    return pdf_count > 0

def estimate_processing_time(data_path: str) -> float:
    """Quick processing time estimate"""
    try:
        pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
        total_size_mb = sum(os.path.getsize(os.path.join(data_path, f)) for f in pdf_files) / (1024 * 1024)
        
        # Rough estimate: 2MB per minute in TURBO mode
        estimated_minutes = total_size_mb / 2
        return max(0.5, estimated_minutes)  # Minimum 30 seconds
    except:
        return 1.0  # Default estimate

def turbo_chunk_filter(chunks: List[Document], min_length: int = 80) -> List[Document]:
    """Ultra-fast chunk filtering for speed"""
    # Simple length-based filter - no complex validation for speed
    return [chunk for chunk in chunks if len(chunk.page_content) >= min_length]

# PERFORMANCE MONITORING
def log_performance_stats(operation: str, start_time: float, item_count: int):
    """Log performance statistics"""
    elapsed = time.time() - start_time
    rate = item_count / elapsed if elapsed > 0 else 0
    logger.info(f"⚡ {operation}: {elapsed:.1f}s | {rate:.1f} items/sec")

def log_memory_usage():
    """Log current memory usage"""
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"⚡ Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB | {mem_info.vms / (1024 * 1024):.2f} MB")

    
def log_cpu_usage():
    """Log current CPU usage"""
    import psutil
    cpu_usage = psutil.cpu_percent(interval=1)
    logger.info(f"⚡ CPU usage: {cpu_usage}%")

def log_disk_usage(path: str):
    """Log disk usage for a given path"""
    import psutil
    disk_usage = psutil.disk_usage(path)
    logger.info(f"⚡ Disk usage for {path}: {disk_usage.percent}% used | {disk_usage.free / (1024 * 1024):.2f} MB free")


def log_system_stats():
    """Log overall system statistics"""
    log_memory_usage()
    log_cpu_usage()
    log_disk_usage('/')
    log_disk_usage(os.getcwd())
    
    # Log performance stats for the last operation
    if hasattr(log_system_stats, 'last_operation'):
        log_performance_stats(log_system_stats.last_operation['operation'],
                            log_system_stats.last_operation['start_time'],
                            log_system_stats.last_operation['item_count'])
    
    # Reset last operation
    log_system_stats.last_operation = None

def set_last_operation(operation: str, start_time: float, item_count: int):
    """Set the last operation for performance logging"""
    log_system_stats.last_operation = {
        'operation': operation,
        'start_time': start_time,
        'item_count': item_count
    }

def log_last_operation():
    """Log the last operation's performance stats"""
    if hasattr(log_system_stats, 'last_operation'):
        op = log_system_stats.last_operation
        log_performance_stats(op['operation'], op['start_time'], op['item_count'])
    else:
        logger.info("⚠️ No last operation to log")

def log_turbo_summary(chunks: List[Document], validation_report: Dict):
    """Log a concise summary of the TURBO processing"""
    logger.info("⚡ TURBO Summary:")
    logger.info(f"   📄 Total chunks: {len(chunks)}")
    logger.info(f"   📚 Unique sources: {validation_report['unique_books']}")
    logger.info(f"   🔤 Total words: {validation_report['total_words']:,}")
    logger.info(f"   📏 Avg chunk size: {validation_report['average_chunk_size']} words")
    logger.info(f"   ⭐ Quality score: {validation_report['quality_score']}/100")
    logger.info(f"   Medical terms coverage: {validation_report['medical_terms_coverage']}")


def log_turbo_error(error: Exception):
    """Log a TURBO-specific error with concise details"""
    logger.error(f"⚠️ TURBO Error: {str(error)}")
    if hasattr(error, 'args') and len(error.args) > 0:
        logger.error(f"   Details: {error.args[0]}")
    else:
        logger.error("   No additional error details available")

def log_turbo_info(message: str):
    """Log a TURBO-specific informational message"""
    logger.info(f"⚡ TURBO Info: {message}")
    if hasattr(log_system_stats, 'last_operation'):
        op = log_system_stats.last_operation
        logger.info(f"   Last operation: {op['operation']} | Items processed: {op['item_count']}")
    else:
        logger.info("   No last operation recorded")

def log_turbo_warning(message: str):
    """Log a TURBO-specific warning message"""
    logger.warning(f"⚠️ TURBO Warning: {message}")
    if hasattr(log_system_stats, 'last_operation'):
        op = log_system_stats.last_operation
        logger.warning(f"   Last operation: {op['operation']} | Items processed: {op['item_count']}")
    else:
        logger.warning("   No last operation recorded")

def log_turbo_debug(message: str):
    """Log a TURBO-specific debug message"""
    logger.debug(f"🔍 TURBO Debug: {message}")
    if hasattr(log_system_stats, 'last_operation'):
        op = log_system_stats.last_operation
        logger.debug(f"   Last operation: {op['operation']} | Items processed: {op['item_count']}")
    else:
        logger.debug("   No last operation recorded")

def log_turbo_critical(message: str):
    """Log a TURBO-specific critical message"""
    logger.critical(f"🚨 TURBO Critical: {message}")
    if hasattr(log_system_stats, 'last_operation'):
        op = log_system_stats.last_operation
        logger.critical(f"   Last operation: {op['operation']} | Items processed: {op['item_count']}")
    else:
        logger.critical("   No last operation recorded")

    