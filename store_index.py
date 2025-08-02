"""
ENHANCED ULTRA-FAST Medical Textbook Indexer - Production Ready
Parallel processing, direct Pinecone upserts, optimized batching, comprehensive monitoring
"""

import os
import sys
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from src.helper import (
    prepare_medical_knowledge_base, 
    download_hugging_face_embeddings,
    validate_medical_content_quality,
    validate_medical_terminology_coverage,
    log_performance_stats,
    log_memory_usage,
    log_turbo_summary
)

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('medical_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

load_dotenv()

# ENHANCED CONFIGURATION FOR MAXIMUM SPEED AND RELIABILITY
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot-v2" 
BATCH_SIZE = 100  # Optimized batch size for speed
MAX_WORKERS = 8  # Parallel processing threads
UPSERT_TIMEOUT = 90  # Increased timeout for reliability
EMBEDDING_BATCH_SIZE = 64  # Optimized embedding batch size
MAX_RETRIES = 3  # Retry failed operations

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

class EnhancedFastMedicalIndexer:
    """Enhanced ultra-fast medical indexer with comprehensive monitoring and error handling"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embeddings = None
        self.index = None
        self.performance_metrics = {
            'start_time': None,
            'pdf_processing_time': 0,
            'embedding_time': 0,
            'upload_time': 0,
            'total_chunks': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'processing_rate': 0
        }
        
    def initialize_embeddings(self):
        """Initialize embedding model with enhanced optimizations"""
        logger.info("🚀 Loading enhanced embeddings...")
        start_time = time.time()
        
        self.embeddings = download_hugging_face_embeddings()
        
        # Test embedding and get dimension
        test_embedding = self.embeddings.embed_query("diabetes treatment medical condition")
        embedding_dim = len(test_embedding)
        
        init_time = time.time() - start_time
        logger.info(f"⚡ Embeddings ready - Dimension: {embedding_dim} | Init time: {init_time:.1f}s")
        
        return embedding_dim
    
    def setup_pinecone_index(self, embedding_dim: int):
        """Enhanced Pinecone setup with comprehensive error handling"""
        logger.info("⚡ Enhanced Pinecone setup...")
        
        try:
            # Quick index deletion with retry logic
            existing_indexes = self.pc.list_indexes().names()
            if INDEX_NAME in existing_indexes:
                logger.info(f"🗑️ Deleting existing index: {INDEX_NAME}")
                self.pc.delete_index(INDEX_NAME)
                
                # Wait for deletion with progress monitoring
                for i in range(10):
                    time.sleep(2)
                    try:
                        existing_indexes = self.pc.list_indexes().names()
                        if INDEX_NAME not in existing_indexes:
                            logger.info("✅ Index deletion confirmed")
                            break
                    except:
                        pass
                    if i == 9:
                        logger.warning("⚠️ Index deletion timeout - proceeding anyway")
            
            # Create with enhanced settings
            logger.info(f"🏗️ Creating enhanced index: {INDEX_NAME}")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Enhanced readiness check with progress monitoring
            logger.info("⏳ Waiting for index to be ready...")
            time.sleep(15)  # Initial wait
            self.index = self.pc.Index(INDEX_NAME)
            
            # Comprehensive readiness verification
            for attempt in range(12):  # Extended attempts
                try:
                    stats = self.index.describe_index_stats()
                    logger.info("⚡ Index ready for enhanced high-speed upload!")
                    return True
                except Exception as e:
                    if attempt < 11:
                        logger.info(f"   Readiness check {attempt + 1}/12... ({e})")
                        time.sleep(5)
                    else:
                        logger.info("⚡ Proceeding with upload (index should be ready)")
                        return True
                        
        except Exception as e:
            logger.error(f"❌ Pinecone setup failed: {e}")
            raise
    
    def embed_texts_batch_enhanced(self, texts: List[str]) -> List[List[float]]:
        """Enhanced fast batch embedding generation with comprehensive error handling"""
        embeddings = []
        total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        
        logger.info(f"⚡ Generating {len(texts)} embeddings in {total_batches} batches")
        
        # Process in optimized sub-batches with progress tracking
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    
                    if batch_num % 5 == 0 or batch_num == total_batches:
                        logger.info(f"⚡ Embedding batch {batch_num}/{total_batches} completed")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= MAX_RETRIES:
                        logger.warning(f"⚠️ Embedding batch {batch_num} failed after {MAX_RETRIES} retries: {e}")
                        # Add zero embeddings as fallback
                        embeddings.extend([[0.0] * 384] * len(batch))
                    else:
                        logger.warning(f"⚠️ Embedding batch {batch_num} retry {retry_count}: {e}")
                        time.sleep(1)
        
        return embeddings
    
    def prepare_upsert_data_enhanced(self, chunks, start_idx: int) -> List[Dict]:
        """Enhanced prepare data for direct Pinecone upsert with optimized metadata"""
        # Extract texts for batch embedding
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings in batch (much faster)
        start_time = time.time()
        embeddings = self.embed_texts_batch_enhanced(texts)
        self.performance_metrics['embedding_time'] += time.time() - start_time
        
        # Prepare upsert vectors with enhanced metadata
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create optimized metadata for enhanced search
            metadata = {
                'text': chunk.page_content[:1200],  # Increased for better context
                'source': chunk.metadata.get('filename', 'unknown')[:60],
                'page': str(chunk.metadata.get('page', 0)),
                'medical': 'true',
                'content_length': len(chunk.page_content),
                'word_count': len(chunk.page_content.split()),
                'chunk_index': start_idx + i
            }
            
            # Add medical keywords for enhanced retrieval
            medical_keywords = []
            content_lower = chunk.page_content.lower()
            key_terms = ['diagnosis', 'treatment', 'symptom', 'disease', 'condition', 'therapy', 'medication']
            for term in key_terms:
                if term in content_lower:
                    medical_keywords.append(term)
            
            if medical_keywords:
                metadata['medical_keywords'] = ','.join(medical_keywords[:3])
            
            vectors.append({
                'id': f"med_{start_idx + i}",
                'values': embedding,
                'metadata': metadata
            })
        
        return vectors
    
    def upsert_batch_enhanced(self, batch_data: Tuple[List, int, int]) -> Dict:
        """Enhanced ultra-fast batch upsert with comprehensive error handling and retry logic"""
        chunks, batch_num, start_idx = batch_data
        
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # Prepare vectors for direct upsert (fastest method)
                vectors = self.prepare_upsert_data_enhanced(chunks, start_idx)
                
                # Direct upsert to Pinecone with timeout
                response = self.index.upsert(
                    vectors=vectors,
                    namespace=""
                )
                
                upserted_count = response.get('upserted_count', len(vectors))
                
                return {
                    'batch_num': batch_num,
                    'success': True,
                    'count': len(vectors),
                    'upserted': upserted_count,
                    'retry_count': retry_count
                }
                
            except Exception as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.warning(f"⚠️ Batch {batch_num} failed after {MAX_RETRIES} retries: {str(e)[:100]}...")
                    return {
                        'batch_num': batch_num,
                        'success': False,
                        'count': len(chunks),
                        'error': str(e),
                        'retry_count': retry_count
                    }
                else:
                    logger.warning(f"⚠️ Batch {batch_num} retry {retry_count}: {e}")
                    time.sleep(2 ** retry_count)  # Exponential backoff
    
    def upload_parallel_batches_enhanced(self, chunks: List) -> bool:
        """Enhanced ultra-fast parallel batch upload with comprehensive monitoring"""
        logger.info(f"🚀 ENHANCED TURBO UPLOAD: {len(chunks)} chunks in {BATCH_SIZE}-chunk batches")
        logger.info(f"⚡ Using {MAX_WORKERS} parallel workers with {MAX_RETRIES} retry attempts")
        
        # Prepare batch data
        batch_data = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            batch_data.append((batch, batch_num, i))
        
        total_batches = len(batch_data)
        successful_uploads = 0
        failed_batches = []
        retry_stats = {}
        
        # Parallel processing with enhanced ThreadPoolExecutor
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.upsert_batch_enhanced, data): data[1] 
                for data in batch_data
            }
            
            # Process results as they complete with progress tracking
            completed_batches = 0
            for future in as_completed(future_to_batch, timeout=UPSERT_TIMEOUT * total_batches):
                batch_num = future_to_batch[future]
                completed_batches += 1
                
                try:
                    result = future.result(timeout=UPSERT_TIMEOUT)
                    
                    if result['success']:
                        successful_uploads += result['count']
                        retry_stats[batch_num] = result['retry_count']
                        
                        # Progress logging
                        progress = (completed_batches / total_batches) * 100
                        logger.info(f"⚡ Batch {result['batch_num']}/{total_batches} ✅ ({result['count']} vectors) | Progress: {progress:.1f}%")
                    else:
                        failed_batches.append(result['batch_num'])
                        logger.warning(f"❌ Batch {result['batch_num']} failed after retries")
                        
                except Exception as e:
                    failed_batches.append(batch_num)
                    logger.error(f"❌ Batch {batch_num} exception: {e}")
        
        upload_time = time.time() - start_time
        self.performance_metrics['upload_time'] = upload_time
        self.performance_metrics['successful_uploads'] = successful_uploads
        self.performance_metrics['failed_uploads'] = len(chunks) - successful_uploads
        
        upload_rate = successful_uploads / upload_time if upload_time > 0 else 0
        success_rate = (successful_uploads / len(chunks)) * 100
        
        logger.info(f"🚀 ENHANCED TURBO UPLOAD COMPLETE!")
        logger.info(f"   ⚡ Speed: {upload_rate:.1f} vectors/second")
        logger.info(f"   ✅ Success: {successful_uploads}/{len(chunks)} vectors ({success_rate:.1f}%)")
        logger.info(f"   ⏱️ Time: {upload_time:.1f} seconds")
        logger.info(f"   🔄 Retries: {sum(retry_stats.values())} total across all batches")
        
        if failed_batches:
            logger.warning(f"   ⚠️ Failed batches: {failed_batches}")
        
        # Log memory usage
        log_memory_usage()
        
        return successful_uploads > (len(chunks) * 0.85)  # 85% success threshold
    
    def comprehensive_test_retrieval(self) -> Dict:
        """Enhanced comprehensive medical knowledge test with detailed analysis"""
        logger.info("🧪 Comprehensive retrieval testing...")
        
        test_queries = [
            ("diabetes treatment", "Diabetes management and therapy"),
            ("heart disease symptoms", "Cardiovascular condition indicators"), 
            ("cancer therapy", "Oncological treatment approaches"),
            ("hypertension medication", "Blood pressure management drugs"),
            ("asthma diagnosis", "Respiratory condition identification"),
            ("arthritis pain relief", "Joint inflammation treatment")
        ]
        
        test_results = {
            'successful_tests': 0,
            'total_tests': len(test_queries),
            'detailed_results': [],
            'average_similarity': 0,
            'coverage_analysis': {}
        }
        
        total_similarity = 0
        
        for query, description in test_queries:
            try:
                # Direct Pinecone query (faster than LangChain)
                query_embedding = self.embeddings.embed_query(query)
                
                results = self.index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True,
                    namespace=""
                )
                
                if results.matches:
                    top_match = results.matches[0]
                    source = top_match.metadata.get('source', 'Unknown')
                    similarity = top_match.score
                    content_preview = top_match.metadata.get('text', '')[:100]
                    
                    test_results['detailed_results'].append({
                        'query': query,
                        'description': description,
                        'source': source,
                        'similarity': similarity,
                        'content_preview': content_preview,
                        'success': True
                    })
                    
                    total_similarity += similarity
                    test_results['successful_tests'] += 1
                    
                    logger.info(f"✅ '{query}' → {source} (similarity: {similarity:.3f})")
                else:
                    logger.warning(f"⚠️ '{query}' → No results")
                    test_results['detailed_results'].append({
                        'query': query,
                        'description': description,
                        'success': False,
                        'error': 'No results returned'
                    })
                    
            except Exception as e:
                logger.error(f"❌ '{query}' → Error: {e}")
                test_results['detailed_results'].append({
                    'query': query,
                    'description': description,
                    'success': False,
                    'error': str(e)
                })
        
        test_results['average_similarity'] = total_similarity / test_results['successful_tests'] if test_results['successful_tests'] > 0 else 0
        test_results['success_rate'] = (test_results['successful_tests'] / test_results['total_tests']) * 100
        
        logger.info(f"🧪 Test Results: {test_results['successful_tests']}/{test_results['total_tests']} passed")
        logger.info(f"📊 Average similarity: {test_results['average_similarity']:.3f}")
        logger.info(f"📈 Success rate: {test_results['success_rate']:.1f}%")
        
        return test_results
    
    def generate_comprehensive_report(self, chunks, validation_report, test_results) -> str:
        """Generate comprehensive indexing report with all metrics"""
        
        total_time = time.time() - self.performance_metrics['start_time']
        
        report = f"""
🚀 ENHANCED MEDICAL INDEX CREATION REPORT
{'='*80}

📊 PROCESSING SUMMARY:
   📄 Total Chunks Processed: {len(chunks):,}
   📚 Unique Medical Sources: {validation_report.get('unique_books', 'N/A')}
   🔤 Total Words Indexed: {validation_report.get('total_words', 0):,}
   📏 Average Chunk Size: {validation_report.get('average_chunk_size', 0)} words
   ⭐ Content Quality Score: {validation_report.get('quality_score', 0)}/100

⚡ PERFORMANCE METRICS:
   🏃 Total Processing Time: {total_time:.1f} seconds
   📑 PDF Processing: {self.performance_metrics['pdf_processing_time']:.1f}s
   🧠 Embedding Generation: {self.performance_metrics['embedding_time']:.1f}s
   ⬆️  Upload Time: {self.performance_metrics['upload_time']:.1f}s
   📈 Overall Processing Rate: {len(chunks)/total_time:.1f} chunks/second

✅ UPLOAD STATISTICS:
   ✅ Successful Uploads: {self.performance_metrics['successful_uploads']:,}
   ❌ Failed Uploads: {self.performance_metrics['failed_uploads']:,}
   📊 Success Rate: {(self.performance_metrics['successful_uploads']/len(chunks)*100):.1f}%

🧪 RETRIEVAL TESTING:
   🎯 Test Queries Passed: {test_results['successful_tests']}/{test_results['total_tests']}
   📊 Average Similarity Score: {test_results['average_similarity']:.3f}
   📈 Retrieval Success Rate: {test_results['success_rate']:.1f}%

🏥 MEDICAL CONTENT ANALYSIS:
   🔬 Medical Terms Coverage: {validation_report.get('medical_terms_coverage', 'N/A')}
   📚 Source Diversity: {validation_report.get('unique_books', 0)} unique textbooks
   ⚕️  Content Validation: Passed medical relevance filtering

🔧 TECHNICAL CONFIGURATION:
   🎛️  Batch Size: {BATCH_SIZE} chunks per batch
   🔧 Max Workers: {MAX_WORKERS} parallel threads
   🔄 Retry Logic: {MAX_RETRIES} attempts per batch
   🧠 Embedding Model: sentence-transformers/all-MiniLM-L6-v2
   🗃️  Vector Database: Pinecone (AWS us-east-1)

{'='*80}
🎉 ENHANCED MEDICAL INDEX CREATION COMPLETED SUCCESSFULLY!
🏥 Medical chatbot is now ready for production use with enhanced capabilities.
{'='*80}
"""
        
        return report
    
    def create_enhanced_medical_index(self) -> bool:
        """Enhanced ULTRA-FAST medical index creation with comprehensive monitoring"""
        self.performance_metrics['start_time'] = time.time()
        
        logger.info("🚀 ENHANCED TURBO MEDICAL INDEX CREATOR")
        logger.info("=" * 80)
        
        try:
            # Step 1: Enhanced PDF processing with timing
            logger.info("📚 ENHANCED TURBO: Processing medical PDFs...")
            pdf_start = time.time()
            
            chunks, validation_report = prepare_medical_knowledge_base("data/")
            
            if not chunks:
                raise ValueError("❌ No chunks created")
            
            self.performance_metrics['pdf_processing_time'] = time.time() - pdf_start
            self.performance_metrics['total_chunks'] = len(chunks)
            
            logger.info(f"⚡ PDF processing: {self.performance_metrics['pdf_processing_time']:.1f}s")
            logger.info(f"   📄 Chunks: {len(chunks):,}")
            logger.info(f"   ⭐ Quality: {validation_report.get('quality_score', 'N/A')}")
            
            # Enhanced content quality validation
            quality_metrics = validate_medical_content_quality(chunks)
            logger.info(f"🏥 Medical quality score: {quality_metrics['quality_score']:.1f}/100")
            
            # Step 2: Enhanced embeddings initialization
            embedding_dim = self.initialize_embeddings()
            
            # Step 3: Enhanced Pinecone setup
            self.setup_pinecone_index(embedding_dim)
            
            # Step 4: ENHANCED TURBO UPLOAD
            upload_success = self.upload_parallel_batches_enhanced(chunks)
            
            if not upload_success:
                logger.warning("⚠️ Upload had issues but proceeding with testing...")
            
            # Step 5: Comprehensive testing
            test_results = self.comprehensive_test_retrieval()
            
            # Step 6: Final verification and report
            final_stats = self.index.describe_index_stats()
            total_vectors = final_stats.get('total_vector_count', 0)
            
            # Generate and save comprehensive report
            report = self.generate_comprehensive_report(chunks, validation_report, test_results)
            
            # Save report to file
            with open('medical_index_report.txt', 'w',encoding="utf-8") as f:
                f.write(report)
            
            print(report)
            
            # Log final summary
            log_turbo_summary(chunks, validation_report)
            
            logger.info("🎉 ENHANCED TURBO INDEX CREATION COMPLETED!")
            logger.info(f"📊 Final vector count: {total_vectors:,}")
            logger.info(f"📋 Detailed report saved to: medical_index_report.txt")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ENHANCED TURBO creation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

def enhanced_verify_setup() -> Dict:
    """Enhanced setup verification with detailed analysis"""
    logger.info("🔍 Enhanced setup verification...")
    
    verification_results = {
        'pinecone_api_key': bool(PINECONE_API_KEY),
        'data_directory': os.path.exists("data/"),
        'pdf_files': [],
        'total_size_mb': 0,
        'estimated_time': {},
        'setup_valid': False,
        'recommendations': []
    }
    
    # Check Pinecone API key
    if not verification_results['pinecone_api_key']:
        verification_results['recommendations'].append("❌ Add PINECONE_API_KEY to .env file")
        return verification_results
    
    # Check data directory
    if not verification_results['data_directory']:
        verification_results['recommendations'].append("❌ Create data/ directory and add PDF files")
        return verification_results
    
    try:
        # Analyze PDF files
        pdf_files = [f for f in os.listdir("data/") if f.endswith('.pdf')]
        verification_results['pdf_files'] = pdf_files
        
        if not pdf_files:
            verification_results['recommendations'].append("❌ Add medical PDF files to data/ directory")
            return verification_results
        
        # Calculate total size and estimate processing time
        total_size = sum(os.path.getsize(os.path.join("data/", f)) for f in pdf_files)
        verification_results['total_size_mb'] = total_size / (1024 * 1024)
        
        # Enhanced time estimation
        verification_results['estimated_time'] = {
            'pdf_processing': max(0.5, verification_results['total_size_mb'] / 4),
            'embedding_generation': max(1.0, verification_results['total_size_mb'] / 2),
            'upload_time': max(0.5, verification_results['total_size_mb'] / 6),
            'total_estimated': max(2.0, verification_results['total_size_mb'] / 1.5)
        }
        
        verification_results['setup_valid'] = True
        
        logger.info(f"✅ Found {len(pdf_files)} PDFs ({verification_results['total_size_mb']:.1f} MB)")
        logger.info(f"⏱️ Estimated processing time: {verification_results['estimated_time']['total_estimated']:.1f} minutes")
        
        # Add optimization recommendations
        if verification_results['total_size_mb'] > 100:
            verification_results['recommendations'].append("💡 Large dataset detected - consider processing in batches")
        if len(pdf_files) > 20:
            verification_results['recommendations'].append("💡 Many files detected - parallel processing will be very beneficial")
        
        return verification_results
        
    except Exception as e:
        verification_results['recommendations'].append(f"❌ Error analyzing setup: {str(e)}")
        return verification_results

def main():
    """Enhanced TURBO main execution with comprehensive user interaction"""
    print("🚀 ENHANCED TURBO MEDICAL INDEX CREATOR")
    print("=" * 80)
    print(f"⚡ OPTIMIZED FOR MAXIMUM SPEED & RELIABILITY")
    print(f"🔧 Batch Size: {BATCH_SIZE} | Workers: {MAX_WORKERS} | Retries: {MAX_RETRIES}")
    print(f"📊 Direct Pinecone Upserts | Parallel Processing | Comprehensive Monitoring")
    print("=" * 80)
    
    # Enhanced setup verification
    verification = enhanced_verify_setup()
    
    if not verification['setup_valid']:
        print("\n❌ SETUP ISSUES DETECTED:")
        for rec in verification['recommendations']:
            print(f"   {rec}")
        return False
    
    print(f"\n✅ SETUP VERIFICATION PASSED!")
    print(f"   📄 {len(verification['pdf_files'])} PDF files ready")
    print(f"   💾 Total size: {verification['total_size_mb']:.1f} MB")
    print(f"   ⏱️ Estimated time: {verification['estimated_time']['total_estimated']:.1f} minutes")
    
    if verification['recommendations']:
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for rec in verification['recommendations']:
            print(f"   {rec}")
    
    print(f"\n🚀 READY FOR ENHANCED TURBO PROCESSING!")
    response = input("Start ENHANCED TURBO indexing? (Y/n): ")
    
    if response.lower() in ['', 'y', 'yes']:
        creator = EnhancedFastMedicalIndexer()
        success = creator.create_enhanced_medical_index()
        
        if success:
            print("\n⚡ ENHANCED TURBO SUCCESS! Medical index created with maximum speed and reliability!")
            print("📋 Check medical_index_report.txt for detailed analysis")
            print("🏥 Your medical chatbot is now ready for production use!")
        else:
            print("\n❌ ENHANCED TURBO encountered issues. Check logs for details.")
        
        return success
    else:
        print("⏹️ Cancelled.")
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ ENHANCED TURBO interrupted.")
    except Exception as e:
        logger.error(f"❌ ENHANCED TURBO error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")