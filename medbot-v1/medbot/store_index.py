"""
ULTRA-FAST Medical Textbook Indexer - Optimized for Speed
Parallel processing, direct Pinecone upserts, larger batches
"""

import os
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from src.helper import prepare_medical_knowledge_base, download_hugging_face_embeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

load_dotenv()

# OPTIMIZED CONFIGURATION FOR SPEED
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot-v2" 
BATCH_SIZE = 80  # Larger batches for speed
MAX_WORKERS = 8  # Parallel processing threads
UPSERT_TIMEOUT = 60  # Seconds
EMBEDDING_BATCH_SIZE = 32  # Process embeddings in batches

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

class FastMedicalIndexer:
    """Ultra-fast medical indexer with parallel processing"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.embeddings = None
        self.index = None
        
    def initialize_embeddings(self):
        """Initialize embedding model with optimizations"""
        logger.info("🚀 Loading optimized embeddings...")
        self.embeddings = download_hugging_face_embeddings()
        
        # Test embedding
        test_embedding = self.embeddings.embed_query("diabetes treatment")
        embedding_dim = len(test_embedding)
        logger.info(f"⚡ Embeddings ready - Dimension: {embedding_dim}")
        return embedding_dim
    
    def setup_pinecone_index(self, embedding_dim: int):
        """Fast Pinecone setup with minimal wait times"""
        logger.info("⚡ Fast Pinecone setup...")
        
        # Quick index deletion
        existing_indexes = self.pc.list_indexes().names()
        if INDEX_NAME in existing_indexes:
            logger.info(f"🗑️ Deleting existing index...")
            self.pc.delete_index(INDEX_NAME)
            time.sleep(5)  # Reduced wait time
        
        # Create with optimized settings
        logger.info(f"🏗️ Creating index: {INDEX_NAME}")
        self.pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Minimal wait - check readiness quickly
        time.sleep(10)  # Reduced initial wait
        self.index = self.pc.Index(INDEX_NAME)
        
        # Quick readiness check
        for attempt in range(6):  # Reduced attempts
            try:
                self.index.describe_index_stats()
                logger.info("⚡ Index ready for high-speed upload!")
                return True
            except:
                if attempt < 5:
                    logger.info(f"   Quick check {attempt + 1}/6...")
                    time.sleep(3)  # Faster retry
                else:
                    logger.info("⚡ Proceeding with upload (index should be ready)")
                    return True
    
    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Fast batch embedding generation"""
        embeddings = []
        
        # Process in optimized sub-batches
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def prepare_upsert_data(self, chunks, start_idx: int) -> List[Dict]:
        """Prepare data for direct Pinecone upsert (fastest method)"""
        # Extract texts for batch embedding
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings in batch (much faster)
        embeddings = self.embed_texts_batch(texts)
        
        # Prepare upsert vectors
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create minimal metadata for speed
            metadata = {
                'text': chunk.page_content[:1000],  # Truncate to save space
                'source': chunk.metadata.get('filename', 'unknown')[:50],
                'page': str(chunk.metadata.get('page', 0)),
                'medical': 'true'
            }
            
            vectors.append({
                'id': f"med_{start_idx + i}",
                'values': embedding,
                'metadata': metadata
            })
        
        return vectors
    
    def upsert_batch_fast(self, batch_data: Tuple[List, int, int]) -> Dict:
        """Ultra-fast batch upsert with minimal overhead"""
        chunks, batch_num, start_idx = batch_data
        
        try:
            # Prepare vectors for direct upsert (fastest method)
            vectors = self.prepare_upsert_data(chunks, start_idx)
            
            # Direct upsert to Pinecone (bypasses LangChain overhead)
            response = self.index.upsert(
                vectors=vectors,
                namespace=""
            )
            
            return {
                'batch_num': batch_num,
                'success': True,
                'count': len(vectors),
                'upserted': response.get('upserted_count', len(vectors))
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Batch {batch_num} failed: {str(e)[:100]}...")
            return {
                'batch_num': batch_num,
                'success': False,
                'count': len(chunks),
                'error': str(e)
            }
    
    def upload_parallel_batches(self, chunks: List) -> bool:
        """Ultra-fast parallel batch upload"""
        logger.info(f"🚀 TURBO UPLOAD: {len(chunks)} chunks in {BATCH_SIZE}-chunk batches")
        logger.info(f"⚡ Using {MAX_WORKERS} parallel workers")
        
        # Prepare batch data
        batch_data = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            batch_data.append((batch, batch_num, i))
        
        total_batches = len(batch_data)
        successful_uploads = 0
        failed_batches = []
        
        # Parallel processing with ThreadPoolExecutor
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.upsert_batch_fast, data): data[1] 
                for data in batch_data
            }
            
            # Process results as they complete
            for future in as_completed(future_to_batch, timeout=UPSERT_TIMEOUT * total_batches):
                batch_num = future_to_batch[future]
                
                try:
                    result = future.result(timeout=UPSERT_TIMEOUT)
                    
                    if result['success']:
                        successful_uploads += result['count']
                        logger.info(f"⚡ Batch {result['batch_num']}/{total_batches} ✅ ({result['count']} vectors)")
                    else:
                        failed_batches.append(result['batch_num'])
                        logger.warning(f"❌ Batch {result['batch_num']} failed")
                        
                except Exception as e:
                    failed_batches.append(batch_num)
                    logger.error(f"❌ Batch {batch_num} error: {e}")
        
        elapsed = time.time() - start_time
        upload_rate = successful_uploads / elapsed if elapsed > 0 else 0
        
        logger.info(f"🚀 TURBO UPLOAD COMPLETE!")
        logger.info(f"   ⚡ Speed: {upload_rate:.1f} vectors/second")
        logger.info(f"   ✅ Success: {successful_uploads}/{len(chunks)} vectors")
        logger.info(f"   ⏱️ Time: {elapsed:.1f} seconds")
        
        if failed_batches:
            logger.warning(f"   ⚠️ Failed batches: {failed_batches}")
        
        return successful_uploads > (len(chunks) * 0.8)  # 80% success threshold
    
    def quick_test_retrieval(self) -> int:
        """Quick medical knowledge test"""
        logger.info("🧪 Quick retrieval test...")
        
        test_queries = [
            "diabetes treatment",
            "heart disease symptoms", 
            "cancer therapy"
        ]
        
        successful_tests = 0
        
        for query in test_queries:
            try:
                # Direct Pinecone query (faster than LangChain)
                query_embedding = self.embeddings.embed_query(query)
                
                results = self.index.query(
                    vector=query_embedding,
                    top_k=1,
                    include_metadata=True,
                    namespace=""
                )
                
                if results.matches:
                    match = results.matches[0]
                    source = match.metadata.get('source', 'Unknown')
                    score = match.score
                    logger.info(f"✅ '{query}' → {source} (score: {score:.3f})")
                    successful_tests += 1
                else:
                    logger.warning(f"⚠️ '{query}' → No results")
                    
            except Exception as e:
                logger.error(f"❌ '{query}' → Error: {e}")
        
        return successful_tests
    
    def create_medical_index_fast(self) -> bool:
        """ULTRA-FAST medical index creation"""
        logger.info("🚀 TURBO MEDICAL INDEX CREATOR")
        logger.info("=" * 60)
        
        try:
            # Step 1: Fast PDF processing
            logger.info("📚 TURBO: Processing medical PDFs...")
            start_time = time.time()
            
            chunks, validation_report = prepare_medical_knowledge_base("data/")
            
            if not chunks:
                raise ValueError("❌ No chunks created")
            
            process_time = time.time() - start_time
            logger.info(f"⚡ PDF processing: {process_time:.1f}s")
            logger.info(f"   📄 Chunks: {len(chunks)}")
            logger.info(f"   ⭐ Quality: {validation_report.get('quality_score', 'N/A')}")
            
            # Step 2: Fast embeddings
            embedding_dim = self.initialize_embeddings()
            
            # Step 3: Fast Pinecone setup
            self.setup_pinecone_index(embedding_dim)
            
            # Step 4: TURBO UPLOAD
            upload_success = self.upload_parallel_batches(chunks)
            
            if not upload_success:
                raise Exception("Upload failed - too many batch failures")
            
            # Step 5: Quick test
            successful_tests = self.quick_test_retrieval()
            
            # Final stats
            final_stats = self.index.describe_index_stats()
            total_vectors = final_stats.get('total_vector_count', 0)
            
            total_time = time.time() - start_time
            
            logger.info("🎉 TURBO INDEX CREATION COMPLETED!")
            logger.info("=" * 60)
            logger.info("⚡ PERFORMANCE STATS:")
            logger.info(f"   🚀 Total Time: {total_time:.1f} seconds")
            logger.info(f"   📄 Total Vectors: {total_vectors}")
            logger.info(f"   ⚡ Processing Speed: {total_vectors/total_time:.1f} vectors/sec")
            logger.info(f"   🧪 Tests Passed: {successful_tests}/3")
            logger.info(f"   🎯 Batch Size: {BATCH_SIZE}")
            logger.info(f"   🔧 Workers: {MAX_WORKERS}")
            logger.info("=" * 60)
            logger.info("🚀 TURBO MEDICAL CHATBOT READY!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ TURBO creation failed: {e}")
            return False

def verify_setup_fast() -> bool:
    """Quick setup verification"""
    logger.info("🔍 Quick setup check...")
    
    # Essential checks only
    if not PINECONE_API_KEY:
        logger.error("❌ PINECONE_API_KEY missing")
        return False
    
    if not os.path.exists("data/"):
        logger.error("❌ data/ directory missing")
        return False
    
    pdf_files = [f for f in os.listdir("data/") if f.endswith('.pdf')]
    if not pdf_files:
        logger.error("❌ No PDFs in data/")
        return False
    
    logger.info(f"✅ {len(pdf_files)} PDFs ready for TURBO processing")
    return True

def main():
    """TURBO main execution"""
    print("🚀 TURBO MEDICAL INDEX CREATOR")
    print("=" * 60)
    print(f"⚡ Optimized for MAXIMUM SPEED")
    print(f"🔧 Batch Size: {BATCH_SIZE} | Workers: {MAX_WORKERS}")
    print(f"📊 Direct Pinecone Upserts | Parallel Processing")
    print("=" * 60)
    
    if not verify_setup_fast():
        print("❌ Setup failed")
        return False
    
    print(f"\n🚀 READY FOR TURBO UPLOAD!")
    response = input("Start TURBO indexing? (Y/n): ")
    
    if response.lower() in ['', 'y', 'yes']:
        creator = FastMedicalIndexer()
        success = creator.create_medical_index_fast()
        
        if success:
            print("\n⚡ TURBO SUCCESS! Medical index created at maximum speed!")
        else:
            print("\n❌ TURBO failed. Check logs.")
        
        return success
    else:
        print("⏹️ Cancelled.")
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ TURBO interrupted.")
    except Exception as e:
        logger.error(f"❌ TURBO error: {e}")