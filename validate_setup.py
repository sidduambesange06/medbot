#!/usr/bin/env python3
"""
Enhanced Medical Chatbot Validation Script
Validates all components of the enhanced medical chatbot setup
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemValidator:
    def __init__(self):
        self.results = {
            "environment": {"status": "unknown", "details": []},
            "database": {"status": "unknown", "details": []},
            "ai_services": {"status": "unknown", "details": []},
            "vector_store": {"status": "unknown", "details": []},
            "security": {"status": "unknown", "details": []},
            "file_structure": {"status": "unknown", "details": []}
        }
        
    async def run_validation(self):
        """Run complete system validation"""
        logger.info("🔍 Starting Enhanced Medical Chatbot Validation")
        logger.info("=" * 60)
        
        # Run all validation checks
        await self.validate_environment()
        await self.validate_file_structure()
        await self.validate_database_connection()
        await self.validate_ai_services()
        await self.validate_vector_store()
        await self.validate_security_components()
        
        # Generate report
        self.generate_report()
        
    async def validate_environment(self):
        """Validate environment variables and configuration"""
        logger.info("⚙️  Validating Environment Configuration...")
        
        try:
            # Check if python-dotenv is available
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                logger.warning("⚠️  python-dotenv not installed, loading .env manually")
                self.load_env_file()
            
            required_vars = {
                "SUPABASE_URL": "Supabase project URL",
                "SUPABASE_KEY": "Supabase anon key",
                "JWT_SECRET": "JWT secret for authentication",
            }
            
            optional_vars = {
                "GROQ_API_KEY": "Groq API for LLM",
                "OPENAI_API_KEY": "OpenAI API for LLM",
                "PINECONE_API_KEY": "Pinecone for vector storage",
                "PINECONE_ENVIRONMENT": "Pinecone environment",
                "PINECONE_INDEX_NAME": "Pinecone index name"
            }
            
            missing_required = []
            missing_optional = []
            valid_vars = []
            
            # Check required variables
            for var, description in required_vars.items():
                value = os.getenv(var)
                if not value or value.startswith('your-'):
                    missing_required.append(f"{var} ({description})")
                else:
                    valid_vars.append(f"{var}: ✅ Set")
            
            # Check optional variables
            for var, description in optional_vars.items():
                value = os.getenv(var)
                if not value or value.startswith('your-'):
                    missing_optional.append(f"{var} ({description})")
                else:
                    valid_vars.append(f"{var}: ✅ Set")
            
            self.results["environment"]["details"] = valid_vars
            
            if missing_required:
                self.results["environment"]["status"] = "failed"
                self.results["environment"]["details"].extend([
                    "❌ Missing required variables:",
                    *[f"  - {var}" for var in missing_required]
                ])
            elif missing_optional:
                self.results["environment"]["status"] = "warning"
                self.results["environment"]["details"].extend([
                    "⚠️  Missing optional variables:",
                    *[f"  - {var}" for var in missing_optional]
                ])
            else:
                self.results["environment"]["status"] = "success"
            
            logger.info("✅ Environment validation completed")
            
        except Exception as e:
            self.results["environment"]["status"] = "failed"
            self.results["environment"]["details"] = [f"❌ Environment validation error: {e}"]
            logger.error(f"❌ Environment validation failed: {e}")
    
    def load_env_file(self):
        """Manually load .env file if python-dotenv is not available"""
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
    
    async def validate_file_structure(self):
        """Validate required files and directories exist"""
        logger.info("📁 Validating File Structure...")
        
        try:
            required_files = {
                "app.py": "Main Flask application",
                "store_index.py": "Document indexing script",
                "requirements.txt": "Python dependencies",
                "src/helper.py": "Helper functions",
                "src/prompt.py": "Prompt engineering",
                "src/chat_service.py": "Enhanced chat service",
                "src/security.py": "Security manager",
                ".env": "Environment variables"
            }
            
            required_dirs = [
                "src",
                "data",
                "logs",
                "static",
                "templates"
            ]
            
            existing_files = []
            missing_files = []
            existing_dirs = []
            missing_dirs = []
            
            # Check files
            for file_path, description in required_files.items():
                if Path(file_path).exists():
                    existing_files.append(f"{file_path}: ✅ Found")
                else:
                    missing_files.append(f"{file_path} ({description})")
            
            # Check directories
            for dir_path in required_dirs:
                if Path(dir_path).exists():
                    existing_dirs.append(f"{dir_path}/: ✅ Found")
                else:
                    missing_dirs.append(f"{dir_path}/")
            
            self.results["file_structure"]["details"] = existing_files + existing_dirs
            
            if missing_files or missing_dirs:
                self.results["file_structure"]["status"] = "failed"
                if missing_files:
                    self.results["file_structure"]["details"].extend([
                        "❌ Missing files:",
                        *[f"  - {file}" for file in missing_files]
                    ])
                if missing_dirs:
                    self.results["file_structure"]["details"].extend([
                        "❌ Missing directories:",
                        *[f"  - {dir}" for dir in missing_dirs]
                    ])
            else:
                self.results["file_structure"]["status"] = "success"
            
            logger.info("✅ File structure validation completed")
            
        except Exception as e:
            self.results["file_structure"]["status"] = "failed"
            self.results["file_structure"]["details"] = [f"❌ File structure validation error: {e}"]
            logger.error(f"❌ File structure validation failed: {e}")
    
    async def validate_database_connection(self):
        """Validate Supabase database connection and schema"""
        logger.info("🗄️  Validating Database Connection...")
        
        try:
            try:
                from supabase import create_client
            except ImportError:
                self.results["database"]["status"] = "failed"
                self.results["database"]["details"] = ["❌ supabase-py not installed. Run: pip install supabase"]
                return
            
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                self.results["database"]["status"] = "failed"
                self.results["database"]["details"] = ["❌ Missing Supabase credentials"]
                return
            
            supabase = create_client(supabase_url, supabase_key)
            
            # Test basic connection
            result = supabase.table("users").select("id").limit(1).execute()
            
            # Check required tables
            required_tables = [
                "users",
                "chat_history", 
                "conversation_summaries",
                "user_context_preferences",
                "medical_documents",
                "medical_knowledge_chunks"
            ]
            
            existing_tables = []
            missing_tables = []
            
            for table in required_tables:
                try:
                    supabase.table(table).select("*").limit(1).execute()
                    existing_tables.append(f"{table}: ✅ Accessible")
                except Exception:
                    missing_tables.append(table)
            
            self.results["database"]["details"] = [
                "✅ Supabase connection successful",
                *existing_tables
            ]
            
            if missing_tables:
                self.results["database"]["status"] = "warning"
                self.results["database"]["details"].extend([
                    "⚠️  Missing or inaccessible tables:",
                    *[f"  - {table}" for table in missing_tables],
                    "Run setup_database.sql in Supabase SQL Editor"
                ])
            else:
                self.results["database"]["status"] = "success"
            
            logger.info("✅ Database validation completed")
            
        except Exception as e:
            self.results["database"]["status"] = "failed"
            self.results["database"]["details"] = [f"❌ Database connection error: {e}"]
            logger.error(f"❌ Database validation failed: {e}")
    
    async def validate_ai_services(self):
        """Validate AI service connections"""
        logger.info("🤖 Validating AI Services...")
        
        try:
            ai_services = []
            working_services = []
            failed_services = []
            
            # Check Groq
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key and not groq_key.startswith('your-'):
                try:
                    from groq import Groq
                    client = Groq(api_key=groq_key)
                    # Test with a simple completion
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=5
                    )
                    working_services.append("Groq: ✅ Connected")
                except ImportError:
                    failed_services.append("Groq: ❌ groq library not installed")
                except Exception as e:
                    failed_services.append(f"Groq: ❌ Connection failed - {str(e)[:50]}...")
            
            # Check OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and not openai_key.startswith('your-'):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_key)
                    # Test with a simple completion
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=5
                    )
                    working_services.append("OpenAI: ✅ Connected")
                except ImportError:
                    failed_services.append("OpenAI: ❌ openai library not installed")
                except Exception as e:
                    failed_services.append(f"OpenAI: ❌ Connection failed - {str(e)[:50]}...")
            
            self.results["ai_services"]["details"] = working_services + failed_services
            
            if not working_services:
                self.results["ai_services"]["status"] = "failed"
                self.results["ai_services"]["details"].insert(0, "❌ No AI services available")
            elif failed_services:
                self.results["ai_services"]["status"] = "warning"
            else:
                self.results["ai_services"]["status"] = "success"
            
            logger.info("✅ AI services validation completed")
            
        except Exception as e:
            self.results["ai_services"]["status"] = "failed"
            self.results["ai_services"]["details"] = [f"❌ AI services validation error: {e}"]
            logger.error(f"❌ AI services validation failed: {e}")
    
    async def validate_vector_store(self):
        """Validate vector store connection"""
        logger.info("🔍 Validating Vector Store...")
        
        try:
            vector_services = []
            
            # Check Pinecone
            pinecone_key = os.getenv("PINECONE_API_KEY")
            pinecone_index = os.getenv("PINECONE_INDEX_NAME")
            
            if pinecone_key and pinecone_index:
                try:
                    from pinecone import Pinecone
                    
                    # Initialize Pinecone client (v6+ API)
                    pc = Pinecone(api_key=pinecone_key)
                    
                    # List indexes - new API returns IndexList object
                    indexes_list = pc.list_indexes()
                    
                    # Extract index names from the response
                    if hasattr(indexes_list, 'indexes'):
                        index_names = [idx.name for idx in indexes_list.indexes]
                    elif hasattr(indexes_list, '__iter__'):
                        index_names = [idx.name if hasattr(idx, 'name') else str(idx) for idx in indexes_list]
                    else:
                        index_names = []
                    
                    vector_services.append(f"Pinecone: ✅ Connected (v6+)")
                    vector_services.append(f"  Available indexes: {index_names}")
                    
                    if pinecone_index in index_names:
                        vector_services.append(f"  Index '{pinecone_index}': ✅ Found")
                        
                        # Test index connection
                        try:
                            index = pc.Index(pinecone_index)
                            stats = index.describe_index_stats()
                            total_vectors = stats.get('total_vector_count', 'unknown')
                            vector_services.append(f"  Vectors: {total_vectors}")
                        except Exception as e:
                            vector_services.append(f"  ⚠️  Index accessible but stats failed: {str(e)[:30]}...")
                    else:
                        vector_services.append(f"  Index '{pinecone_index}': ❌ Not found")
                        vector_services.append("  Create it at: https://app.pinecone.io/")
                        
                except ImportError:
                    vector_services.append("Pinecone: ❌ pinecone package not installed")
                    vector_services.append("  Install with: pip install pinecone")
                except Exception as e:
                    error_msg = str(e)
                    if "renamed" in error_msg.lower():
                        vector_services.append("Pinecone: ❌ Package version mismatch")
                        vector_services.append("  Try: pip uninstall pinecone-client && pip install pinecone")
                    else:
                        vector_services.append(f"Pinecone: ❌ Connection failed - {error_msg[:60]}...")
                        
            else:
                if not pinecone_key:
                    vector_services.append("Pinecone: ❌ PINECONE_API_KEY not set")
                if not pinecone_index:
                    vector_services.append("Pinecone: ❌ PINECONE_INDEX_NAME not set")
            
            # Check local vector store (if implemented)
            if Path("data/vector_store.pkl").exists():
                vector_services.append("Local Vector Store: ✅ File found")
            
            self.results["vector_store"]["details"] = vector_services
            
            if not vector_services:
                self.results["vector_store"]["status"] = "warning"
                self.results["vector_store"]["details"] = ["⚠️  No vector store configured"]
            elif any("❌" in service for service in vector_services):
                self.results["vector_store"]["status"] = "failed"
            else:
                self.results["vector_store"]["status"] = "success"
            
            logger.info("✅ Vector store validation completed")
            
        except Exception as e:
            self.results["vector_store"]["status"] = "failed"
            self.results["vector_store"]["details"] = [f"❌ Vector store validation error: {e}"]
            logger.error(f"❌ Vector store validation failed: {e}")
    
    async def validate_security_components(self):
        """Validate security components"""
        logger.info("🔐 Validating Security Components...")
        
        try:
            security_checks = []
            
            # Check JWT secret
            jwt_secret = os.getenv("JWT_SECRET")
            if jwt_secret and len(jwt_secret) >= 32:
                security_checks.append("JWT Secret: ✅ Set and sufficient length")
            else:
                security_checks.append("JWT Secret: ❌ Missing or too short (min 32 chars)")
            
            # Check if security.py exists and has required functions
            security_file = Path("src/security.py")
            if security_file.exists():
                security_checks.append("Security module: ✅ Found")
                
                # Check for required security functions
                try:
                    with open(security_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        required_functions = [
                            "hash_password",
                            "verify_password", 
                            "generate_token",
                            "verify_token",
                            "sanitize_input"
                        ]
                        
                        for func in required_functions:
                            if f"def {func}" in content:
                                security_checks.append(f"  {func}: ✅ Found")
                            else:
                                security_checks.append(f"  {func}: ❌ Missing")
                                
                except Exception as e:
                    security_checks.append(f"  ❌ Error reading security.py: {e}")
            else:
                security_checks.append("Security module: ❌ Not found")
            
            # Check SSL/HTTPS configuration hints
            if os.getenv("FLASK_ENV") == "production":
                security_checks.append("Environment: ✅ Set to production")
            else:
                security_checks.append("Environment: ⚠️  Not set to production")
            
            self.results["security"]["details"] = security_checks
            
            if any("❌" in check for check in security_checks):
                self.results["security"]["status"] = "failed"
            elif any("⚠️" in check for check in security_checks):
                self.results["security"]["status"] = "warning"
            else:
                self.results["security"]["status"] = "success"
            
            logger.info("✅ Security validation completed")
            
        except Exception as e:
            self.results["security"]["status"] = "failed"
            self.results["security"]["details"] = [f"❌ Security validation error: {e}"]
            logger.error(f"❌ Security validation failed: {e}")
    
    def generate_report(self):
        """Generate and display validation report"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 VALIDATION REPORT")
        logger.info("=" * 60)
        
        overall_status = "success"
        
        for component, result in self.results.items():
            status = result["status"]
            details = result["details"]
            
            # Determine status icon
            if status == "success":
                icon = "✅"
            elif status == "warning":
                icon = "⚠️ "
                if overall_status == "success":
                    overall_status = "warning"
            else:
                icon = "❌"
                overall_status = "failed"
            
            logger.info(f"\n{icon} {component.upper().replace('_', ' ')}: {status.upper()}")
            
            for detail in details:
                logger.info(f"   {detail}")
        
        # Overall summary
        logger.info("\n" + "=" * 60)
        if overall_status == "success":
            logger.info("🎉 ALL SYSTEMS OPERATIONAL")
            logger.info("Your medical chatbot is ready to run!")
        elif overall_status == "warning":
            logger.info("⚠️  SYSTEMS MOSTLY READY")
            logger.info("Some optional components need attention, but core functionality should work.")
        else:
            logger.info("❌ CRITICAL ISSUES FOUND")
            logger.info("Please resolve the failed components before running the chatbot.")
        
        logger.info("=" * 60)
        
        # Save report to file
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "results": self.results
        }
        
        with open(f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"📋 Detailed report saved to validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

async def main():
    """Main function to run validation"""
    try:
        validator = SystemValidator()
        await validator.run_validation()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Validation script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    # Run the validation
    asyncio.run(main())