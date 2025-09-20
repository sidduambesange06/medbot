#!/usr/bin/env python3
"""
CRITICAL ISSUES ANALYSIS - Testing Engineer Perspective
Deep technical analysis to identify ALL system-breaking issues
"""

import os
import sys
import subprocess
import importlib.util
import json
import traceback
from pathlib import Path
from datetime import datetime

class CriticalIssuesAnalyzer:
    def __init__(self):
        self.critical_issues = []
        self.warnings = []
        self.blocking_issues = []
        self.dependency_issues = []
        
    def log_critical(self, category, issue, details, severity="CRITICAL"):
        """Log critical issue"""
        self.critical_issues.append({
            'category': category,
            'issue': issue,
            'details': details,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        
    def log_blocking(self, issue, details):
        """Log blocking issue"""
        self.blocking_issues.append({
            'issue': issue,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })

    def test_python_environment(self):
        """Test Python environment and basic imports"""
        print("🔍 TESTING PYTHON ENVIRONMENT...")
        
        try:
            # Test Python version
            if sys.version_info < (3, 8):
                self.log_critical("PYTHON", "Python version too old", 
                                f"Required: 3.8+, Found: {sys.version}", "BLOCKING")
            
            # Test critical system imports
            critical_imports = [
                'flask', 'redis', 'pinecone', 'openai', 'sentence_transformers',
                'langchain', 'faiss', 'numpy', 'pandas', 'torch', 'transformers'
            ]
            
            missing_packages = []
            for package in critical_imports:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.log_blocking("MISSING_PACKAGES", 
                                f"Critical packages missing: {missing_packages}")
                
            print(f"  📦 Missing packages: {len(missing_packages)}")
            
        except Exception as e:
            self.log_critical("PYTHON_ENV", "Python environment test failed", str(e))

    def test_file_structure(self):
        """Test critical file structure"""
        print("🔍 TESTING FILE STRUCTURE...")
        
        critical_files = [
            'app_production.py',
            'optimized_login_manager.py', 
            '.env',
            'templates/Oauth.html',
            'templates/admin_login.html'
        ]
        
        missing_files = []
        for file in critical_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.log_blocking("MISSING_FILES", f"Critical files missing: {missing_files}")
            
        # Check if data directory exists
        if not Path('data').exists():
            self.log_critical("DATA_DIR", "Data directory missing", "No medical documents directory")
            
        print(f"  📁 Missing critical files: {len(missing_files)}")

    def test_env_configuration(self):
        """Test environment configuration"""
        print("🔍 TESTING ENVIRONMENT CONFIGURATION...")
        
        if not Path('.env').exists():
            self.log_blocking("ENV_FILE", ".env file completely missing")
            return
            
        try:
            with open('.env', 'r') as f:
                env_content = f.read()
            
            # Critical environment variables
            critical_env_vars = [
                'SECRET_KEY', 'SUPABASE_URL', 'SUPABASE_KEY', 
                'GROQ_API_KEY', 'PINECONE_API_KEY', 'REDIS_URL'
            ]
            
            missing_env_vars = []
            for var in critical_env_vars:
                if var not in env_content or f'{var}=' not in env_content:
                    missing_env_vars.append(var)
            
            # Check for placeholder values
            placeholder_issues = []
            placeholders = [
                ('GOOGLE_CLIENT_ID', 'your-google-client-id'),
                ('GITHUB_CLIENT_ID', 'your-github-client-id'),
                ('SECRET_KEY', 'your-secret-key'),
            ]
            
            for var, placeholder in placeholders:
                if placeholder in env_content:
                    placeholder_issues.append(f"{var} has placeholder value")
            
            if missing_env_vars:
                self.log_blocking("ENV_VARS", f"Critical env vars missing: {missing_env_vars}")
                
            if placeholder_issues:
                self.log_critical("ENV_PLACEHOLDERS", "Placeholder values found", 
                                str(placeholder_issues))
                
            print(f"  🔑 Missing env vars: {len(missing_env_vars)}")
            print(f"  ⚠️ Placeholder issues: {len(placeholder_issues)}")
            
        except Exception as e:
            self.log_critical("ENV_READ", "Cannot read .env file", str(e))

    def test_database_connections(self):
        """Test database connections"""
        print("🔍 TESTING DATABASE CONNECTIONS...")
        
        # Test Redis connection
        try:
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
            print("  ✅ Redis: Connected")
        except Exception as e:
            self.log_critical("REDIS", "Redis connection failed", str(e))
            print(f"  ❌ Redis: {e}")
        
        # Test Pinecone connection
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            pinecone_key = os.getenv('PINECONE_API_KEY')
            if not pinecone_key or 'your-' in pinecone_key:
                self.log_critical("PINECONE_CONFIG", "Pinecone API key invalid", pinecone_key)
            else:
                # Try to initialize Pinecone
                from pinecone import Pinecone
                pc = Pinecone(api_key=pinecone_key)
                print("  ✅ Pinecone: Configured")
        except Exception as e:
            self.log_critical("PINECONE", "Pinecone connection failed", str(e))
            print(f"  ❌ Pinecone: {e}")

    def test_application_imports(self):
        """Test if main application can be imported"""
        print("🔍 TESTING APPLICATION IMPORTS...")
        
        try:
            # Try to import main application
            spec = importlib.util.spec_from_file_location("app_production", "app_production.py")
            if spec is None:
                self.log_blocking("APP_IMPORT", "Cannot load app_production.py spec")
                return
                
            # Try to load the module (this will execute imports)
            module = importlib.util.module_from_spec(spec)
            
            # This is where it will likely fail if there are import issues
            spec.loader.exec_module(module)
            print("  ✅ app_production.py imports successfully")
            
        except ImportError as e:
            self.log_blocking("APP_IMPORTS", f"Import error in app_production.py: {str(e)}")
            print(f"  ❌ Import error: {e}")
            
        except Exception as e:
            self.log_critical("APP_LOAD", "Failed to load application", str(e))
            print(f"  ❌ Load error: {e}")

    def test_gpu_cuda_availability(self):
        """Test GPU/CUDA availability"""
        print("🔍 TESTING GPU/CUDA...")
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f"  ✅ CUDA: Available - {torch.cuda.get_device_name(0)}")
            else:
                self.log_critical("CUDA", "CUDA not available", "GPU acceleration disabled")
                print("  ⚠️ CUDA: Not available - using CPU (slower)")
                
        except Exception as e:
            self.log_critical("TORCH", "PyTorch/CUDA test failed", str(e))

    def test_port_availability(self):
        """Test if required ports are available"""
        print("🔍 TESTING PORT AVAILABILITY...")
        
        import socket
        ports_to_test = [5000, 6379]  # Flask, Redis
        
        for port in ports_to_test:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        if port == 6379:  # Redis should be running
                            print(f"  ✅ Port {port}: Redis running")
                        else:  # Port 5000 should be free
                            self.log_critical("PORT_CONFLICT", f"Port {port} already in use", 
                                            "Another service using Flask port")
                    else:
                        if port == 6379:  # Redis should be running
                            self.log_critical("REDIS_PORT", "Redis not running on port 6379", 
                                            "Start Redis service")
                        else:
                            print(f"  ✅ Port {port}: Available")
            except Exception as e:
                self.log_critical("PORT_TEST", f"Port {port} test failed", str(e))

    def test_template_integrity(self):
        """Test template file integrity"""
        print("🔍 TESTING TEMPLATE INTEGRITY...")
        
        templates = [
            ('templates/Oauth.html', ['createClient', 'supabase', 'Google', 'GitHub']),
            ('templates/admin_login.html', ['admin', 'login', 'form']),
        ]
        
        for template_path, required_content in templates:
            try:
                if not Path(template_path).exists():
                    self.log_blocking("TEMPLATE_MISSING", f"{template_path} missing")
                    continue
                    
                content = Path(template_path).read_text(encoding='utf-8', errors='ignore')
                
                missing_content = []
                for item in required_content:
                    if item not in content:
                        missing_content.append(item)
                
                if missing_content:
                    self.log_critical("TEMPLATE_CONTENT", 
                                    f"{template_path} missing content", 
                                    f"Missing: {missing_content}")
                    
            except Exception as e:
                self.log_critical("TEMPLATE_READ", f"Cannot read {template_path}", str(e))

    def test_critical_routes_syntax(self):
        """Test if critical route definitions have syntax errors"""
        print("🔍 TESTING ROUTE SYNTAX...")
        
        try:
            # Read the app file and look for common syntax issues
            with open('app_production.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common issues
            syntax_issues = []
            
            # Check for unclosed parentheses in route definitions
            route_lines = [line for line in content.split('\n') if '@app.route' in line]
            for i, line in enumerate(route_lines):
                if line.count('(') != line.count(')'):
                    syntax_issues.append(f"Line {i}: Unmatched parentheses in route")
            
            # Check for missing colons
            if 'def ' in content:
                def_lines = [line for line in content.split('\n') if 'def ' in line and not line.strip().endswith(':')]
                if def_lines:
                    syntax_issues.append(f"Missing colons in function definitions")
            
            if syntax_issues:
                self.log_critical("SYNTAX", "Syntax issues in app_production.py", 
                                str(syntax_issues))
                
        except Exception as e:
            self.log_critical("SYNTAX_CHECK", "Cannot check syntax", str(e))

    def run_comprehensive_analysis(self):
        """Run all tests"""
        print("="*80)
        print("🔥 CRITICAL ISSUES ANALYSIS - TESTING ENGINEER PERSPECTIVE")
        print("="*80)
        
        # Run all tests
        self.test_python_environment()
        self.test_file_structure()
        self.test_env_configuration()
        self.test_database_connections()
        self.test_application_imports()
        self.test_gpu_cuda_availability()
        self.test_port_availability()
        self.test_template_integrity()
        self.test_critical_routes_syntax()
        
        # Generate report
        self.generate_critical_report()

    def generate_critical_report(self):
        """Generate comprehensive critical issues report"""
        print("\n" + "="*80)
        print("🚨 CRITICAL ISSUES REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"   🔥 Blocking Issues: {len(self.blocking_issues)}")
        print(f"   ⚠️ Critical Issues: {len(self.critical_issues)}")
        print(f"   📝 Total Issues: {len(self.blocking_issues) + len(self.critical_issues)}")
        
        # Show blocking issues first
        if self.blocking_issues:
            print(f"\n🚫 BLOCKING ISSUES (WILL PREVENT STARTUP):")
            print("-" * 60)
            for i, issue in enumerate(self.blocking_issues, 1):
                print(f"{i}. 🔥 {issue['issue']}")
                print(f"   Details: {issue['details']}")
                print()
        
        # Show critical issues
        if self.critical_issues:
            print(f"\n⚠️ CRITICAL ISSUES (MAY CAUSE FAILURES):")
            print("-" * 60)
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"{i}. {issue['severity']} - {issue['category']}: {issue['issue']}")
                print(f"   Details: {issue['details']}")
                print()
        
        # Determine overall status
        if self.blocking_issues:
            print("🚫 OVERALL STATUS: SYSTEM WILL NOT START")
            print("   Fix blocking issues before attempting to run the application.")
        elif len(self.critical_issues) > 5:
            print("⚠️ OVERALL STATUS: SYSTEM LIKELY TO FAIL")
            print("   Multiple critical issues may cause runtime failures.")
        elif self.critical_issues:
            print("⚠️ OVERALL STATUS: SYSTEM MAY HAVE ISSUES")
            print("   Some functionality may not work correctly.")
        else:
            print("✅ OVERALL STATUS: NO CRITICAL ISSUES DETECTED")
            print("   System should start and function normally.")
        
        # Save detailed report
        self.save_detailed_report()

    def save_detailed_report(self):
        """Save detailed report to file"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'blocking_issues': len(self.blocking_issues),
                'critical_issues': len(self.critical_issues),
                'total_issues': len(self.blocking_issues) + len(self.critical_issues)
            },
            'blocking_issues': self.blocking_issues,
            'critical_issues': self.critical_issues
        }
        
        with open('CRITICAL_ISSUES_REPORT.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: CRITICAL_ISSUES_REPORT.json")

if __name__ == "__main__":
    analyzer = CriticalIssuesAnalyzer()
    analyzer.run_comprehensive_analysis()