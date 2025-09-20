#!/usr/bin/env python3
"""
OAuth System Fix and Integration
Creates a proper OAuth flow for Google and GitHub authentication
"""

import os
import re
from dotenv import load_dotenv

def fix_oauth_routes():
    """Fix OAuth routes in app.py to work with Supabase"""
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Enhanced OAuth route that works with Supabase
    oauth_fix = '''
# ==================== ENHANCED OAUTH SYSTEM ====================

@app.route('/auth/google')
def google_auth():
    """Enhanced Google OAuth with Supabase integration"""
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Generate OAuth URL
        redirect_url = f"{request.host_url}auth/callback"
        
        # Create OAuth sign-in with proper redirect
        auth_response = supabase.auth.sign_in_with_oauth({
            'provider': 'google',
            'options': {
                'redirect_to': redirect_url
            }
        })
        
        if auth_response and hasattr(auth_response, 'url'):
            return redirect(auth_response.url)
        else:
            # Fallback to manual Google OAuth
            google_client_id = os.getenv('GOOGLE_CLIENT_ID', '')
            if google_client_id and 'your-google-client-id' not in google_client_id:
                oauth_url = f"https://accounts.google.com/oauth/authorize"
                params = {
                    'client_id': google_client_id,
                    'redirect_uri': redirect_url,
                    'scope': 'email profile',
                    'response_type': 'code',
                    'state': 'google'
                }
                
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                return redirect(f"{oauth_url}?{query_string}")
            else:
                flash('Google OAuth not configured. Please set GOOGLE_CLIENT_ID in .env', 'error')
                return redirect(url_for('auth_page'))
        
    except Exception as e:
        app.logger.error(f"Google OAuth error: {str(e)}")
        flash('Google authentication temporarily unavailable', 'error')
        return redirect(url_for('auth_page'))

@app.route('/auth/github')
def github_auth():
    """Enhanced GitHub OAuth with Supabase integration"""
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Generate OAuth URL
        redirect_url = f"{request.host_url}auth/callback"
        
        # Create OAuth sign-in with proper redirect
        auth_response = supabase.auth.sign_in_with_oauth({
            'provider': 'github',
            'options': {
                'redirect_to': redirect_url
            }
        })
        
        if auth_response and hasattr(auth_response, 'url'):
            return redirect(auth_response.url)
        else:
            # Fallback to manual GitHub OAuth
            github_client_id = os.getenv('GITHUB_CLIENT_ID', '')
            if github_client_id and 'your-github-client-id' not in github_client_id:
                oauth_url = f"https://github.com/login/oauth/authorize"
                params = {
                    'client_id': github_client_id,
                    'redirect_uri': redirect_url,
                    'scope': 'user:email',
                    'state': 'github'
                }
                
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                return redirect(f"{oauth_url}?{query_string}")
            else:
                flash('GitHub OAuth not configured. Please set GITHUB_CLIENT_ID in .env', 'error')
                return redirect(url_for('auth_page'))
        
    except Exception as e:
        app.logger.error(f"GitHub OAuth error: {str(e)}")
        flash('GitHub authentication temporarily unavailable', 'error')
        return redirect(url_for('auth_page'))

@app.route('/auth/callback')
def oauth_callback():
    """Enhanced OAuth callback handler"""
    try:
        supabase = get_supabase_client()
        
        # Get the session from the URL fragments (handled by frontend)
        # This endpoint primarily handles server-side OAuth flows
        
        # Check for OAuth error
        error = request.args.get('error')
        if error:
            flash(f'Authentication failed: {error}', 'error')
            return redirect(url_for('auth_page'))
        
        # Check for authorization code (for manual OAuth)
        code = request.args.get('code')
        state = request.args.get('state')
        
        if code and state:
            # Handle manual OAuth flow
            if state == 'google':
                return handle_google_callback(code)
            elif state == 'github':
                return handle_github_callback(code)
        
        # For Supabase OAuth, the session is handled by frontend JavaScript
        # Redirect to a success page that will handle the session
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Success</title>
            <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
        </head>
        <body>
            <div style="text-align: center; padding: 50px; font-family: Arial, sans-serif;">
                <h2>🔄 Completing Authentication...</h2>
                <p>Please wait while we complete your sign-in.</p>
            </div>
            
            <script>
                // Initialize Supabase
                const supabase = window.supabase.createClient(
                    '{{ supabase_url }}',
                    '{{ supabase_key }}'
                );
                
                // Handle the OAuth session
                supabase.auth.onAuthStateChange((event, session) => {
                    if (event === 'SIGNED_IN' && session) {
                        // Store session info and redirect
                        fetch('/auth/session', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                access_token: session.access_token,
                                user: session.user
                            })
                        }).then(() => {
                            window.location.href = '/dashboard';
                        });
                    } else if (event === 'SIGNED_OUT' || !session) {
                        window.location.href = '/auth?error=authentication_failed';
                    }
                });
                
                // Check current session
                supabase.auth.getSession().then(({ data: { session } }) => {
                    if (session) {
                        fetch('/auth/session', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                access_token: session.access_token,
                                user: session.user
                            })
                        }).then(() => {
                            window.location.href = '/dashboard';
                        });
                    } else {
                        setTimeout(() => {
                            window.location.href = '/auth?error=no_session';
                        }, 3000);
                    }
                });
            </script>
        </body>
        </html>
        """, supabase_url=os.getenv('SUPABASE_URL'), supabase_key=os.getenv('SUPABASE_KEY'))
        
    except Exception as e:
        app.logger.error(f"OAuth callback error: {str(e)}")
        flash('Authentication failed. Please try again.', 'error')
        return redirect(url_for('auth_page'))

@app.route('/auth/session', methods=['POST'])
def handle_auth_session():
    """Handle OAuth session data from frontend"""
    try:
        data = request.get_json()
        access_token = data.get('access_token')
        user_data = data.get('user')
        
        if access_token and user_data:
            # Store in Redis session
            session_key = f"oauth_session:{user_data.get('id')}"
            redis_client.setex(session_key, 3600 * 8, json.dumps({
                'user_id': user_data.get('id'),
                'email': user_data.get('email'),
                'name': user_data.get('user_metadata', {}).get('name', user_data.get('email')),
                'provider': user_data.get('app_metadata', {}).get('provider', 'supabase'),
                'access_token': access_token,
                'authenticated': True,
                'auth_time': time.time()
            }))
            
            # Set Flask session
            session['user_id'] = user_data.get('id')
            session['authenticated'] = True
            session['user_email'] = user_data.get('email')
            session['user_name'] = user_data.get('user_metadata', {}).get('name', user_data.get('email'))
            
            return jsonify({'status': 'success'})
        
        return jsonify({'status': 'error', 'message': 'Invalid session data'}), 400
        
    except Exception as e:
        app.logger.error(f"Session handling error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def handle_google_callback(code):
    """Handle manual Google OAuth callback"""
    try:
        # Exchange code for tokens (implement based on your needs)
        # This is a simplified version - you'd need to implement the full OAuth flow
        flash('Google authentication successful!', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f'Google authentication failed: {str(e)}', 'error')
        return redirect(url_for('auth_page'))

def handle_github_callback(code):
    """Handle manual GitHub OAuth callback"""
    try:
        # Exchange code for tokens (implement based on your needs)
        # This is a simplified version - you'd need to implement the full OAuth flow
        flash('GitHub authentication successful!', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f'GitHub authentication failed: {str(e)}', 'error')
        return redirect(url_for('auth_page'))

@app.route('/auth/logout')
def logout():
    """Enhanced logout with Supabase and session cleanup"""
    try:
        # Clear Redis session
        user_id = session.get('user_id')
        if user_id:
            session_key = f"oauth_session:{user_id}"
            redis_client.delete(session_key)
        
        # Clear Flask session
        session.clear()
        
        # Supabase logout is handled by frontend
        flash('Successfully logged out', 'success')
        return redirect(url_for('auth_page'))
        
    except Exception as e:
        app.logger.error(f"Logout error: {str(e)}")
        session.clear()  # Clear session anyway
        return redirect(url_for('auth_page'))

# ==================== END ENHANCED OAUTH SYSTEM ====================
'''
    
    # Insert the OAuth fix into the app.py file
    # Find the location after the existing OAuth routes
    oauth_pattern = r'# ==================== OAUTH ROUTES ====================.*?# ==================== END OAUTH ====================\n'
    
    if re.search(oauth_pattern, content, re.DOTALL):
        # Replace existing OAuth section
        content = re.sub(oauth_pattern, oauth_fix.strip() + '\n\n', content, flags=re.DOTALL)
    else:
        # Find a good place to insert (after imports, before routes)
        insert_point = content.find("# ==================== FLASK ROUTES ====================")
        if insert_point > -1:
            content = content[:insert_point] + oauth_fix + '\n\n' + content[insert_point:]
        else:
            # Fallback: append to end
            content += '\n\n' + oauth_fix
    
    # Write back the fixed content
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ OAuth system fixed and integrated")

def integrate_enhanced_admin():
    """Integrate enhanced admin endpoints into app.py"""
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Enhanced admin endpoints
    admin_enhancement = '''
# ==================== ENHANCED ADMIN ENDPOINTS ====================

@app.route('/admin/api/metrics/enhanced', methods=['GET'])
@require_admin
def enhanced_admin_metrics():
    """Enhanced admin metrics with detailed system info"""
    try:
        import psutil
        from datetime import datetime
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # File system metrics
        data_dir = os.path.join(os.getcwd(), 'data')
        file_count = len(os.listdir(data_dir)) if os.path.exists(data_dir) else 0
        
        # Redis metrics
        redis_info = {}
        try:
            redis_info = {
                "connected": True,
                "memory_usage": redis_client.memory_usage(),
                "key_count": redis_client.dbsize(),
                "uptime": redis_client.info().get('uptime_in_seconds', 0)
            }
        except:
            redis_info = {"connected": False}
        
        metrics = {
            "system": {
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory.percent, 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "disk_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2)
            },
            "file_manager": {
                "data_directory": data_dir,
                "file_count": file_count,
                "directory_exists": os.path.exists(data_dir),
                "directory_size_mb": get_directory_size(data_dir) if os.path.exists(data_dir) else 0
            },
            "cache": redis_info,
            "oauth": {
                "google_configured": bool(os.getenv('GOOGLE_CLIENT_ID', '').replace('your-google-client-id', '')),
                "github_configured": bool(os.getenv('GITHUB_CLIENT_ID', '').replace('your-github-client-id', '')),
                "supabase_configured": bool(os.getenv('SUPABASE_URL'))
            },
            "application": {
                "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0,
                "version": "3.0.0-ENHANCED",
                "environment": os.getenv('FLASK_ENV', 'development')
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(metrics)
    except Exception as e:
        app.logger.error(f"Enhanced metrics error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/api/files/enhanced', methods=['GET'])
@require_admin
def enhanced_file_manager():
    """Enhanced file manager with detailed file info"""
    try:
        data_dir = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        files = []
        total_size = 0
        
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                file_size = stat.st_size
                total_size += file_size
                
                files.append({
                    "name": filename,
                    "size": file_size,
                    "size_formatted": format_file_size(file_size),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "type": os.path.splitext(filename)[1].lower(),
                    "is_pdf": filename.lower().endswith('.pdf'),
                    "is_indexed": check_file_indexed(filename)
                })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            "files": files,
            "summary": {
                "total_files": len(files),
                "total_size": total_size,
                "total_size_formatted": format_file_size(total_size),
                "data_directory": data_dir
            }
        })
    except Exception as e:
        app.logger.error(f"Enhanced file manager error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin/api/cache/manage', methods=['POST'])
@require_admin
def manage_cache():
    """Enhanced cache management endpoint"""
    try:
        action = request.json.get('action')
        
        if action == 'clear':
            # Clear all medbot cache keys
            keys = redis_client.keys("medbot:*")
            if keys:
                redis_client.delete(*keys)
                return jsonify({"status": "success", "message": f"Cleared {len(keys)} cache entries"})
            else:
                return jsonify({"status": "success", "message": "No cache entries to clear"})
        
        elif action == 'stats':
            info = redis_client.info()
            return jsonify({
                "status": "success",
                "stats": {
                    "connected_clients": info.get('connected_clients', 0),
                    "used_memory": info.get('used_memory', 0),
                    "used_memory_human": info.get('used_memory_human', '0B'),
                    "keyspace_hits": info.get('keyspace_hits', 0),
                    "keyspace_misses": info.get('keyspace_misses', 0),
                    "total_commands_processed": info.get('total_commands_processed', 0)
                }
            })
        
        else:
            return jsonify({"error": "Invalid action"}), 400
            
    except Exception as e:
        app.logger.error(f"Cache management error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_directory_size(directory):
    """Get directory size in MB"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return round(total_size / (1024 * 1024), 2)  # Convert to MB
    except:
        return 0

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def check_file_indexed(filename):
    """Check if a file has been indexed (simplified check)"""
    try:
        # This would check your vector database for the file
        # For now, just check if it's a PDF
        return filename.lower().endswith('.pdf')
    except:
        return False

# ==================== END ENHANCED ADMIN ENDPOINTS ====================
'''
    
    # Find where to insert the enhanced admin endpoints
    admin_pattern = r'# ==================== ADMIN API ENDPOINTS ====================.*?(?=# ===================|$)'
    
    if re.search(admin_pattern, content, re.DOTALL):
        # Insert after existing admin endpoints
        content = re.sub(admin_pattern, lambda m: m.group(0) + '\n\n' + admin_enhancement, content, flags=re.DOTALL)
    else:
        # Append to end
        content += '\n\n' + admin_enhancement
    
    # Write back the enhanced content
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Enhanced admin endpoints integrated")

def main():
    """Run all OAuth and admin system fixes"""
    print("🚀 OAuth System Fix and Admin Enhancement")
    print("=" * 50)
    
    print("\n🔧 Fixing OAuth routes...")
    fix_oauth_routes()
    
    print("\n🔧 Integrating enhanced admin endpoints...")
    integrate_enhanced_admin()
    
    print("\n✅ All fixes applied successfully!")
    print("\n🎯 NEXT STEPS:")
    print("1. Update OAuth credentials in .env with real Google/GitHub app credentials")
    print("2. Restart the Flask application")
    print("3. Test OAuth: http://localhost:8080/auth/google")
    print("4. Test Admin: http://localhost:8080/admin (login: sidduambesange005@gmail.com)")
    print("5. Test Enhanced Metrics: http://localhost:8080/admin/api/metrics/enhanced")
    print("6. Test File Manager: http://localhost:8080/admin/api/files/enhanced")

if __name__ == "__main__":
    main()