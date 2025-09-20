# 🚀 MedBot Ultra v3.0 - Production Deployment Guide

## 🏗️ Ultra-Optimized Production Architecture

### **Complete Production Stack:**
- **Flask Application**: Ultra-optimized with advanced caching
- **Redis**: High-performance caching and session storage  
- **Nginx**: Load balancer and reverse proxy
- **Prometheus + Grafana**: Comprehensive monitoring
- **Docker**: Containerized deployment
- **SSL/TLS**: Production security

---

## 📋 Prerequisites

### **Required Services:**
- ✅ **Pinecone**: Vector database for medical knowledge
- ✅ **Groq API**: LLM inference for medical responses
- ✅ **Supabase**: OAuth authentication (optional)
- ✅ **Redis**: Caching and session storage
- ✅ **Docker & Docker Compose**: Container orchestration

### **System Requirements:**
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended  
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection
- **OS**: Linux (Ubuntu 20.04+ recommended)

---

## ⚙️ Configuration Setup

### 1. **Environment Variables**
Create `.env` file in the project root:

```bash
# Core API Keys (REQUIRED)
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key

# Supabase Authentication (Optional)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Security Configuration
SECRET_KEY=ultra-secure-secret-key-change-in-production
ADMIN_USERNAME=admin
ADMIN_PASSWORD=ultra-secure-admin-password

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Performance Tuning
MAX_WORKERS=8
BATCH_SIZE=100
EMBEDDING_BATCH_SIZE=64
CACHE_DEFAULT_TIMEOUT=3600

# Monitoring (Optional)
SENTRY_DSN=your_sentry_dsn
GRAFANA_PASSWORD=secure_grafana_password

# Database (Optional - defaults to SQLite)
DATABASE_URL=postgresql://user:password@localhost/medbot

# SSL Configuration (Production)
SSL_CERT_PATH=/path/to/ssl/cert.pem
SSL_KEY_PATH=/path/to/ssl/private.key
```

### 2. **SSL Certificate Setup** (Production)
```bash
# Create SSL directory
mkdir -p ssl

# Option A: Let's Encrypt (Recommended)
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/

# Option B: Self-signed (Development)
openssl req -x509 -newkey rsa:4096 -keyout ssl/private.key -out ssl/cert.pem -days 365 -nodes
```

### 3. **Nginx Configuration**
Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream medbot_app {
        server medbot-app:8080 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/private.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        client_max_body_size 100M;
        
        location / {
            proxy_pass http://medbot_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

---

## 🚀 Deployment Options

### **Option 1: Quick Start (Recommended)**

```bash
# 1. Clone repository
git clone <your-repository>
cd medbot-v2/medbot

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 3. Deploy with Docker Compose
docker-compose -f docker-compose.production.yml up -d

# 4. Check status
docker-compose -f docker-compose.production.yml ps
```

### **Option 2: Step-by-Step Deployment**

```bash
# 1. Prepare data directory
mkdir -p data uploads logs
chmod 755 data uploads logs

# 2. Add medical textbooks to data directory
cp /path/to/medical/books/*.pdf data/

# 3. Build custom Docker image
docker build -f Dockerfile.production -t medbot-ultra:3.0 .

# 4. Start Redis
docker run -d --name medbot-redis \
  -p 6379:6379 \
  redis:7.2-alpine

# 5. Start application
docker run -d --name medbot-app \
  --link medbot-redis:redis \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  medbot-ultra:3.0

# 6. Start Nginx (if needed)
docker run -d --name medbot-nginx \
  --link medbot-app:medbot-app \
  -p 80:80 -p 443:443 \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v $(pwd)/ssl:/etc/nginx/ssl:ro \
  nginx:1.25-alpine
```

### **Option 3: Native Installation**

```bash
# 1. Install Python dependencies
pip install -r requirements_production.txt

# 2. Install and configure Redis
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# 3. Run application
export FLASK_ENV=production
gunicorn --bind 0.0.0.0:8080 \
         --workers 4 \
         --worker-class gevent \
         --timeout 120 \
         app_optimized:app
```

---

## 📊 Monitoring Setup

### **Prometheus Configuration**
Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'medbot-app'
    static_configs:
      - targets: ['medbot-app:8080']
    metrics_path: '/admin/api/metrics'
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### **Grafana Dashboard**
Access Grafana at `http://localhost:3000`:
- **Username**: admin
- **Password**: (from GRAFANA_PASSWORD env var)

**Key Metrics to Monitor:**
- ✅ Request rate and response times
- ✅ Cache hit rates
- ✅ Error rates and types
- ✅ System resources (CPU, memory)
- ✅ Active user sessions
- ✅ Vector database performance

---

## 🔧 Production Operations

### **Health Checks**
```bash
# Application health
curl -f http://localhost:8080/health

# Redis health
docker exec medbot-redis redis-cli ping

# Full system status
curl -s http://localhost:8080/health | jq
```

### **Log Management**
```bash
# View application logs
docker logs medbot-app

# View real-time logs
docker logs -f medbot-app

# Access log files
tail -f logs/medbot_production.log
tail -f logs/medbot_errors.log
```

### **Backup Strategy**
```bash
# Create automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup data and uploads
tar -czf "$BACKUP_DIR/medbot_data_$DATE.tar.gz" data/ uploads/

# Backup Redis
docker exec medbot-redis redis-cli BGSAVE
docker cp medbot-redis:/data/dump.rdb "$BACKUP_DIR/redis_$DATE.rdb"

# Clean old backups (keep 7 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.rdb" -mtime +7 -delete

echo "Backup completed: $DATE"
```

### **Scaling Operations**
```bash
# Scale application instances
docker-compose -f docker-compose.production.yml up -d --scale medbot-app=3

# Update configuration without downtime
docker-compose -f docker-compose.production.yml exec medbot-app reload

# Rolling updates
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d
```

---

## 🔒 Security Checklist

### **Application Security:**
- ✅ Change default admin credentials
- ✅ Use strong SECRET_KEY
- ✅ Enable HTTPS with valid SSL certificates
- ✅ Configure firewall rules
- ✅ Regular security updates
- ✅ Monitor access logs

### **Container Security:**
- ✅ Run containers as non-root user
- ✅ Use specific image tags (not 'latest')
- ✅ Scan images for vulnerabilities
- ✅ Limit container resources
- ✅ Network segmentation

### **Data Security:**
- ✅ Encrypt sensitive data at rest
- ✅ Secure API keys and credentials
- ✅ Regular data backups
- ✅ Access control and auditing

---

## 🚦 Performance Optimization

### **Application Tuning:**
```python
# Recommended environment variables for high performance
MAX_WORKERS=8                    # Number of worker processes
WORKER_CONNECTIONS=1000          # Connections per worker
CACHE_DEFAULT_TIMEOUT=3600       # Cache TTL in seconds
BATCH_SIZE=100                   # Vector processing batch size
EMBEDDING_BATCH_SIZE=64          # Embedding generation batch size
```

### **Redis Optimization:**
```bash
# Add to redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 60
timeout 300
```

### **System Optimization:**
```bash
# Increase file descriptors
echo "* soft nofile 65535" >> /etc/security/limits.conf
echo "* hard nofile 65535" >> /etc/security/limits.conf

# Optimize network settings
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 2048" >> /etc/sysctl.conf
sysctl -p
```

---

## 🆘 Troubleshooting

### **Common Issues:**

**1. Application won't start:**
```bash
# Check logs
docker logs medbot-app

# Verify environment variables
docker exec medbot-app env | grep -E "(PINECONE|GROQ|REDIS)"

# Test connectivity
docker exec medbot-app curl -f http://localhost:8080/health
```

**2. Redis connection issues:**
```bash
# Test Redis connectivity
docker exec medbot-redis redis-cli ping

# Check Redis logs
docker logs medbot-redis

# Reset Redis connection
docker restart medbot-redis
```

**3. Performance issues:**
```bash
# Monitor system resources
docker stats

# Check application metrics
curl -s http://localhost:8080/admin/api/metrics | jq

# Analyze response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/health
```

**4. Memory issues:**
```bash
# Check memory usage
docker exec medbot-app ps aux

# Monitor Redis memory
docker exec medbot-redis redis-cli info memory

# Clear caches if needed
curl -X POST http://localhost:8080/admin/api/clear-cache
```

---

## 📈 Production Monitoring

### **Key Performance Indicators:**
- **Response Time**: < 500ms average
- **Cache Hit Rate**: > 80%
- **Error Rate**: < 1%
- **Uptime**: > 99.9%
- **Memory Usage**: < 80%
- **CPU Usage**: < 70%

### **Alerting Setup:**
Configure alerts for:
- High error rates
- Slow response times
- Memory/CPU threshold breaches
- Service unavailability
- Certificate expiration

---

## 🔄 Maintenance

### **Regular Tasks:**
- **Daily**: Monitor system health and performance
- **Weekly**: Review error logs and optimize performance
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Full system backup and disaster recovery testing

### **Update Procedure:**
1. Create full system backup
2. Test updates in staging environment
3. Schedule maintenance window
4. Deploy updates with rolling restart
5. Verify system functionality
6. Monitor for issues

---

## 🎯 Access URLs (Production)

- **🌐 Main Application**: `https://yourdomain.com`
- **🔐 Admin Panel**: `https://yourdomain.com/admin`
- **📊 Health Check**: `https://yourdomain.com/health`
- **📈 Grafana**: `https://yourdomain.com:3000`
- **⚡ Prometheus**: `https://yourdomain.com:9090`

---

## 📞 Support

For deployment support:
- **Documentation**: Review this guide thoroughly
- **Logs**: Always check application and system logs first
- **Monitoring**: Use Grafana dashboards for insights
- **Health Checks**: Monitor `/health` endpoint regularly

**Ultra-Optimized MedBot v3.0** is designed for production excellence with enterprise-grade features, monitoring, and scalability. Follow this guide for a successful deployment! 🚀