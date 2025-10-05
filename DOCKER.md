# 🐳 Docker Deployment Guide

This guide explains how to run the Art Trend Classifier application using Docker.

## 📋 Prerequisites

- Docker (version 20.10+)
- Docker Compose (version 2.0+)
- At least 4GB RAM available for Docker

## 🚀 Quick Start

### 1. Clone and Navigate
```bash
git clone https://github.com/Ne0n3k/Art-Trend-Classifier.git
cd Art-Trend-Classifier
```

### 2. Start All Services
```bash
# Using the management script (recommended)
./docker-scripts.sh start

# Or using docker-compose directly
docker-compose up --build -d
```

### 3. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛠️ Management Commands

### Using the Management Script
```bash
./docker-scripts.sh start     # Start all services
./docker-scripts.sh stop      # Stop all services
./docker-scripts.sh restart   # Restart all services
./docker-scripts.sh logs      # View logs
./docker-scripts.sh status    # Check service status
./docker-scripts.sh cleanup   # Remove everything
./docker-scripts.sh help      # Show help
```

### Using Docker Compose Directly
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build -d

# Check status
docker-compose ps
```

## 🏗️ Architecture

### Services

1. **Backend** (`art-classifier-backend`)
   - FastAPI application
   - Port: 8000
   - Health check: `/` endpoint
   - Model mounted from `./ml_model`

2. **Frontend** (`art-classifier-frontend`)
   - Nginx server
   - Port: 3000
   - Serves HTML files
   - Health check: `/health` endpoint

### Network
- Custom bridge network: `art-classifier-network`
- Services can communicate using service names

### Volumes
- `./ml_model` → `/app/ml_model` (read-only)
- `./logs` → `/app/logs` (for application logs)

## 🔧 Configuration

### Environment Variables
```yaml
# Backend
PYTHONUNBUFFERED=1
ENVIRONMENT=production
```

### Ports
- Frontend: `3000:80`
- Backend: `8000:8000`

## 📊 Monitoring

### Health Checks
- Backend: `curl http://localhost:8000/`
- Frontend: `curl http://localhost:3000/health`

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Resource Usage
```bash
# Container stats
docker stats

# Service status
docker-compose ps
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   lsof -i :3000
   
   # Kill the process
   kill -9 <PID>
   ```

2. **Model File Not Found**
   ```bash
   # Ensure model file exists
   ls -la ml_model/model/
   
   # Check volume mount
   docker-compose exec backend ls -la /app/ml_model/model/
   ```

3. **Build Failures**
   ```bash
   # Clean build
   docker-compose down
   docker system prune -f
   docker-compose up --build -d
   ```

4. **Memory Issues**
   ```bash
   # Check Docker memory limit
   docker info | grep -i memory
   
   # Increase Docker memory in Docker Desktop settings
   ```

### Debug Mode
```bash
# Run with debug output
docker-compose up --build

# Access container shell
docker-compose exec backend bash
docker-compose exec frontend sh
```

## 🔄 Updates

### Updating the Application
```bash
# Pull latest changes
git pull

# Rebuild and restart
./docker-scripts.sh restart
```

### Updating Dependencies
```bash
# Edit requirements.txt
# Then rebuild
docker-compose up --build -d
```

## 🧹 Cleanup

### Remove Everything
```bash
./docker-scripts.sh cleanup
```

### Manual Cleanup
```bash
# Stop and remove containers
docker-compose down

# Remove images
docker-compose down --rmi all

# Remove volumes
docker-compose down -v

# Clean up Docker system
docker system prune -f
```
