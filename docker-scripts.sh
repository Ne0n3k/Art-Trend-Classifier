#!/bin/bash

# Docker management scripts for Art Trend Classifier

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build and start all services
start_all() {
    print_status "Building and starting Art Trend Classifier..."
    
    # Create logs directory
    mkdir -p logs
    
    # Build and start services
    docker-compose up --build -d
    
    if [ $? -eq 0 ]; then
        print_success "All services started successfully!"
        print_status "Backend: http://localhost:8000"
        print_status "Frontend: http://localhost:3000"
        print_status "Use 'docker-compose logs -f' to view logs"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Stop all services
stop_all() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

# Restart all services
restart_all() {
    print_status "Restarting all services..."
    docker-compose restart
    print_success "All services restarted"
}

# View logs
view_logs() {
    print_status "Viewing logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Check service status
status() {
    print_status "Service status:"
    docker-compose ps
}

# Clean up (remove containers, images, volumes)
cleanup() {
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Show help
show_help() {
    echo "Art Trend Classifier - Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Build and start all services"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  logs      View logs from all services"
    echo "  status    Show service status"
    echo "  cleanup   Remove all containers, images, and volumes"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start the application"
    echo "  $0 logs     # View logs"
    echo "  $0 status   # Check status"
}

# Main script logic
case "$1" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        restart_all
        ;;
    logs)
        view_logs
        ;;
    status)
        status
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
