#!/bin/bash

# Setup Verification Script for Motion Analysis Agent
# This script checks if all prerequisites and configuration are correct

echo "=========================================="
echo "Motion Analysis Agent - Setup Verification"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track if any checks fail
ERRORS=0

# Function to print success
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
error() {
    echo -e "${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

# Function to print warning
warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "Checking prerequisites..."
echo ""

# Check Docker
echo -n "Docker: "
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d ' ' -f 3 | cut -d ',' -f 1)
    success "Installed (version $DOCKER_VERSION)"
else
    error "Not installed. Install from https://docs.docker.com/get-docker/"
fi

# Check Docker Compose
echo -n "Docker Compose: "
if command -v docker compose &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short)
    success "Installed (version $COMPOSE_VERSION)"
else
    error "Not installed. Update Docker Desktop to latest version"
fi

# Check if Docker daemon is running
echo -n "Docker Daemon: "
if docker info &> /dev/null; then
    success "Running"
else
    error "Not running. Start Docker Desktop"
fi

echo ""
echo "Checking configuration files..."
echo ""

# Check if .env file exists
echo -n ".env file: "
if [ -f ".env" ]; then
    success "Found"
    
    # Check if API key is set
    echo -n "Google API Key: "
    if grep -q "GOOGLE_API_KEY=AIza" .env; then
        success "Configured"
    elif grep -q "GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE" .env; then
        error "Not set. Replace YOUR_GOOGLE_API_KEY_HERE with your actual key"
    elif grep -q "GOOGLE_API_KEY=" .env; then
        warning "Possibly not set. Verify your API key is correct"
    else
        error "GOOGLE_API_KEY not found in .env file"
    fi
else
    error "Not found. Run: cp .env.example .env"
fi

# Check docker-compose.yml
echo -n "docker-compose.yml: "
if [ -f "docker-compose.yml" ]; then
    success "Found"
else
    error "Not found. This file is required"
fi

echo ""
echo "Checking directory structure..."
echo ""

# Check required directories exist
for dir in backend frontend; do
    echo -n "$dir/: "
    if [ -d "$dir" ]; then
        success "Found"
    else
        error "Not found. Directory structure may be incomplete"
    fi
done

# Check critical backend files
echo -n "backend/requirements.txt: "
if [ -f "backend/requirements.txt" ]; then
    success "Found"
    
    # Check for new dependencies
    if grep -q "mediapipe" backend/requirements.txt; then
        success "Dependencies updated (mediapipe found)"
    else
        error "backend/requirements.txt missing mediapipe"
    fi
else
    error "Not found"
fi

echo -n "backend/main.py: "
if [ -f "backend/main.py" ]; then
    success "Found"
else
    error "Not found"
fi

# Check frontend files
echo -n "frontend/package.json: "
if [ -f "frontend/package.json" ]; then
    success "Found"
else
    error "Not found"
fi

# Create necessary directories if they don't exist
echo ""
echo "Creating required directories..."
echo ""

mkdir -p backend/uploads backend/processed

echo -n "backend/uploads/: "
if [ -d "backend/uploads" ]; then
    success "Created/exists"
else
    error "Failed to create"
fi

echo -n "backend/processed/: "
if [ -d "backend/processed" ]; then
    success "Created/exists"
else
    error "Failed to create"
fi

echo ""
echo "Checking Docker resources..."
echo ""

# Check Docker memory allocation
DOCKER_MEMORY=$(docker info --format '{{.MemTotal}}' 2>/dev/null)
if [ ! -z "$DOCKER_MEMORY" ]; then
    MEMORY_GB=$((DOCKER_MEMORY / 1024 / 1024 / 1024))
    echo -n "Docker Memory: "
    if [ $MEMORY_GB -ge 4 ]; then
        success "${MEMORY_GB}GB allocated (sufficient)"
    else
        warning "${MEMORY_GB}GB allocated (4GB recommended)"
        echo "   Increase memory in Docker Desktop: Preferences → Resources → Memory"
    fi
fi

# Check disk space
echo -n "Disk Space: "
if command -v df &> /dev/null; then
    AVAILABLE=$(df -h . | tail -1 | awk '{print $4}')
    success "$AVAILABLE available"
else
    warning "Could not check disk space"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "Your system is ready. Next steps:"
    echo "  1. docker compose up -d --build"
    echo "  2. Wait ~30 seconds for services to start"
    echo "  3. Open http://localhost:3000 in your browser"
    echo ""
else
    echo -e "${RED}Found $ERRORS error(s)${NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    echo "See QUICKSTART.md for detailed setup instructions."
    echo ""
fi

exit $ERRORS