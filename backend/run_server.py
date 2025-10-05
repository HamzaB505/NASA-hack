#!/usr/bin/env python3
"""
Simple script to run the FastAPI server for ExoPlanet AI
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    print("🚀 Starting ExoPlanet AI Backend Server...")
    print("📡 API will be available at: http://localhost:8000")
    print("🌐 Frontend will be available at: http://localhost:8000")
    print("📚 API Documentation at: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
