#!/usr/bin/env python3
"""
Project Structure Setup Script for Marketplace Validator API
Creates a production-ready FastAPI project structure
"""

from pathlib import Path

def create_file(path, content=""):
    """Create a file with content"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)
    print(f" {path}")

def create_project_structure():
    """Create complete project structure"""
    
    packages = [
        "src/validators", "src/models", "src/utils",
        "tests/unit", "tests/integration"
    ]
    for pkg in packages:
        create_file(f"{pkg}/__init__.py")
    
    for d in ["k8s", "monitoring/grafana-dashboards", "docs", "scripts", "sample_listings"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    files = {
        "main.py": "# FastAPI main application\n# Run: uvicorn main:app --reload\n",
        "src/validators/llm_validator.py": "# LLM validation logic\n",
        "src/validators/rule_validator.py": "# Rule-based validation logic\n",
        "src/models/schemas.py": "# Pydantic models\n",
        "src/utils/cache.py": "# Caching utilities\n",
        "src/utils/rate_limiter.py": "# Rate limiting utilities\n",
        "tests/conftest.py": "# Pytest fixtures\n",
        "tests/test_api.py": "# API endpoint tests\n",
        
        ".env.example": """LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
REDIS_URL=redis://localhost:6379/0
ENVIRONMENT=development
LOG_LEVEL=INFO
""",
        
        ".gitignore": """__pycache__/
*.py[cod]
venv/
.env
.pytest_cache/
.coverage
htmlcov/
*.log
.DS_Store
""",
        
        "pytest.ini": """[pytest]
testpaths = tests
addopts = -v --cov=. --cov-report=html
""",
        
        "requirements.txt": """fastapi
uvicorn[standard]
pydantic
aiohttp
redis
prometheus-client
pytest
pytest-asyncio
pytest-cov
black
python-dotenv
""",
       
        
        "sample_listings/compliant.txt": """Premium Card: 0% intro APR for 12 months, then 18.9% APR (variable).
Annual fee: £95. Requires good credit (score 881-999).
""",
        
        "sample_listings/non_compliant.txt": """BEST CARD EVER! Guaranteed approval! No credit check!
""",
    }
    
    for path, content in files.items():
        create_file(path, content)

if __name__ == "__main__":
    print("Setting up Marketplace Validator...\n")
    create_project_structure()
    print("\n Done! Next steps:")
    print("1. python -m venv venv")
    print("2. source venv/bin/activate")
    print("3. pip install -r requirements.txt")
    print("4. cp .env.example .env")
    print("5. uvicorn main:app --reload")