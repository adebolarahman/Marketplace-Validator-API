"""
Marketplace Validator API - Production Ready with Groq LLM
Advanced features: Caching, Rate Limiting, Retry Logic, Monitoring
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import json
import aiohttp
import asyncio
import hashlib
import time
import logging
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30
    RATE_LIMIT: int = 100

@lru_cache()
def get_settings():
    return Settings()

class ValidationRequest(BaseModel):
    listing: str = Field(..., min_length=10, max_length=10000)
    listing_id: Optional[str] = None
    lender_id: Optional[str] = None
    strict_mode: bool = False
    
    @validator('listing')
    def validate_listing(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Listing must be at least 10 characters')
        return v.strip()

class ValidationIssue(BaseModel):
    category: str
    severity: str  
    issue: str
    current_text: Optional[str] = None
    suggested_replacement: Optional[str] = None
    standard_reference: str

class ValidationResponse(BaseModel):
    listing_id: Optional[str]
    is_compliant: bool
    compliance_score: float = Field(..., ge=0.0, le=100.0)
    issues: List[ValidationIssue]
    summary: Dict[str, int]
    processed_at: datetime
    processing_time_ms: float

class BatchValidationRequest(BaseModel):
    listings: List[ValidationRequest] = Field(..., max_items=50)

class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    model: str
    cache_size: int
    timestamp: datetime

class SimpleCache:
    """In-memory cache with TTL"""
    def __init__(self):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    logger.info(f"Cache hit: {key[:16]}...")
                    return value
                else:
                    del self._cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: int):
        async with self._lock:
            expiry = time.time() + ttl
            self._cache[key] = (value, expiry)
    
    def size(self) -> int:
        return len(self._cache)
    
    async def clear(self):
        async with self._lock:
            self._cache.clear()

cache = SimpleCache()


class RateLimiter:
    """Simple rate limiter"""
    def __init__(self, max_requests: int = 100, window: int = 60):
        self._max_requests = max_requests
        self._window = window
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def check_limit(self, client_id: str) -> bool:
        async with self._lock:
            now = time.time()
            if client_id not in self._requests:
                self._requests[client_id] = []
            
            self._requests[client_id] = [
                req_time for req_time in self._requests[client_id]
                if now - req_time < self._window
            ]
            
            if len(self._requests[client_id]) >= self._max_requests:
                return False
            
            self._requests[client_id].append(now)
            return True

rate_limiter = RateLimiter()

class GroqValidator:
    """Enhanced Groq validator with retry logic and caching"""
    
    def __init__(self, api_key: str, model: str, cache_ttl: int = 3600):
        self.api_key = api_key
        self.model = model
        self.cache_ttl = cache_ttl
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.semaphore = asyncio.Semaphore(10)  
    
    def _create_cache_key(self, listing: str, strict_mode: bool) -> str:
        """Create cache key from listing"""
        content = f"{listing}:{strict_mode}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _build_prompt(self, listing: str) -> str:
        """Build validation prompt"""
        return f"""You are an expert FCA compliance validator for credit card listings.

CRITICAL REQUIREMENTS:
1. APR Disclosure: Must state both intro and ongoing APR (e.g., "0% intro APR for 12 months, then 18.9% APR representative (variable)")
2. Fee Transparency: Must list ALL fees upfront (annual fee, late payment fee)
3. Credit Requirements: Must specify eligibility (e.g., "Requires good credit (score 881-999)")

PROHIBITED LANGUAGE (critical violations):
- "Guaranteed approval", "No credit check", "Instant approval", "Risk-free"
- "Best on the market", "Unlimited credit", "Unbeatable rates"
- "Absolutely free", "Incredible deal", "Revolutionary"
- "Limited-time offer" (without specifics)

LISTING TO VALIDATE:
{listing}

Analyze thoroughly and respond with ONLY valid JSON (no markdown, no extra text):
{{
  "issues": [
    {{
      "category": "Regulatory Compliance",
      "severity": "critical",
      "issue": "Clear description of the problem",
      "current_text": "The exact problematic text",
      "suggested_replacement": "Specific compliant alternative",
      "standard_reference": "Section 1: Regulatory Requirements"
    }}
  ]
}}

If fully compliant, return: {{"issues": []}}
Be thorough - check for ALL potential issues."""
    
    async def validate(self, listing: str, strict_mode: bool = False) -> Dict[str, Any]:
        """Validate listing with caching and retry"""
        
        cache_key = self._create_cache_key(listing, strict_mode)
        cached = await cache.get(cache_key)
        if cached:
            return cached
        
        async with self.semaphore:
            for attempt in range(get_settings().MAX_RETRIES):
                try:
                    result = await self._call_groq_api(listing)
                    processed = self._process_result(result, listing, strict_mode)
                    
                    await cache.set(cache_key, processed, self.cache_ttl)
                    return processed
                    
                except Exception as e:
                    logger.error(f"Validation attempt {attempt + 1} failed: {str(e)}")
                    if attempt == get_settings().MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt) 
    
    async def _call_groq_api(self, listing: str) -> str:
        """Call Groq API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self._build_prompt(listing)}
            ],
            "temperature": 0.2,
            "max_tokens": 3000,
            "response_format": {"type": "json_object"}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=get_settings().TIMEOUT)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Groq API error {response.status}: {error_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    def _process_result(self, content: str, listing: str, strict_mode: bool) -> Dict[str, Any]:
        """Process and structure validation result"""
        try:

            parsed = json.loads(content)
            issues = parsed.get("issues", [])
            

            if strict_mode:
                for issue in issues:
                    if issue.get("severity") == "warning":
                        issue["severity"] = "critical"
            

            critical = sum(1 for i in issues if i.get("severity") == "critical")
            warning = sum(1 for i in issues if i.get("severity") == "warning")
            info = sum(1 for i in issues if i.get("severity") == "info")
            
            score = 100.0
            score -= critical * 20
            score -= warning * 10
            score -= info * 5
            score = max(0.0, min(100.0, score))
            
            is_compliant = score >= 80.0 and critical == 0
            
            return {
                "is_compliant": is_compliant,
                "compliance_score": score,
                "issues": issues,
                "summary": {
                    "critical": critical,
                    "warning": warning,
                    "info": info,
                    "total": len(issues)
                }
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}, Content: {content}")
            raise Exception(f"Failed to parse LLM response: {str(e)}")


settings = get_settings()
validator = None
if settings.GROQ_API_KEY:
    validator = GroqValidator(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        cache_ttl=settings.CACHE_TTL
    )
    logger.info(f"✓ Groq validator initialized with model: {settings.GROQ_MODEL}")
else:
    logger.warning(" GROQ_API_KEY not configured - API will not function")


app = FastAPI(
    title="Marketplace Validator API",
    description="Production-ready credit card listing validator using Groq LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting"""
    client_id = request.client.host if request.client else "unknown"
    
    if not await rate_limiter.check_limit(client_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."}
        )
    
    return await call_next(request)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging"""
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    
    logger.info(
        f"Method={request.method} Path={request.url.path} "
        f"Status={response.status_code} Duration={duration:.2f}ms"
    )
    
    return response

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Marketplace Validator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "llm": {
            "provider": "Groq",
            "model": settings.GROQ_MODEL,
            "configured": validator is not None
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if validator else "degraded",
        llm_configured=validator is not None,
        model=settings.GROQ_MODEL if validator else "not configured",
        cache_size=cache.size(),
        timestamp=datetime.utcnow()
    )

@app.post("/api/v1/validate", response_model=ValidationResponse, tags=["Validation"])
async def validate_listing(request: ValidationRequest):
    """
    Validate a single credit card listing.
    
    - **listing**: Credit card listing text (10-10000 chars)
    - **listing_id**: Optional unique identifier
    - **strict_mode**: Enable strict validation (warnings become critical)
    
    Returns detailed validation with issues and compliance score.
    """
    if not validator:
        raise HTTPException(
            status_code=503,
            detail="Validator not configured. Set GROQ_API_KEY in environment."
        )
    
    start_time = time.time()
    
    try:
        result = await validator.validate(request.listing, request.strict_mode)
        duration = (time.time() - start_time) * 1000
        
        response = ValidationResponse(
            listing_id=request.listing_id,
            is_compliant=result["is_compliant"],
            compliance_score=result["compliance_score"],
            issues=[ValidationIssue(**issue) for issue in result["issues"]],
            summary=result["summary"],
            processed_at=datetime.utcnow(),
            processing_time_ms=duration
        )
        
        logger.info(
            f"Validated listing_id={request.listing_id}, "
            f"compliant={response.is_compliant}, "
            f"score={response.compliance_score:.1f}, "
            f"duration={duration:.0f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/api/v1/validate/batch", tags=["Validation"])
async def validate_batch(request: BatchValidationRequest):
    """
    Validate multiple listings in batch (max 50).
    
    Returns validation results for all listings.
    """
    if not validator:
        raise HTTPException(
            status_code=503,
            detail="Validator not configured. Set GROQ_API_KEY in environment."
        )
    
    start_time = time.time()
    
    try:
        #Process concurrently
        tasks = [
            validate_listing(listing_req)
            for listing_req in request.listings
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = [
            r for r in results if not isinstance(r, Exception)
        ]
        
        duration = (time.time() - start_time) * 1000
        
        logger.info(
            f"Batch validated: {len(successful_results)}/{len(request.listings)} "
            f"successful, duration={duration:.0f}ms"
        )
        
        return {
            "results": successful_results,
            "total_processed": len(successful_results),
            "total_requested": len(request.listings),
            "processing_time_ms": duration
        }
        
    except Exception as e:
        logger.error(f"Batch validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")

@app.post("/api/v1/cache/clear", tags=["Admin"])
async def clear_cache():
    """Clear the validation cache (admin endpoint)"""
    await cache.clear()
    logger.info("Cache cleared")
    return {"message": "Cache cleared successfully", "timestamp": datetime.utcnow()}

@app.get("/api/v1/stats", tags=["Admin"])
async def get_stats():
    """Get API statistics"""
    return {
        "cache_size": cache.size(),
        "validator_configured": validator is not None,
        "model": settings.GROQ_MODEL if validator else None,
        "timestamp": datetime.utcnow()
    }

@app.on_event("startup")
async def startup_event():
    logger.info(" Marketplace Validator API starting...")
    logger.info(f"Model: {settings.GROQ_MODEL}")
    logger.info(f"Cache TTL: {settings.CACHE_TTL}s")
    logger.info(f"Validator configured: {validator is not None}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Marketplace Validator API...")
    await cache.clear()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")