<<<<<<< HEAD
--- This document provides quick start guide for the Marketplace Validator API, and guidance on choosing between different validator implementations based on specific requirements, scale, and constraints. 


1. Marketplace Validator API - Quick Start Guide 

Prerequisites

Python 
Groq API Key (free at https://console.groq.com)
Terminal/Command Prompt

Setup

 1. Install Dependencies


pip install fastapi uvicorn aiohttp python-dotenv pydantic


2. Configure Environment

Create `.env` file in project root:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
CACHE_TTL=3600

3. Start API Server


uvicorn main:app --reload

Expected Output:

INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
Groq validator initialized with model: llama-3.3-70b-versatile



Testing Methods

Method 1: Swagger UI (Recommended)

1. Open browser:** http://localhost:8000/docs
2. Click: `POST /api/v1/validate`
3. Click: "Try it out"
4. Replace JSON with:

```json
{
  "listing": "Premium Card: 0% intro APR for 12 months, then 18.9% APR representative (variable). Annual fee: £95. Late payment fee: up to £12. Requires good credit (score 881-999).",
  "listing_id": "test-001"
}
```

5. Click:** "Execute"
6. View results in Response body


Method 2: Python Test Script

Use the `test_validator.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"


payload = {
    "listing": "Premium Card: 0% intro APR for 12 months, then 18.9% APR (variable). Annual fee: £95."
}

response = requests.post(f"{BASE_URL}/api/v1/validate", json=payload)
print(json.dumps(response.json(), indent=2))
```

Run:
```bash
python test_validator.py
```

---



2. Choosing between different validator implementations based on specific requirements, scale, and constraints. 


Implementation Options

Option 1: Current Implementation (Simple LLM-Only)
What it is: Single Groq API call per validation, with caching and retry logic.

Option 2: Hybrid Validator (Rule-Based + LLM)
What it is: Fast rule-based pre-filtering, then LLM for nuanced checks.

Option 3: Fine-Tuned Model
What it is: Custom-trained model specifically for credit card validation.

Option 4: Ensemble Validator
What it is: Multiple validators vote on results for maximum accuracy.


Decision Framework

When to Use Current Implementation (LLM-Only)

Choose, if:
- You're just starting out
- Validating < 100k listings/month
- Need quick deployment (same day)
- Budget is limited (< $1000/month)
- Requirements change frequently
- Team has no ML expertise
- Accuracy of 85% is acceptable

Don't, if:
- Latency must be < 200ms
- Validating > 1M listings/month
- Cost per validation must be < $0.0005
- Need > 90% accuracy



When to Use Hybrid Validator (Rules + LLM)

Choose, if:
- Validating 100k - 5M listings/month
- Need 60% latency reduction
- Want 70% cost savings
- Have clear regulatory requirements
- Can invest 1-2 weeks for setup
- Need deterministic checks for compliance

Don't, if:
- Requirements are unclear
- No engineering resources for rule maintenance
- Validation rules change daily



When to Use Fine-Tuned Model

Choose, if:
- Validating > 5M listings/month
- Have 10k+ labeled examples
- Need 92%+ accuracy
- Requirements are stable
- Have ML engineering expertise
- 3x faster inference needed

Don't, if:
- < 1M validations/month (not worth it)
- No labeled training data
- Requirements change monthly
- No ML expertise on team
- Budget < $2000/month



When to Use Ensemble Validator

Choose, if:
- Accuracy > 95% is critical
- Validating high-value listings only
- Cost is not primary concern
- False positives are expensive
- Legal/compliance implications
- Can tolerate 3x higher latency

Don't, if:
- Need fast response times
- High volume (> 1M/month)
- Budget constrained
- 85-90% accuracy sufficient

---


3. High level design for how to serve this in a production cloud environment.


                   
AWS (Recommended)
Services:
  Compute: EKS (Elastic Kubernetes Service)
  Load Balancer: Application Load Balancer (ALB)
  Cache: ElastiCache for Redis
  Secrets: AWS Secrets Manager
  Monitoring: CloudWatch + X-Ray
  Logging: CloudWatch Logs
  CDN: CloudFront
  WAF: AWS WAF
  DNS: Route 53


