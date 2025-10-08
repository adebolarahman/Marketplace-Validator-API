"""
LLM Integration Module
Supports multiple LLM providers: Gemini, OpenAI, Anthropic
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: str, model: str, temperature: float = 0.3):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM"""
        pass
    
    def _load_standards(self) -> str:
        """Load marketplace standards"""

        return """# Marketplace Standards

## 1. Regulatory & Compliance Requirements

All product listings must include:

- **APR disclosure**: State both introductory and ongoing rates (representative APR as required by FCA)
  -  "0% intro APR for 12 months, then 18.9% APR representative (variable)"
  -  "Low rates available"

- **Fee transparency**: List all applicable fees upfront
  -  "Annual fee: £95, Late payment fee: up to £12"
  -  "Competitive fees apply"

- **Credit requirements**: Define eligibility criteria clearly
  -  "Requires good to excellent credit (credit score 881-999 or equivalent)"
  -  "Credit approval required"

## 2. Content Quality Standards

### Language & Clarity
- Use simple, jargon-free language accessible to general audiences
- Write concise sentences (aim for 15-20 words maximum)
- Maintain proper grammar, punctuation, and spelling
- Employ active voice and positive phrasing

## 5. Prohibited Language & Practices

### Misleading Claims
**Avoid**: "Guaranteed approval", "No credit check", "Instant approval", "Risk-free"
**Use instead**: "Subject to status and credit approval", "Quick application process"

### Exaggerated Superlatives
**Avoid**: "Best on the market", "Unlimited credit", "Unbeatable rates"
**Use instead**: "Competitive rates", "Generous credit limits available"

### Vague Promotional Terms
**Avoid**: "Incredible deal", "Revolutionary", "Life-changing", "Absolutely free"
**Use instead**: Specific benefits and clear terms

### Non-transparent Language
**Avoid**: "Hidden charges", "Limited-time offer" (without specifics), "Free money"
**Use instead**: Clear fee schedules, specific offer end dates, accurate benefit descriptions
"""
    
    def _build_validation_prompt(self, listing: str) -> str:
        """Build the validation prompt"""
        standards = self._load_standards()
        
        return f"""You are an AI validator for a credit card marketplace. Your task is to validate that the listing meets marketplace standards and FCA compliance requirements.

{standards}

**Your task:**
1. Analyze the listing carefully against ALL standards above
2. Identify ALL issues with specific locations in the text
3. Categorize each issue by severity: critical, warning, or info
4. Provide specific, actionable suggestions for each issue
5. Return your response as a valid JSON object

**Listing to validate:**
{listing}

**Response format (return ONLY valid JSON, no markdown):**
{{
  "issues": [
    {{
      "category": "Regulatory Compliance / Content Quality / Prohibited Language",
      "severity": "critical / warning / info",
      "issue": "Clear description of the problem",
      "location": "Where in the listing this appears",
      "current_text": "The problematic text",
      "suggested_replacement": "Specific replacement text",
      "standard_reference": "Which section of standards this violates"
    }}
  ]
}}

Return ONLY the JSON object, no additional text or markdown formatting."""


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.3):
        super().__init__(api_key, model, temperature)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Gemini API"""
        url = f"{self.base_url}/models/{self.model}:generateContent"
        
        headers = {
            "Content-Type": "application/json"
        }
        

        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
            }
        }
        
        params = {"key": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=payload, 
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Gemini API error: {error_text}")
                        raise Exception(f"Gemini API error: {response.status}")
                    
                    result = await response.json()
                    
                    if "candidates" in result and len(result["candidates"]) > 0:
                        content = result["candidates"][0]["content"]["parts"][0]["text"]
                        return content
                    else:
                        raise Exception("No response generated from Gemini")
        
        except asyncio.TimeoutError:
            logger.error("Gemini API timeout")
            raise Exception("LLM request timed out")
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3):
        super().__init__(api_key, model, temperature)
        self.base_url = "https://api.openai.com/v1"
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using OpenAI API"""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {error_text}")
                        raise Exception(f"OpenAI API error: {response.status}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        
        except asyncio.TimeoutError:
            logger.error("OpenAI API timeout")
            raise Exception("LLM request timed out")
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929", temperature: float = 0.3):
        super().__init__(api_key, model, temperature)
        self.base_url = "https://api.anthropic.com/v1"
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Anthropic API"""
        url = f"{self.base_url}/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error: {error_text}")
                        raise Exception(f"Anthropic API error: {response.status}")
                    
                    result = await response.json()
                    return result["content"][0]["text"]
        
        except asyncio.TimeoutError:
            logger.error("Anthropic API timeout")
            raise Exception("LLM request timed out")
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(
        provider_name: str,
        api_key: str,
        model: Optional[str] = None,
        temperature: float = 0.3
    ) -> BaseLLMProvider:
        """Create an LLM provider instance"""
        provider_name = provider_name.lower()
        
        if provider_name == "gemini":
            model = model or "gemini-1.5-flash"
            return GeminiProvider(api_key, model, temperature)
        elif provider_name == "openai":
            model = model or "gpt-4o-mini"
            return OpenAIProvider(api_key, model, temperature)
        elif provider_name == "anthropic":
            model = model or "claude-sonnet-4-5-20250929"
            return AnthropicProvider(api_key, model, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")


class EnhancedLLMValidator:
    """Enhanced validator with retry logic, fallback, and response parsing"""
    
    def __init__(
        self,
        provider: BaseLLMProvider,
        max_retries: int = 3,
        fallback_provider: Optional[BaseLLMProvider] = None
    ):
        self.provider = provider
        self.max_retries = max_retries
        self.fallback_provider = fallback_provider
    
    async def validate_listing(
        self,
        listing: str,
        strict_mode: bool = False
    ) -> Dict[str, Any]:
        """Validate a listing with retry logic"""
        prompt = self.provider._build_validation_prompt(listing)
        

        for attempt in range(self.max_retries):
            try:
                response = await self.provider.generate(prompt)
                parsed = self._parse_response(response)
                
                if parsed:
                    return self._enhance_validation_result(parsed, listing, strict_mode)
                
                logger.warning(f"Failed to parse response, attempt {attempt + 1}/{self.max_retries}")
                
            except Exception as e:
                logger.error(f"Validation attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:

                    if self.fallback_provider:
                        try:
                            response = await self.fallback_provider.generate(prompt)
                            parsed = self._parse_response(response)
                            if parsed:
                                return self._enhance_validation_result(parsed, listing, strict_mode)
                        except Exception as fallback_error:
                            logger.error(f"Fallback provider failed: {str(fallback_error)}")
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("Validation failed after all retries")
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response and extract JSON"""
        try:

            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            
            # Parse JSON
            parsed = json.loads(response)
            
            if "issues" not in parsed:
                logger.error("Response missing 'issues' field")
                return None
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")

            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    return json.loads(json_str)
            except:
                pass
            return None
    
    def _enhance_validation_result(
        self,
        parsed_result: Dict[str, Any],
        listing: str,
        strict_mode: bool
    ) -> Dict[str, Any]:
        """Enhance the validation result with additional analysis"""
        issues = parsed_result.get("issues", [])
        

        if strict_mode:

            for issue in issues:
                if issue.get("severity") == "warning":
                    issue["severity"] = "critical"
        
        critical_count = sum(1 for i in issues if i.get("severity") == "critical")
        warning_count = sum(1 for i in issues if i.get("severity") == "warning")
        info_count = sum(1 for i in issues if i.get("severity") == "info")
        
        compliance_score = 100.0
        compliance_score -= critical_count * 20
        compliance_score -= warning_count * 10
        compliance_score -= info_count * 5
        compliance_score = max(0.0, min(100.0, compliance_score))
        
        is_compliant = compliance_score >= 80.0 and critical_count == 0
        
        word_count = len(listing.split())
        has_structured_format = any(marker in listing for marker in ["**", "##", "•", "-", "✅"])
        
        return {
            "is_compliant": is_compliant,
            "compliance_score": compliance_score,
            "issues": issues,
            "summary": {
                "critical": critical_count,
                "warning": warning_count,
                "info": info_count,
                "total": len(issues)
            },
            "metadata": {
                "word_count": word_count,
                "has_structured_format": has_structured_format,
                "strict_mode": strict_mode
            }
        }