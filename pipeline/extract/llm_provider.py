"""
Multi-provider LLM client factory with rate limiting.

Supports:
- OpenAI (gpt-4o, gpt-4o-mini, etc.)
- Anthropic (claude-3-5-sonnet, claude-3-haiku, etc.)
- Google (gemini-1.5-flash, gemini-1.5-pro, etc.)

Usage:
    client = create_instructor_client(provider="anthropic", model="claude-3-5-sonnet-20241022")
    result = client.chat.completions.create(
        model=model,
        response_model=MySchema,
        messages=[...],
    )
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Type, TypeVar

import instructor
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# Provider-specific model mappings
PROVIDER_MODELS = {
    LLMProvider.OPENAI: [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    LLMProvider.ANTHROPIC: [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
    ],
    LLMProvider.GOOGLE: [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-flash-latest",
    ],
}

# Model aliases for convenience
MODEL_ALIASES = {
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-haiku-3": "claude-3-haiku-20240307",
    "claude-opus": "claude-3-opus-20240229",
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-2.5-pro",
}


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: Optional[int] = None  # Max requests per minute (None = no limit)
    delay_between_calls: float = 0.0  # Fixed delay between API calls (seconds)
    tokens_per_minute: Optional[int] = None  # For future TPM-based limiting
    max_retries: int = 5  # Max retries on rate limit errors
    initial_retry_delay: float = 2.0  # Initial retry delay for exponential backoff
    max_retry_delay: float = 60.0  # Maximum retry delay

    def get_delay(self) -> float:
        """Calculate delay to apply between calls."""
        if self.delay_between_calls > 0:
            return self.delay_between_calls
        if self.requests_per_minute and self.requests_per_minute > 0:
            # Calculate minimum delay to stay under RPM limit
            return 60.0 / self.requests_per_minute
        return 0.0


# Provider-specific default rate limit configs
ANTHROPIC_RATE_LIMITS = {
    # Tier 1 defaults (50 RPM)
    "tier1": RateLimitConfig(
        requests_per_minute=40,  # Stay below 50 RPM limit
        delay_between_calls=1.5,  # 1.5s between calls
        max_retries=5,
        initial_retry_delay=2.0,
        max_retry_delay=60.0,
    ),
    # Tier 2 defaults (1000 RPM)
    "tier2": RateLimitConfig(
        requests_per_minute=800,  # Stay below 1000 RPM limit
        delay_between_calls=0.1,  # 100ms between calls
        max_retries=3,
        initial_retry_delay=1.0,
        max_retry_delay=30.0,
    ),
}

# Current Anthropic tier (update based on your account)
ANTHROPIC_CURRENT_TIER = "tier2"

OPENAI_RATE_LIMITS = {
    # Default rate limits (usually higher than Anthropic)
    "default": RateLimitConfig(
        requests_per_minute=60,
        delay_between_calls=0.5,
        max_retries=3,
        initial_retry_delay=1.0,
        max_retry_delay=30.0,
    ),
}


class RateLimitedMixin:
    """Mixin to add rate limiting to any client."""

    _last_call_time: float = 0.0
    _rate_limit_config: Optional[RateLimitConfig] = None

    def _apply_rate_limit(self):
        """Apply rate limiting delay if configured."""
        if not self._rate_limit_config:
            return

        delay = self._rate_limit_config.get_delay()
        if delay <= 0:
            return

        elapsed = time.time() - self._last_call_time
        if elapsed < delay:
            sleep_time = delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_call_time = time.time()


def detect_provider(model: str) -> LLMProvider:
    """
    Auto-detect provider from model name.

    Args:
        model: Model name or alias

    Returns:
        Detected LLMProvider

    Raises:
        ValueError: If provider cannot be detected
    """
    # Resolve alias first
    resolved_model = MODEL_ALIASES.get(model, model)

    # Check each provider's model list
    for provider, models in PROVIDER_MODELS.items():
        for supported_model in models:
            if resolved_model.startswith(supported_model.split("-")[0]):
                # Match on prefix (e.g., "claude" matches "claude-3-5-sonnet-20241022")
                if provider == LLMProvider.ANTHROPIC and resolved_model.startswith("claude"):
                    return provider
                elif provider == LLMProvider.GOOGLE and resolved_model.startswith("gemini"):
                    return provider
                elif provider == LLMProvider.OPENAI and resolved_model.startswith("gpt"):
                    return provider

    # Default to OpenAI for unknown models
    logger.warning(f"Could not detect provider for model '{model}', defaulting to OpenAI")
    return LLMProvider.OPENAI


def resolve_model_name(model: str) -> str:
    """Resolve model alias to full model name."""
    return MODEL_ALIASES.get(model, model)


def create_instructor_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    rate_limit: Optional[RateLimitConfig] = None,
) -> Any:
    """
    Create an instructor-wrapped LLM client.

    Args:
        provider: Provider name ("openai", "anthropic", "google"). Auto-detected if None.
        model: Model name (used for auto-detection if provider is None)
        api_key: API key. Uses environment variable if None.
        rate_limit: Rate limiting configuration

    Returns:
        Instructor-wrapped client ready for structured extraction

    Raises:
        ValueError: If provider is invalid or required dependencies are missing
    """
    # Auto-detect provider from model if not specified
    if provider is None:
        if model is None:
            provider = LLMProvider.OPENAI.value
        else:
            provider = detect_provider(model).value

    provider_enum = LLMProvider(provider.lower())

    # Resolve model name for provider-specific handling
    resolved_model = resolve_model_name(model) if model else None

    if provider_enum == LLMProvider.OPENAI:
        return _create_openai_client(api_key, rate_limit)
    elif provider_enum == LLMProvider.ANTHROPIC:
        return _create_anthropic_client(api_key, rate_limit)
    elif provider_enum == LLMProvider.GOOGLE:
        # Gemini requires model to be set at client creation time
        return _create_google_client(api_key, rate_limit, resolved_model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _create_openai_client(
    api_key: Optional[str] = None,
    rate_limit: Optional[RateLimitConfig] = None,
) -> Any:
    """Create OpenAI instructor client."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    instructor_client = instructor.from_openai(client)

    # Wrap with rate limiting
    if rate_limit:
        instructor_client = _wrap_with_rate_limiting(instructor_client, rate_limit)

    return instructor_client


def _create_anthropic_client(
    api_key: Optional[str] = None,
    rate_limit: Optional[RateLimitConfig] = None,
) -> Any:
    """Create Anthropic instructor client with rate limit aware settings."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Install with: pip install anthropic"
        )

    # Use environment variable if api_key not provided
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Use current Anthropic tier defaults if no rate_limit provided
    if rate_limit is None:
        rate_limit = ANTHROPIC_RATE_LIMITS[ANTHROPIC_CURRENT_TIER]
        logger.info(f"Using Anthropic {ANTHROPIC_CURRENT_TIER} rate limits ({rate_limit.delay_between_calls}s delay, {rate_limit.requests_per_minute} RPM)")

    # Create client with built-in retry configuration
    # Anthropic SDK supports max_retries parameter
    client = anthropic.Anthropic(
        api_key=api_key,
        max_retries=rate_limit.max_retries,
    )
    instructor_client = instructor.from_anthropic(client)

    # Wrap with rate limiting
    instructor_client = _wrap_with_rate_limiting(instructor_client, rate_limit)

    return instructor_client


def _create_google_client(
    api_key: Optional[str] = None,
    rate_limit: Optional[RateLimitConfig] = None,
    model: Optional[str] = None,
) -> Any:
    """Create Google Gemini instructor client."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. Install with: pip install google-generativeai"
        )

    # Use environment variable if api_key not provided
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    # Default model if not specified
    if model is None:
        model = "gemini-2.0-flash"

    genai.configure(api_key=api_key)
    # Gemini requires model to be set at client creation time (not per-call like OpenAI)
    client = genai.GenerativeModel(model_name=model)
    instructor_client = instructor.from_gemini(client)

    # Wrap with rate limiting
    if rate_limit:
        instructor_client = _wrap_with_rate_limiting(instructor_client, rate_limit)

    return instructor_client


def _wrap_with_rate_limiting(client: Any, rate_limit: RateLimitConfig) -> Any:
    """
    Wrap an instructor client with rate limiting.

    This wraps the chat.completions.create method to add delays.
    """
    original_create = client.chat.completions.create
    last_call_time = [0.0]  # Use list for mutable closure

    def rate_limited_create(*args, **kwargs):
        delay = rate_limit.get_delay()
        if delay > 0:
            elapsed = time.time() - last_call_time[0]
            if elapsed < delay:
                sleep_time = delay - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

        result = original_create(*args, **kwargs)
        last_call_time[0] = time.time()
        return result

    client.chat.completions.create = rate_limited_create
    return client


def create_raw_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    rate_limit: Optional[RateLimitConfig] = None,
) -> Any:
    """
    Create a raw (non-instructor) LLM client for JSON mode calls.

    This is used by scoped_agentic.py which needs direct JSON mode.

    Args:
        provider: Provider name ("openai", "anthropic", "google")
        model: Model name (used for auto-detection)
        api_key: API key
        rate_limit: Rate limiting configuration

    Returns:
        Raw client (OpenAI, Anthropic, or Google client)
    """
    # Auto-detect provider from model if not specified
    if provider is None:
        if model is None:
            provider = LLMProvider.OPENAI.value
        else:
            provider = detect_provider(model).value

    provider_enum = LLMProvider(provider.lower())

    if provider_enum == LLMProvider.OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    elif provider_enum == LLMProvider.ANTHROPIC:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed")
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
    elif provider_enum == LLMProvider.GOOGLE:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed")
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        client = genai
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client


def _retry_with_backoff(
    func,
    rate_limit: Optional[RateLimitConfig] = None,
    provider: str = "openai",
) -> Any:
    """
    Execute a function with exponential backoff retry on rate limit errors.

    Args:
        func: Callable to execute
        rate_limit: Rate limit config with retry parameters
        provider: Provider name for error type detection

    Returns:
        Result from func()

    Raises:
        Last exception if all retries exhausted
    """
    import random

    max_retries = rate_limit.max_retries if rate_limit else 3
    initial_delay = rate_limit.initial_retry_delay if rate_limit else 2.0
    max_delay = rate_limit.max_retry_delay if rate_limit else 60.0

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if this is a rate limit error (429)
            is_rate_limit = (
                "429" in error_str or
                "rate" in error_str or
                "too many requests" in error_str or
                "overloaded" in error_str
            )

            if not is_rate_limit:
                # Not a rate limit error, re-raise immediately
                raise

            if attempt == max_retries:
                # Exhausted all retries
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise

            # Calculate backoff with jitter
            delay = min(initial_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0.5, 1.5)
            sleep_time = delay * jitter

            logger.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)

    raise last_exception


def call_llm_json(
    client: Any,
    provider: str,
    model: str,
    messages: list[dict],
    rate_limit: Optional[RateLimitConfig] = None,
    _last_call_time: list = [0.0],
) -> dict:
    """
    Make a JSON-mode LLM call with provider abstraction.

    This normalizes the API across OpenAI, Anthropic, and Google.
    Includes automatic retry with exponential backoff for rate limits.

    Args:
        client: Raw client from create_raw_client()
        provider: Provider name
        model: Model to use
        messages: Messages in OpenAI format [{"role": "user", "content": "..."}]
        rate_limit: Rate limiting config (uses provider defaults if None)

    Returns:
        Parsed JSON response as dict
    """
    import json

    # Use provider-specific defaults if no rate_limit provided
    if rate_limit is None:
        if provider.lower() == "anthropic":
            rate_limit = ANTHROPIC_RATE_LIMITS[ANTHROPIC_CURRENT_TIER]
        else:
            rate_limit = OPENAI_RATE_LIMITS["default"]

    # Apply rate limiting delay
    delay = rate_limit.get_delay()
    if delay > 0:
        elapsed = time.time() - _last_call_time[0]
        if elapsed < delay:
            sleep_time = delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

    provider_enum = LLMProvider(provider.lower())
    resolved_model = resolve_model_name(model)

    if provider_enum == LLMProvider.OPENAI:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        _last_call_time[0] = time.time()
        return json.loads(response.choices[0].message.content)

    elif provider_enum == LLMProvider.ANTHROPIC:
        # Convert messages format for Anthropic
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        # Add JSON instruction to prompt
        if user_messages:
            user_messages[-1]["content"] += "\n\nRespond with valid JSON only."

        def _make_anthropic_call():
            return client.messages.create(
                model=resolved_model,
                max_tokens=4096,
                system=system_msg,
                messages=user_messages,
            )

        # Use retry with backoff for rate limit handling
        response = _retry_with_backoff(
            _make_anthropic_call,
            rate_limit=rate_limit,
            provider="anthropic"
        )
        _last_call_time[0] = time.time()

        # Parse JSON from response
        content = response.content[0].text
        # Try to extract JSON from response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from Anthropic response: {content[:200]}")

    elif provider_enum == LLMProvider.GOOGLE:
        # Google Gemini uses a different API structure
        model_obj = client.GenerativeModel(model_name=resolved_model)

        # Combine messages into a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\nRespond with valid JSON only."

        response = model_obj.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
            }
        )
        _last_call_time[0] = time.time()

        return json.loads(response.text)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# =============================================================================
# Provider Info Utilities
# =============================================================================

def get_provider_info() -> dict:
    """Get information about supported providers and models."""
    return {
        "providers": {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "models": PROVIDER_MODELS[LLMProvider.OPENAI],
                "installed": _check_openai_installed(),
            },
            "anthropic": {
                "env_var": "ANTHROPIC_API_KEY",
                "models": PROVIDER_MODELS[LLMProvider.ANTHROPIC],
                "installed": _check_anthropic_installed(),
            },
            "google": {
                "env_var": "GOOGLE_API_KEY or GEMINI_API_KEY",
                "models": PROVIDER_MODELS[LLMProvider.GOOGLE],
                "installed": _check_google_installed(),
            },
        },
        "aliases": MODEL_ALIASES,
    }


def _check_openai_installed() -> bool:
    try:
        import openai
        return True
    except ImportError:
        return False


def _check_anthropic_installed() -> bool:
    try:
        import anthropic
        return True
    except ImportError:
        return False


def _check_google_installed() -> bool:
    try:
        import google.generativeai
        return True
    except ImportError:
        return False
