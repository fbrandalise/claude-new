"""
Calls the LLM to enrich product attributes using a given prompt template.
"""

import json
import httpx
import os
import re

from openai import OpenAI


def get_client() -> OpenAI:
    skip_ssl = os.getenv("SKIP_SSL_VERIFY", "").lower() in ("1", "true", "yes")
    http_client = httpx.Client(verify=not skip_ssl) if skip_ssl else None
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        http_client=http_client,
    )


def enrich_product(
    product_name: str,
    raw_description: str,
    prompt_template: str,
    model: str | None = None,
) -> dict:
    """
    Sends the product info through the prompt template and returns
    the extracted attributes as a dict.
    """
    model = model or os.getenv("ENRICHMENT_MODEL", "gpt-4.1-mini")
    client = get_client()

    prompt = prompt_template.format(
        product_name=product_name,
        raw_description=raw_description,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )

    raw_output = response.choices[0].message.content.strip()
    return _parse_json(raw_output), raw_output


def _parse_json(text: str) -> dict:
    """Try to parse JSON from the model output, handling markdown fences."""
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": text}
