"""
Patches the OpenAI client instantiation to skip SSL verification
when SKIP_SSL_VERIFY is enabled. Covers both our enrichment client
and the DeepEval judge client.
"""

import os


def apply_ssl_patch():
    """Call once at app startup when SKIP_SSL_VERIFY is truthy."""
    skip = os.getenv("SKIP_SSL_VERIFY", "").lower() in ("1", "true", "yes")
    if not skip:
        return

    import httpx
    from deepeval.models.llms.openai_model import GPTModel

    _original_build_client = GPTModel._build_client

    def _patched_build_client(self, cls):
        from openai import AsyncOpenAI, OpenAI

        if cls is AsyncOpenAI:
            http_client = httpx.AsyncClient(verify=False)
        else:
            http_client = httpx.Client(verify=False)

        from deepeval.models.utils import require_secret_api_key

        api_key = require_secret_api_key(
            self.api_key,
            provider_label="OpenAI",
            env_var_name="OPENAI_API_KEY",
            param_hint="`api_key` to GPTModel(...)",
        )

        kw = dict(
            api_key=api_key,
            base_url=self.base_url,
            http_client=http_client,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise

    GPTModel._build_client = _patched_build_client
