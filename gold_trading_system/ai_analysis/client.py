"""
AI Client — wraps the actual model call. Fail-safe by design: ANY failure
(network error, malformed JSON, schema violation) returns AISignal.no_trade_fallback(),
never raises up into the trading loop and never silently fails open into a trade.
"""
import json
import logging
from typing import Callable

from ai_analysis.schema import AISignal
from ai_analysis.prompt_builder import SYSTEM_PROMPT

logger = logging.getLogger("ai_analysis")


class AIClient:
    """
    caller: injectable function(system_prompt, user_context) -> raw_text_response.
    Defaults to a real Anthropic API call; tests inject a mock/stub instead.
    This keeps the fail-safe logic testable without needing live credentials.
    """

    def __init__(self, caller: Callable[[str, str], str] | None = None, max_retries: int = 1):
        self.caller = caller or self._default_caller
        self.max_retries = max_retries

    def _default_caller(self, system_prompt: str, user_context: str) -> str:
        # Real implementation would call the Anthropic API here. Not wired
        # in this build phase — the interface is what matters for now.
        raise NotImplementedError(
            "Default AI caller not wired yet — inject a real caller when ready to go live."
        )

    def get_signal(self, context: str) -> AISignal:
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.caller(SYSTEM_PROMPT, context)
            except Exception as e:
                last_error = f"API call failed: {type(e).__name__}: {e}"
                logger.warning(last_error)
                continue

            cleaned = self._strip_markdown_fences(raw)

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = f"Response was not valid JSON: {e}"
                logger.warning(last_error)
                continue

            try:
                signal = AISignal(**parsed)
                return signal
            except Exception as e:
                last_error = f"Response did not match AISignal schema: {e}"
                logger.warning(last_error)
                continue

        # every attempt failed — fail-safe, never fail-open
        return AISignal.no_trade_fallback(reason=last_error)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
        return t
