"""Cliente LLM compartilhado usando OpenRouter.

Centraliza a chamada a API para facilitar troca de provider.
Usa OpenRouter por padrao, com fallback para Anthropic direto.
"""
from loguru import logger

from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


class LLMClient:
    """Cliente para chamadas LLM via OpenRouter."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is not None:
            return self._client
        if OPENROUTER_API_KEY:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
            logger.debug("[LLM] Usando OpenRouter")
        else:
            from config.settings import ANTHROPIC_API_KEY
            if ANTHROPIC_API_KEY:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.debug("[LLM] Usando Anthropic direto (fallback)")
            else:
                self._client = None
        return self._client

    def call(self, system: str, prompt: str, max_tokens: int = 1024,
             temperature: float = 0.3) -> str:
        """Chama o LLM com system prompt e mensagem do usuario.

        Args:
            system: System prompt
            prompt: Mensagem do usuario
            max_tokens: Max tokens na resposta
            temperature: Temperature (0-1)

        Returns:
            Texto da resposta.
        """
        client = self.client
        if client is None:
            return (
                "API nao configurada. Defina OPENROUTER_API_KEY ou "
                "ANTHROPIC_API_KEY no .env\n\n"
                f"Dados locais:\n{prompt[:600]}..."
            )

        model = OPENROUTER_MODEL

        try:
            if hasattr(client, "messages") and hasattr(client.messages, "create"):
                # OpenAI-compatible (OpenRouter)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/palpites/football_ai",
                        "X-Title": "Football AI",
                    },
                )
                return response.choices[0].message.content or ""
            else:
                # Anthropic direto (fallback)
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

        except Exception as e:
            logger.error(f"[LLM] Erro na chamada: {e}")
            return f"Erro ao chamar LLM: {e}"
