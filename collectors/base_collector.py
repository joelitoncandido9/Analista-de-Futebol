"""Classe base para coletores com retry, rate limit e logs."""
import time
import random
from datetime import datetime
from typing import Any

import requests
from loguru import logger


class BaseCollector:
    """Classe base para todos os coletores de dados."""

    def __init__(self, name: str, rate_per_min: int = 10):
        self.name = name
        self.rate_per_min = rate_per_min
        self._last_request = 0
        self._min_interval = 60.0 / rate_per_min

    def _rate_limit(self):
        """Aguarda o intervalo minimo entre requisicoes."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _request(
        self,
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
        retries: int = 3,
        timeout: int = 30,
    ) -> dict | None:
        """Faz request com retry e backoff exponencial."""
        for attempt in range(retries):
            self._rate_limit()
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=timeout)
                if resp.status_code == 429:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit em {url}, aguardando {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    logger.warning(f"HTTP {resp.status_code} em {url}")
                    time.sleep(2 ** attempt)
                    continue
                return resp
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout em {url} (tentativa {attempt + 1}/{retries})")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Erro em {url}: {e}")
                time.sleep(2 ** attempt)
        logger.error(f"Falha apos {retries} tentativas: {url}")
        return None

    def log_start(self, action: str):
        logger.info(f"[{self.name}] Iniciando {action}...")

    def log_success(self, action: str, count: int):
        logger.info(f"[{self.name}] {action} concluido: {count} registros")

    def log_error(self, action: str, error: Any):
        logger.error(f"[{self.name}] Erro em {action}: {error}")
