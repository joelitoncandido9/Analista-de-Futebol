"""Configuracao do ChromaDB para busca semantica de partidas e jogadores.

Usa OpenAI embeddings para indexar descricoes de partidas,
estatisticas de times e perfis de jogadores.
"""
from __future__ import annotations
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from loguru import logger

from config.settings import DATA_DIR


CHROMA_DIR = DATA_DIR / "chroma"

# Nomes das colecoes
COLL_MATCHES = "matches"
COLL_TEAMS = "teams"
COLL_PLAYERS = "players"


def get_client() -> chromadb.PersistentClient:
    """Retorna cliente ChromaDB persistente."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient | None = None,
                              name: str = COLL_MATCHES) -> chromadb.Collection:
    """Obtem ou cria colecao no ChromaDB."""
    if client is None:
        client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _safe_metadata(value) -> str | int | float | bool | None:
    """Converte valores para tipos aceitos pelo ChromaDB metadata."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=str)
    return str(value)


def _make_metadata(d: dict) -> dict:
    """Limpa dict para metadata do ChromaDB."""
    return {k: _safe_metadata(v) for k, v in d.items()}
