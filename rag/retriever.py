"""Retriever para busca semantica no ChromaDB.

Busca partidas, times e jogadores por similaridade semantica
para alimentar os agentes com contexto relevante.
"""
from typing import Any

from loguru import logger

from rag.vectorstore import get_client, get_or_create_collection


class Retriever:
    """Buscador semantico no vectorstore."""

    def __init__(self, n_results: int = 10):
        self.client = get_client()
        self.n_results = n_results

    def search_matches(self, query: str, league: str | None = None,
                       n_results: int | None = None) -> list[dict[str, Any]]:
        """Busca partidas por similaridade semantica.

        Args:
            query: Descricao textual (ex: "jogos com muitos escanteios do Liverpool em casa")
            league: Filtrar por liga (opcional)
            n_results: Quantos resultados

        Returns:
            Lista de dicts com documentos + metadados.
        """
        collection = get_or_create_collection(self.client, "matches")
        n = n_results or self.n_results

        where = {"type": "match"}
        if league:
            where["league"] = league

        results = collection.query(
            query_texts=[query],
            n_results=n,
            where=where,
        )

        return self._format_results(results)

    def search_teams(self, query: str, n_results: int | None = None) -> list[dict[str, Any]]:
        """Busca times por similaridade."""
        collection = get_or_create_collection(self.client, "teams")
        n = n_results or self.n_results

        results = collection.query(
            query_texts=[query],
            n_results=n,
            where={"type": "team"},
        )

        return self._format_results(results)

    def get_team_matches(self, team: str, league: str | None = None,
                         n_results: int = 20) -> list[dict[str, Any]]:
        """Busca partidas de um time especifico."""
        return self.search_matches(
            f"Partidas do {team}",
            league=league,
            n_results=n_results,
        )

    def get_recent_matches(self, league: str, n_results: int = 10) -> list[dict[str, Any]]:
        """Busca partidas recentes de uma liga."""
        return self.search_matches(
            f"Partidas recentes de {league}",
            league=league,
            n_results=n_results,
        )

    def get_high_corner_matches(self, league: str | None = None,
                                 n_results: int = 10) -> list[dict[str, Any]]:
        """Busca partidas com muitos escanteios."""
        return self.search_matches(
            "Partida com muitos escanteios, total acima de 13 escanteios",
            league=league,
            n_results=n_results,
        )

    def get_high_xg_matches(self, league: str | None = None,
                             n_results: int = 10) -> list[dict[str, Any]]:
        """Busca partidas com alto xG."""
        return self.search_matches(
            "Partida com alto xG, muitas chances criadas, times atacando muito",
            league=league,
            n_results=n_results,
        )

    @staticmethod
    def _format_results(raw: dict) -> list[dict[str, Any]]:
        """Formata resultados brutos do ChromaDB."""
        formatted = []
        if not raw.get("ids") or not raw["ids"][0]:
            return formatted

        for i, doc_id in enumerate(raw["ids"][0]):
            formatted.append({
                "id": doc_id,
                "score": raw["distances"][0][i] if raw.get("distances") else None,
                "text": raw["documents"][0][i] if raw.get("documents") else "",
                "metadata": raw["metadatas"][0][i] if raw.get("metadatas") else {},
            })

        return formatted

    def context_for_prompt(self, query: str, league: str | None = None,
                           n_matches: int = 5, n_teams: int = 3) -> str:
        """Gera contexto textual formatado para alimentar prompts de agentes.

        Returns:
            Texto formatado com contexto relevante.
        """
        parts = ["=== CONTEXTO RECUPERADO ==="]

        # Partidas relevantes
        matches = self.search_matches(query, league=league, n_results=n_matches)
        if matches:
            parts.append(f"\n--- Partidas Relevantes ({len(matches)}) ---")
            for m in matches:
                meta = m.get("metadata", {})
                parts.append(
                    f"  {meta.get('home_team', '?')} x {meta.get('away_team', '?')} "
                    f"({meta.get('date', '?')}) - {meta.get('league', '?')} | "
                    f"Placar: {meta.get('home_goals', '?')}-{meta.get('away_goals', '?')} | "
                    f"Esc: {meta.get('home_corners', '?')}-{meta.get('away_corners', '?')} | "
                    f"xG: {meta.get('home_xg', '?')}-{meta.get('away_xg', '?')}"
                )

        # Times
        teams = self.search_teams(query, n_results=n_teams)
        if teams:
            parts.append(f"\n--- Times ({len(teams)}) ---")
            for t in teams:
                parts.append(f"  {t.get('text', '')}")

        return "\n".join(parts)
