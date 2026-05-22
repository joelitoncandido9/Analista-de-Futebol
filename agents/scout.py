"""Agente de scouting com percentil calculado manualmente via SQL.

Avalia jogadores comparando estatisticas com a base de dados,
calculando percentis e gerando relatorios de scout.
"""
import numpy as np
from loguru import logger

from database.schema import get_conn
from prompts.templates import SCOUT_SYSTEM
from agents.client import LLMClient


class ScoutAgent:
    """Olheiro (scout) que avalia jogadores com dados + percentis."""

    def __init__(self):
        self.llm = LLMClient()

    def scout_player(self, player_name: str, league: str | None = None) -> str:
        """Gera relatorio de scout para um jogador."""
        stats = self._get_player_stats(player_name, league)
        if not stats:
            return f"Jogador '{player_name}' nao encontrado no banco."

        percentis = self._calc_percentis(stats, league)

        prompt = self._build_scout_prompt(player_name, league, stats, percentis)
        return self._call_claude(prompt)

    def _get_player_stats(self, player_name: str,
                           league: str | None = None) -> list[dict] | None:
        """Busca estatisticas do jogador no banco."""
        conn = get_conn()
        cur = conn.cursor()

        query = """SELECT pm.*, m.match_date, m.league, m.home_team, m.away_team
                   FROM player_match_stats pm
                   JOIN matches m ON pm.match_id = m.match_id
                   WHERE pm.player_name LIKE ?"""
        params = [f"%{player_name}%"]

        if league:
            query += " AND m.league = ?"
            params.append(league)

        query += " ORDER BY m.match_date DESC LIMIT 20"
        cur.execute(query, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows if rows else None

    def _calc_percentis(self, stats: list[dict],
                        league: str | None = None) -> dict:
        """Calcula percentis do jogador vs base da liga."""
        if not stats:
            return {}

        # Agregar stats do jogador
        n = len(stats)
        totals = {
            "goals": sum(s.get("goals", 0) or 0 for s in stats),
            "assists": sum(s.get("assists", 0) or 0 for s in stats),
            "shots": sum(s.get("shots", 0) or 0 for s in stats),
            "shots_on_target": sum(s.get("shots_on_target", 0) or 0 for s in stats),
            "dribbles": sum(s.get("dribbles", 0) or 0 for s in stats),
            "tackles": sum(s.get("tackles", 0) or 0 for s in stats),
            "fouls": sum(s.get("fouls", 0) or 0 for s in stats),
            "key_passes": sum(s.get("key_passes", 0) or 0 for s in stats),
        }

        # Ratings validos
        ratings = [s.get("rating") for s in stats
                   if s.get("rating") is not None]
        avg_rating = float(np.mean(ratings)) if ratings else None
        minutes = sum(s.get("minutes", 0) or 0 for s in stats)

        # Buscar medias da liga para comparacao
        conn = get_conn()
        cur = conn.cursor()
        league_filter = league or stats[0].get("league")

        league_avgs = {}
        for stat in ["goals", "assists", "shots", "shots_on_target",
                      "dribbles", "tackles", "fouls", "key_passes", "rating"]:
            cur.execute(
                f"SELECT AVG({stat}) as avg, "
                f"STDDEV({stat}) as std FROM player_match_stats "
                f"WHERE {stat} IS NOT NULL AND {stat} > 0"
            )
            row = cur.fetchone()
            league_avgs[stat] = {
                "avg": row[0] or 0,
                "std": row[1] or 0,
            }
        conn.close()

        # Calcular percentis (assumindo normal)
        percentis = {}
        for stat, value in totals.items():
            avg = league_avgs.get(stat, {}).get("avg", 0)
            std = league_avgs.get(stat, {}).get("std", 0)
            if std > 0 and n > 0:
                per_game = value / n
                z = (per_game - avg) / std
                percentil = 0.5 + 0.5 * (2 / (1 + np.exp(-0.7 * z)) - 1)
                percentis[stat] = {
                    "per_game": round(per_game, 2),
                    "total": value,
                    "percentil": round(min(max(percentil, 0.01), 0.99), 3),
                    "liga_avg": round(avg, 2),
                }

        return {
            "n_jogos": n,
            "minutos": minutes,
            "avg_rating": avg_rating,
            "percentis": percentis,
        }

    def _build_scout_prompt(self, player: str, league: str | None,
                             stats: list[dict], percentis: dict) -> str:
        prompt = f"Relatorio de Scout: {player}"
        if league:
            prompt += f" ({league})"
        prompt += "\n\n"

        # Ultimos jogos
        prompt += f"ULTIMOS {len(stats)} JOGOS:\n"
        for s in stats[:10]:
            date = str(s.get("match_date", "")[:10])
            team = s.get("team", "?")
            opponent = s.get("away_team") if s.get("team") == s.get("home_team") else s.get("home_team")
            rating = s.get("rating", "N/A")
            goals = s.get("goals", 0) or ""
            assists = s.get("assists", 0) or ""
            prompt += f"  {date} | {team} vs {opponent} | "
            prompt += f"Nota: {rating}"
            if goals:
                prompt += f" Gols: {goals}"
            if assists:
                prompt += f" Assist: {assists}"
            prompt += "\n"

        # Percentis
        p = percentis.get("percentis", {})
        if p:
            prompt += f"\nPERCENTIS ({percentis.get('n_jogos', 0)} jogos, "
            prompt += f"{percentis.get('minutos', 0)} min):\n"
            top_stats = []
            for stat, data in sorted(p.items(), key=lambda x: -x[1]["percentil"]):
                label = f"Top {int((1-data['percentil'])*100)}%" if data['percentil'] > 0.5 else f"Bottom {int(data['percentil']*100)}%"
                prompt += (
                    f"  {stat}: {data['per_game']}/jogo "
                    f"(liga: {data['liga_avg']}/jogo) - {label}\n"
                )
                if data["percentil"] > 0.7:
                    top_stats.append(stat)

            if percentis.get("avg_rating"):
                prompt += f"\n  Media de rating: {percentis['avg_rating']:.2f}\n"

        prompt += (
            f"\nCom base nesses dados, faca uma avaliacao de scout do jogador. "
            f"Destaque: forcas, fraquezas, estilo de jogo, "
            f"compatibilidade tactica e potencial. "
            f"Use os percentis para contextualizar."
        )

        return prompt

    def _call_claude(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.llm.call(system=SCOUT_SYSTEM, prompt=prompt, max_tokens=max_tokens)
