"""Agente de analise tactica usando LLM + dados do banco.

Recebe perguntas sobre partidas, times ou ligas e responde
com analise baseada em dados reais + contexto do RAG.
"""
from loguru import logger

from prompts.templates import ANALYST_SYSTEM
from database.queries import get_team_recent_matches, get_h2h_stats, get_team_avg_stats
from models.features import load_matches, league_averages
from rag.retriever import Retriever
from agents.client import LLMClient


class AnalystAgent:
    """Analista tactico que consulta dados + RAG + LLM."""

    def __init__(self):
        self.retriever = Retriever(n_results=10)
        self.llm = LLMClient()

    def analyze_match(self, home_team: str, away_team: str,
                      league: str | None = None) -> str:
        """Analisa uma partida especifica."""
        # Dados do confronto direto
        h2h = get_h2h_stats(home_team, away_team)

        # Dados recentes de cada time
        home_recent = get_team_recent_matches(home_team, league, n=10)
        away_recent = get_team_recent_matches(away_team, league, n=10)

        # Medias dos times
        season = "2025/2026"
        home_avg = get_team_avg_stats(home_team, league or "", season, n=10)
        away_avg = get_team_avg_stats(away_team, league or "", season, n=10)

        # Contexto RAG
        rag_context = self.retriever.context_for_prompt(
            f"{home_team} x {away_team} {league}",
            league=league,
            n_matches=5,
        )

        prompt = self._build_match_prompt(
            home_team, away_team, league,
            h2h, home_recent, away_recent,
            home_avg, away_avg, rag_context,
        )

        return self._call_claude(prompt)

    def analyze_league(self, league: str) -> str:
        """Analisa tendencias de uma liga inteira."""
        df = load_matches(league=league)
        if df.empty:
            return f"Sem dados para {league}"

        avgs = league_averages(df)
        league_stats = avgs.get(league, {})

        # Times com mais/menos escanteios
        team_corners = []
        for team in df["home_team"].unique():
            tm = df[(df["home_team"] == team) | (df["away_team"] == team)]
            team_corners.append((team, tm["total_corners"].mean()))
        team_corners.sort(key=lambda x: -x[1])

        prompt = (
            f"Analise a liga {league} com base nos seguintes dados:\n\n"
            f"Media de escanteios por jogo: {league_stats.get('avg_total_corners', 'N/A'):.1f}\n"
            f"Media de finalizacoes por jogo: {league_stats.get('avg_total_shots', 'N/A'):.1f}\n"
            f"Media gols casa: {league_stats.get('avg_home_goals', 'N/A'):.2f}\n"
            f"Media gols fora: {league_stats.get('avg_away_goals', 'N/A'):.2f}\n\n"
            f"Times com mais escanteios:\n"
        )
        for team, avg in team_corners[:5]:
            prompt += f"  {team}: {avg:.1f}\n"
        prompt += f"\nTimes com menos escanteios:\n"
        for team, avg in team_corners[-5:]:
            prompt += f"  {team}: {avg:.1f}\n"

        prompt += (
            f"\nPergunta: Quais sao as tendencias e caracteristicas "
            f"da {league} em termos de escanteios e finalizacoes? "
            f"Que times se destacam? Ha alguma peculiaridade da liga?"
        )

        return self._call_claude(prompt, system=ANALYST_SYSTEM)

    def _build_match_prompt(self, home: str, away: str, league: str | None,
                             h2h: dict, home_recent: list, away_recent: list,
                             home_avg: dict, away_avg: dict,
                             rag_context: str) -> str:
        prompt = f"Analise tactica: {home} x {away}"
        if league:
            prompt += f" ({league})"
        prompt += "\n\n"

        # Confronto direto
        if h2h.get("total", 0) > 0:
            prompt += (
                f"CONFRONTO DIRETO ({h2h['total']} jogos):\n"
                f"  Vitorias {home}: {h2h.get('wins_a', 0)}\n"
                f"  Empates: {h2h.get('draws', 0)}\n"
                f"  Vitorias {away}: {h2h.get('wins_b', 0)}\n"
                f"  Media escanteios totais: {h2h.get('avg_corners_total', 'N/A'):.1f}\n"
                f"  Media finalizacoes totais: {h2h.get('avg_shots_total', 'N/A'):.1f}\n\n"
            )

        # Dados recentes
        prompt += f"ULTIMOS 10 JOGOS:\n"
        for team, recent in [(home, home_recent), (away, away_recent)]:
            if recent:
                gf = sum(r.get("home_goals", 0) if r.get("home_team") == team
                         else r.get("away_goals", 0) for r in recent[:5])
                ga = sum(r.get("away_goals", 0) if r.get("home_team") == team
                         else r.get("home_goals", 0) for r in recent[:5])
                prompt += (
                    f"  {team} (ultimos 5): {gf} gols marcados, {ga} sofridos\n"
                )
        prompt += "\n"

        # Medias do time
        if home_avg.get("avg_corners_for"):
            prompt += (
                f"MEDIAS {home}:\n"
                f"  Escanteios a favor: {home_avg.get('avg_corners_for', 'N/A'):.1f}\n"
                f"  Escanteios contra: {home_avg.get('avg_corners_against', 'N/A'):.1f}\n"
                f"  Finalizacoes a favor: {home_avg.get('avg_shots_for', 'N/A'):.1f}\n"
                f"  Finalizacoes contra: {home_avg.get('avg_shots_against', 'N/A'):.1f}\n"
                f"  xG a favor: {home_avg.get('avg_xg_for', 'N/A'):.2f}\n"
            )
        if away_avg.get("avg_corners_for"):
            prompt += (
                f"\nMEDIAS {away}:\n"
                f"  Escanteios a favor: {away_avg.get('avg_corners_for', 'N/A'):.1f}\n"
                f"  Escanteios contra: {away_avg.get('avg_corners_against', 'N/A'):.1f}\n"
                f"  Finalizacoes a favor: {away_avg.get('avg_shots_for', 'N/A'):.1f}\n"
                f"  Finalizacoes contra: {away_avg.get('avg_shots_against', 'N/A'):.1f}\n"
                f"  xG a favor: {away_avg.get('avg_xg_for', 'N/A'):.2f}\n"
            )

        prompt += f"\nCONTEXTO ADICIONAL:\n{rag_context}\n\n"
        prompt += (
            "Com base nesses dados, faca uma analise tactica da partida. "
            "Destaque padroes de escanteios e finalizacoes, "
            "pontos fortes e fracos de cada time, e o que esperar do jogo."
        )

        return prompt

    def _call_claude(self, prompt: str,
                     system: str = ANALYST_SYSTEM,
                     max_tokens: int = 1024) -> str:
        """Chama LLM com o prompt."""
        return self.llm.call(system=system, prompt=prompt, max_tokens=max_tokens)
