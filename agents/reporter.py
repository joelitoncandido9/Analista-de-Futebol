"""Agente reporter: relatorios pre-jogo e pos-jogo automatizados.

Combina dados do banco, previsoes dos modelos e contexto RAG
para gerar relatorios completos.
"""
from datetime import datetime

from loguru import logger

from database.queries import get_team_recent_matches, get_h2h_stats, get_team_avg_stats
from models.corners.corners_predictor import CornersPredictor
from models.shots.shots_predictor import ShotsPredictor
from models.results.dixon_coles import DixonColes
from prompts.templates import REPORTER_SYSTEM
from agents.client import LLMClient


class ReporterAgent:
    """Gera relatorios pre-jogo e pos-jogo."""

    def __init__(self):
        self.llm = LLMClient()

    def _get_corners_pred(self, league: str):
        return CornersPredictor(league=league)

    def _get_shots_pred(self, league: str):
        return ShotsPredictor(league=league)

    def pre_match(self, home_team: str, away_team: str,
                  league: str, match_date: str | None = None) -> str:
        """Gera relatorio pre-jogo completo."""
        if not match_date:
            match_date = datetime.now().strftime("%Y-%m-%d")

        # Previsoes dos modelos
        corners = self._get_corners_pred(league).predict(home_team, away_team, league, match_date)
        shots = self._get_shots_pred(league).predict(home_team, away_team, league, match_date)

        # Dixon-Coles
        dc = DixonColes(league=league)
        dc_loaded = dc.load()
        if dc_loaded:
            result_pred = dc.predict_score(home_team, away_team)
        else:
            result_pred = None

        # Dados do banco
        h2h = get_h2h_stats(home_team, away_team)
        home_recent = get_team_recent_matches(home_team, league, n=5)
        away_recent = get_team_recent_matches(away_team, league, n=5)
        home_avg = get_team_avg_stats(home_team, league, "2025/2026", n=10)
        away_avg = get_team_avg_stats(away_team, league, "2025/2026", n=10)

        prompt = self._build_pre_match_prompt(
            home_team, away_team, league, match_date,
            corners, shots, result_pred,
            h2h, home_recent, away_recent, home_avg, away_avg,
        )

        return self._call_claude(prompt)

    def post_match(self, home_team: str, away_team: str,
                   league: str, match_date: str | None = None) -> str:
        """Gera relatorio pos-jogo."""
        if not match_date:
            match_date = datetime.now().strftime("%Y-%m-%d")

        # Buscar dados da partida no banco
        from database.schema import get_conn
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM matches
               WHERE home_team = ? AND away_team = ? AND league = ?
               AND match_date LIKE ?""",
            (home_team, away_team, league, f"{match_date[:10]}%"),
        )
        match = dict(cur.fetchone()) if cur.fetchone() else None
        conn.close()

        if not match:
            return f"Partida {home_team} x {away_team} em {match_date} nao encontrada."

        prompt = self._build_post_match_prompt(match)
        return self._call_claude(prompt)

    def _build_pre_match_prompt(self, home: str, away: str, league: str,
                                 date: str, corners: dict | None,
                                 shots: dict | None, result: dict | None,
                                 h2h: dict, home_recent: list, away_recent: list,
                                 home_avg: dict, away_avg: dict) -> str:
        prompt = f"RELATORIO PRE-JOGO: {home} x {away}\n"
        prompt += f"Data: {date} | Liga: {league}\n\n"

        # Previsoes do modelo
        prompt += "--- PREVISOES DOS MODELOS ---\n"
        if corners:
            prompt += (
                f"Escanteios totais previstos: {corners['predicted_total_corners']}\n"
            )
            for line, prob in list(corners.get("probabilities", {}).items())[:4]:
                prompt += f"  {line}: {prob:.1%}\n"

        if shots:
            prompt += (
                f"\nFinalizacoes totais previstas: {shots['predicted_total_shots']}\n"
            )
            for line, prob in list(shots.get("probabilities", {}).items())[:4]:
                prompt += f"  {line}: {prob:.1%}\n"

        if result:
            prompt += (
                f"\nResultado mais provavel: {result['most_likely_score']} "
                f"({result['most_likely_prob']:.1%})\n"
                f"  Casa: {result['prob_home']:.1%} | "
                f"Empate: {result['prob_draw']:.1%} | "
                f"Fora: {result['prob_away']:.1%}\n"
                f"  xG esperado: {result['expected_home_goals']} x {result['expected_away_goals']}\n"
            )

        # H2H
        if h2h.get("total", 0) > 0:
            prompt += f"\n--- CONFRONTO DIRETO ({h2h['total']} jogos) ---\n"
            prompt += (
                f"  {home}: {h2h.get('wins_a', 0)}V | "
                f"Empates: {h2h.get('draws', 0)} | "
                f"{away}: {h2h.get('wins_b', 0)}V\n"
            )
            if h2h.get("avg_corners_total"):
                prompt += f"  Media escanteios: {h2h['avg_corners_total']:.1f}\n"
            if h2h.get("avg_shots_total"):
                prompt += f"  Media finalizacoes: {h2h['avg_shots_total']:.1f}\n"

        # Forma recente
        prompt += f"\n--- FORMA RECENTE ---\n"
        for team, recent in [(home, home_recent), (away, away_recent)]:
            if recent:
                pts = 0
                gf = 0
                ga = 0
                for r in recent:
                    if r.get("home_team") == team:
                        gf += r.get("home_goals", 0) or 0
                        ga += r.get("away_goals", 0) or 0
                        if (r.get("home_goals") or 0) > (r.get("away_goals") or 0):
                            pts += 3
                        elif (r.get("home_goals") or 0) == (r.get("away_goals") or 0):
                            pts += 1
                    else:
                        gf += r.get("away_goals", 0) or 0
                        ga += r.get("home_goals", 0) or 0
                        if (r.get("away_goals") or 0) > (r.get("home_goals") or 0):
                            pts += 3
                        elif (r.get("away_goals") or 0) == (r.get("home_goals") or 0):
                            pts += 1
                prompt += f"  {team}: {pts} pts, {gf} GF, {ga} GA (ultimos 5)\n"

        # Medias do time
        prompt += f"\n--- MEDIAS DO TIME (ultimos 10) ---\n"
        if home_avg.get("avg_corners_for"):
            prompt += (
                f"  {home}: esc {home_avg['avg_corners_for']:.1f} for / "
                f"{home_avg['avg_corners_against']:.1f} contra | "
                f"chutes {home_avg['avg_shots_for']:.1f} for / "
                f"{home_avg['avg_shots_against']:.1f} contra\n"
            )
        if away_avg.get("avg_corners_for"):
            prompt += (
                f"  {away}: esc {away_avg['avg_corners_for']:.1f} for / "
                f"{away_avg['avg_corners_against']:.1f} contra | "
                f"chutes {away_avg['avg_shots_for']:.1f} for / "
                f"{away_avg['avg_shots_against']:.1f} contra\n"
            )

        prompt += (
            "\nCom base em todos esses dados, gere um relatorio pre-jogo completo. "
            "Inclua: contexto da partida, forma recente, chave tactica, "
            "previsoes dos modelos e recomendacoes."
        )

        return prompt

    def _build_post_match_prompt(self, match: dict) -> str:
        home = match.get("home_team", "?")
        away = match.get("away_team", "?")
        prompt = f"RELATORIO POS-JOGO: {home} x {away}\n"
        prompt += f"Data: {str(match.get('match_date', ''))[:10]}\n"
        prompt += f"Liga: {match.get('league', '?')}\n\n"

        prompt += "--- RESULTADO ---\n"
        prompt += f"  {home} {match.get('home_goals', '?')} - {match.get('away_goals', '?')} {away}\n\n"

        prompt += "--- ESTATISTICAS ---\n"
        for stat in ["corners", "shots", "shots_on_target",
                       "fouls", "yellow", "red", "possession",
                       "xg", "ppda", "deep"]:
            h_val = match.get(f"home_{stat}")
            a_val = match.get(f"away_{stat}")
            if h_val is not None:
                prompt += f"  {stat}: {home} {h_val} | {away} {a_val}\n"

        prompt += (
            "\nCom base nesses dados, gere um relatorio pos-jogo. "
            "Inclua: resumo do jogo, analise das estatisticas, "
            "destaques individuais, e licoes para proximos jogos."
        )

        return prompt

    def _call_claude(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.llm.call(system=REPORTER_SYSTEM, prompt=prompt, max_tokens=max_tokens)
