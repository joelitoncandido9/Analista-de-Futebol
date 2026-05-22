"""Detector de value bets: compara probabilidades do modelo vs odds do mercado.

Encontra apostas com valor esperado positivo (EV+) comparando
nossas previsoes com as odds disponiveis no mercado.

Funciona para:
- Escanteios: over/under lines
- Finalizacoes: over/under lines
- Resultados: 1X2 (casa, empate, fora)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class ValueBet:
    """Uma aposta com valor esperado positivo."""
    market: str           # ex: 'corners_over_9.5', 'match_result'
    selection: str        # ex: 'Over 9.5', 'Home', 'Draw'
    fixture_id: str = ""
    home_team: str = ""
    away_team: str = ""
    league: str = ""
    match_date: str = ""
    model_prob: float = 0.0       # Probabilidade estimada pelo modelo
    market_odds: float = 0.0      # Odd do mercado (decimal)
    fair_odds: float = 0.0        # Odd justa (1 / model_prob)
    edge: float = 0.0             # Vantagem percentual
    expected_value: float = 0.0   # EV em percentual
    stake_pct: float = 0.0        # Percentual do bankroll (Kelly)
    model_prediction: float = 0.0 # Valor previsto (ex: 10.5 escanteios)


class ValueBetDetector:
    """Detecta value bets comparando modelos vs odds de mercado.

    Args:
        min_edge: Edge minimo para considerar value bet (0.05 = 5%).
        min_ev: EV minimo (0.05 = 5%).
        kelly_fraction: Fracao do Kelly Criterion (0.25 = conservative).
    """

    def __init__(self, min_edge: float = 0.05,
                 min_ev: float = 0.05,
                 kelly_fraction: float = 0.25):
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.kelly_fraction = kelly_fraction
        self.bets: list[ValueBet] = []

    def check_corners(self, model_pred: dict,
                      market_odds: dict | None = None) -> list[ValueBet]:
        """Avalia value bets em escanteios.

        Args:
            model_pred: Saida do CornersPredictor.predict()
            market_odds: Dict com odds do mercado. Ex: {'over_9.5': 1.85, 'under_9.5': 2.00}

        Returns:
            Lista de ValueBet com EV+.
        """
        found: list[ValueBet] = []
        probs = model_pred.get("probabilities", {})

        if not market_odds:
            return found

        for line_key, market_odd in market_odds.items():
            if not isinstance(market_odd, (int, float)) or market_odd <= 1.0:
                continue

            # Extrair linha e direcao do market_key
            parts = line_key.split("_")
            if len(parts) < 2:
                continue
            direction = parts[0]  # 'over' ou 'under'
            line_value = parts[1] if len(parts) > 1 else ""

            model_prob_key = f"{direction}_{line_value}"
            model_prob = probs.get(model_prob_key, 0.0)

            if model_prob <= 0.0:
                continue

            model_prob = min(model_prob, 0.99)
            fair_odd = 1.0 / model_prob
            edge = (fair_odd - market_odd) / market_odd

            if edge > self.min_edge:
                # Kelly stake
                kelly = ((model_prob * (market_odd - 1)) - (1 - model_prob)) / (market_odd - 1)
                kelly = max(0.0, min(kelly * self.kelly_fraction, 0.25))

                ev = model_prob * market_odd - 1.0

                if ev > self.min_ev:
                    bet = ValueBet(
                        market=f"corners_{line_key}",
                        selection=f"{direction.title()} {line_value}",
                        fixture_id=model_pred.get("fixture_id", ""),
                        home_team=model_pred.get("home_team", ""),
                        away_team=model_pred.get("away_team", ""),
                        league=model_pred.get("league", ""),
                        match_date=model_pred.get("match_date", ""),
                        model_prob=round(model_prob, 4),
                        market_odds=round(market_odd, 2),
                        fair_odds=round(fair_odd, 2),
                        edge=round(edge, 4),
                        expected_value=round(ev, 4),
                        stake_pct=round(kelly * 100, 2),
                        model_prediction=model_pred.get("predicted_total_corners", 0),
                    )
                    found.append(bet)

        return found

    def check_shots(self, model_pred: dict,
                    market_odds: dict | None = None) -> list[ValueBet]:
        """Avalia value bets em finalizacoes."""
        found: list[ValueBet] = []
        probs = model_pred.get("probabilities", {})

        if not market_odds:
            return found

        for line_key, market_odd in market_odds.items():
            if not isinstance(market_odd, (int, float)) or market_odd <= 1.0:
                continue

            parts = line_key.split("_")
            if len(parts) < 2:
                continue
            direction = parts[0]
            line_value = parts[1] if len(parts) > 1 else ""

            model_prob = probs.get(f"{direction}_{line_value}", 0.0)
            if model_prob <= 0.0:
                continue

            model_prob = min(model_prob, 0.99)
            fair_odd = 1.0 / model_prob
            edge = (fair_odd - market_odd) / market_odd

            if edge > self.min_edge:
                kelly = ((model_prob * (market_odd - 1)) - (1 - model_prob)) / (market_odd - 1)
                kelly = max(0.0, min(kelly * self.kelly_fraction, 0.25))
                ev = model_prob * market_odd - 1.0

                if ev > self.min_ev:
                    bet = ValueBet(
                        market=f"shots_{line_key}",
                        selection=f"{direction.title()} {line_value}",
                        fixture_id=model_pred.get("fixture_id", ""),
                        home_team=model_pred.get("home_team", ""),
                        away_team=model_pred.get("away_team", ""),
                        league=model_pred.get("league", ""),
                        match_date=model_pred.get("match_date", ""),
                        model_prob=round(model_prob, 4),
                        market_odds=round(market_odd, 2),
                        fair_odds=round(fair_odd, 2),
                        edge=round(edge, 4),
                        expected_value=round(ev, 4),
                        stake_pct=round(kelly * 100, 2),
                        model_prediction=model_pred.get("predicted_total_shots", 0),
                    )
                    found.append(bet)

        return found

    def check_match_result(self, model_pred: dict,
                           odds_home: float, odds_draw: float, odds_away: float) -> list[ValueBet]:
        """Avalia value bets em 1X2."""
        found: list[ValueBet] = []
        results = [
            ("Home", model_pred.get("prob_home", 0), odds_home),
            ("Draw", model_pred.get("prob_draw", 0), odds_draw),
            ("Away", model_pred.get("prob_away", 0), odds_away),
        ]

        for selection, prob, odd in results:
            if prob <= 0 or odd <= 1:
                continue

            prob = min(prob, 0.99)
            fair_odd = 1.0 / prob
            edge = (fair_odd - odd) / odd
            ev = prob * odd - 1.0

            if edge > self.min_edge and ev > self.min_ev:
                kelly = ((prob * (odd - 1)) - (1 - prob)) / (odd - 1)
                kelly = max(0.0, min(kelly * self.kelly_fraction, 0.25))

                bet = ValueBet(
                    market="match_result",
                    selection=selection,
                    fixture_id=model_pred.get("fixture_id", ""),
                    home_team=model_pred.get("home_team", ""),
                    away_team=model_pred.get("away_team", ""),
                    league=model_pred.get("league", ""),
                    match_date=model_pred.get("match_date", ""),
                    model_prob=round(prob, 4),
                    market_odds=round(odd, 2),
                    fair_odds=round(fair_odd, 2),
                    edge=round(edge, 4),
                    expected_value=round(ev, 4),
                    stake_pct=round(kelly * 100, 2),
                )
                found.append(bet)

        return found

    def summary(self, bets: list[ValueBet] | None = None) -> str:
        """Gera resumo textual das value bets encontradas."""
        if bets is None:
            bets = self.bets

        if not bets:
            return "Nenhuma value bet encontrada."

        lines = [f"Value Bets Encontradas: {len(bets)}"]

        # Por mercado
        for bet in sorted(bets, key=lambda b: -b.edge):
            lines.append(
                f"  {bet.home_team} x {bet.away_team} | {bet.market} | "
                f"{bet.selection} | Prob: {bet.model_prob:.1%} | "
                f"Odd: {bet.market_odds:.2f} | EV: {bet.expected_value:.1%} | "
                f"Stake: {bet.stake_pct:.1f}%"
            )

        return "\n".join(lines)
