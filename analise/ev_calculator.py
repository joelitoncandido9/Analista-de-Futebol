"""Calculo de Expected Value (EV) para mercados alternativos.

Usa medias historicas por liga + distribuicao Poisson para estimar
a probabilidade real de eventos como escanteios, cartoes e finalizacoes,
comparando com as odds do mercado para encontrar valor.
"""
import math
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ValueTip:
    """Palpite com valor esperado positivo."""
    event_id: int
    home_team: str
    away_team: str
    league: str
    market: str          # corners, yellow_cards, total_shots, etc.
    line: float
    direction: str       # over ou under
    odd: float           # odd do mercado
    implied_prob: float  # 1 / odd
    est_prob: float      # nossa estimativa de probabilidade
    ev: float            # est_prob * odd - 1
    confidence: float    # confianca (baseada no n amostral)


def poisson_cdf(k: float, lam: float) -> float:
    """P(X <= k) para Poisson com media lam."""
    # Para k grande (> 20), aproximacao normal
    if k > 20:
        return 0.5 * (1 + math.erf((k + 0.5 - lam) / math.sqrt(2 * lam)))
    # Soma exata para k pequeno
    acc = 0.0
    for i in range(int(k) + 1):
        acc += (lam ** i) * math.exp(-lam) / math.factorial(i)
    return min(acc, 1.0)


def prob_over_line(line: float, avg: float) -> float:
    """P(X > line) para Poisson(avg)."""
    return 1 - poisson_cdf(line, avg)


def prob_under_line(line: float, avg: float) -> float:
    """P(X < line) para Poisson(avg)."""
    return poisson_cdf(line - 0.001, avg)


# Mapeamento: nosso market name -> coluna de media historica
MARKET_TO_STAT = {
    "corners": "avg_corners",
    "home_corners": "avg_corners",
    "away_corners": "avg_corners",
    "yellow_cards": "avg_yellow",
    "total_shots": "avg_shots",
    "total_shots_on_target": "avg_sot",
}


def estimate_probability(
    market: str, line: float, direction: str, league_stats: dict, league: str
) -> tuple[float, float]:
    """Estima probabilidade real para um mercado/linha/direcao.

    Returns:
        (probabilidade_estimada, confianca)
    """
    stat_key = MARKET_TO_STAT.get(market)
    if not stat_key:
        return 0.5, 0.0

    stats = league_stats.get(league, {})
    avg = stats.get(stat_key)
    n = stats.get("n", 0)

    if avg is None or avg == 0:
        # Fallback: media geral de todas as ligas
        all_avgs = [s.get(stat_key, 0) for s in league_stats.values() if s.get(stat_key)]
        avg = sum(all_avgs) / len(all_avgs) if all_avgs else _default_avg(market)
        n = 0

    # Confianca baseada no numero de amostras (n >= 6 aceitavel)
    confidence = min(1.0, n / 20) if n > 0 else 0.1

    # Para mercados de time (home_corners, away_corners), usar metade da media
    if market in ("home_corners", "away_corners"):
        avg = avg / 2

    if direction == "over":
        prob = prob_over_line(line, avg)
    else:
        prob = prob_under_line(line, avg)

    # Suavizar: nunca dar 0% ou 100%
    prob = max(0.01, min(0.99, prob))
    return prob, confidence


def _default_avg(market: str) -> float:
    """Medias padrao quando nao ha dados historicos."""
    defaults = {
        "corners": 9.5,
        "yellow_cards": 4.0,
        "total_shots": 25.0,
        "total_shots_on_target": 9.0,
    }
    return defaults.get(market, 10.0)


def calculate_ev_tips(
    odds_rows: list[dict], league_stats: dict, min_ev: float = 0.05
) -> list[ValueTip]:
    """Calcula EV para cada odd e retorna palpites com valor positivo.

    Args:
        odds_rows: lista de dicts do OddsApiIOCollector.extract_value_odds()
        league_stats: dict de get_league_stats()
        min_ev: EV minimo para considerarmos o palpite (default 5%)

    Returns:
        Lista de ValueTip com EV > min_ev
    """
    tips = []
    for row in odds_rows:
        league = row.get("league", "")
        est_prob, confidence = estimate_probability(
            market=row["market"],
            line=row["line"],
            direction=row["direction"],
            league_stats=league_stats,
            league=league,
        )

        odd = row["odd"]
        ev = est_prob * odd - 1
        implied_prob = 1 / odd if odd > 0 else 0

        # So considerar com amostragem minima (n >= 6)
        if confidence < 0.2:
            continue

        # Ajustar EV pela confianca
        adjusted_ev = ev * min(1.0, confidence * 2)

        if adjusted_ev > min_ev:
            tips.append(ValueTip(
                event_id=row["event_id"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                league=league,
                market=row["market"],
                line=row["line"],
                direction=row["direction"],
                odd=odd,
                implied_prob=round(implied_prob, 3),
                est_prob=round(est_prob, 3),
                ev=round(ev, 4),
                confidence=round(confidence, 2),
            ))

    # Ordenar por EV (melhor primeiro)
    tips.sort(key=lambda t: t.ev, reverse=True)
    return tips
