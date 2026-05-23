"""Mercados derivados do modelo Dixon-Coles: BTTS, Over/Under Gols, Double Chance.

Todas as funções recebem o dicionário `score_probabilities` retornado por
`DixonColes.predict_score()` e computam probabilidades dos mercados derivados.
"""
from typing import Optional


def compute_btts(score_probs: dict, max_goals: int = 6) -> dict:
    """Compute BTTS (Both Teams To Score) probabilities.

    P(ambos marcam) = 1 - P(casa=0) - P(fora=0) + P(casa=0 AND fora=0)
    """
    if not score_probs:
        return {"probabilities": {"sim_0": 0.0, "nao_0": 0.0}, "btts_prob": 0.0}
    prob_home_0 = 0.0
    prob_away_0 = 0.0
    prob_both_0 = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = score_probs.get(f"{i}x{j}", 0)
            if i == 0:
                prob_home_0 += p
            if j == 0:
                prob_away_0 += p
            if i == 0 and j == 0:
                prob_both_0 += p

    prob_both = 1.0 - prob_home_0 - prob_away_0 + prob_both_0
    prob_both = max(0.0, min(1.0, prob_both))

    return {
        "probabilities": {
            "sim_0": round(prob_both, 4),
            "nao_0": round(1.0 - prob_both, 4),
        },
        "btts_prob": round(prob_both, 4),
    }


def compute_over_under(score_probs: dict,
                       lines: Optional[list[float]] = None,
                       max_goals: int = 6) -> dict:
    """Compute Over/Under goals probabilities for multiple lines.

    Para cada linha X.5:
      - Over X.5: P(total > X.5) = P(total >= X+1)
      - Under X.5: P(total < X.5) = P(total <= X)
    """
    if not score_probs:
        probs = {f"{d}_{l}": 0.0 for l in (lines or [0.5, 1.5, 2.5, 3.5, 4.5]) for d in ("over", "under")}
        return {"probabilities": probs, "predicted_total_goals": 0.0}
    if lines is None:
        lines = [0.5, 1.5, 2.5, 3.5, 4.5]

    # Build probability distribution for total goals
    probs_by_total = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            total = i + j
            p = score_probs.get(f"{i}x{j}", 0)
            probs_by_total[total] = probs_by_total.get(total, 0) + p

    expected_total = sum(total * prob for total, prob in probs_by_total.items())

    probabilities = {}
    for line in lines:
        over = sum(p for total, p in probs_by_total.items() if total > line)
        under = 1.0 - over
        probabilities[f"over_{line}"] = round(over, 4)
        probabilities[f"under_{line}"] = round(under, 4)

    return {
        "probabilities": probabilities,
        "predicted_total_goals": round(expected_total, 2),
    }


def compute_double_chance(score_probs: dict, max_goals: int = 6) -> dict:
    """Compute Double Chance probabilities.

    - Casa ou Empate: P(home) + P(draw)
    - Fora ou Empate:  P(away) + P(draw)
    """
    if not score_probs:
        return {"probabilities": {"casa-empate_0": 0.0, "fora-empate_0": 0.0},
                "predicted_double_chance": "casa-empate"}
    prob_home = 0.0
    prob_draw = 0.0
    prob_away = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = score_probs.get(f"{i}x{j}", 0)
            if i > j:
                prob_home += p
            elif i == j:
                prob_draw += p
            else:
                prob_away += p

    casa_empate = prob_home + prob_draw
    fora_empate = prob_away + prob_draw

    return {
        "probabilities": {
            "casa-empate_0": round(casa_empate, 4),
            "fora-empate_0": round(fora_empate, 4),
        },
        "predicted_double_chance": "casa-empate" if casa_empate > fora_empate else "fora-empate",
    }
