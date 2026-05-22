"""Alertas Telegram + WhatsApp para value bets, relatorios e notificacoes.

Envia mensagens para ambos os canais configurados.
Telegram: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
WhatsApp: Evolution API (EVO_INSTANCE)
"""
from datetime import datetime, date

import requests
from loguru import logger

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_telegram(message: str) -> bool:
    """Envia mensagem texto para o Telegram.

    Args:
        message: Texto da mensagem (suporta Markdown basico).

    Returns:
        True se enviou com sucesso.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Alert] Telegram nao configurado (TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID)")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            logger.debug("[Alert] Mensagem enviada ao Telegram")
            return True
        else:
            logger.warning(f"[Alert] Erro Telegram: HTTP {resp.status_code} - {resp.text[:100]}")
            return False
    except Exception as e:
        logger.warning(f"[Alert] Erro ao enviar Telegram: {e}")
        return False


def _send_both(message: str) -> bool:
    """Envia mensagem para Telegram e WhatsApp."""
    tg_ok = send_telegram(message)
    wa_ok = send_whatsapp(message)
    return tg_ok or wa_ok


def send_whatsapp(message: str) -> bool:
    """Envia mensagem via WhatsApp."""
    try:
        from scheduler.whatsapp import send_text
        return send_text(message)
    except Exception as e:
        logger.warning(f"[Alert] Erro WhatsApp: {e}")
        return False


def alert_pre_match(home: str, away: str, league: str,
                     corners_pred: dict | None = None,
                     result_pred: dict | None = None,
                     value_bets: list | None = None) -> bool:
    """Envia relatorio pre-jogo."""
    lines = [
        f"*PRE-MATCH: {home} x {away}*",
        f"Liga: {league}",
        f"Data: {date.today().strftime('%d/%m/%Y')}",
        "",
    ]

    if corners_pred:
        lines.extend([
            f"*Escanteios*: {corners_pred.get('predicted_total_corners', '?')} totais",
        ])
        for line, prob in list(corners_pred.get("probabilities", {}).items())[:4]:
            lines.append(f"  {line}: {prob:.1%}")

    if result_pred:
        lines.extend([
            "",
            f"*Resultado*: {result_pred.get('most_likely_score', '?')}",
            f"  Casa: {result_pred.get('prob_home', 0):.1%} | "
            f"Emp: {result_pred.get('prob_draw', 0):.1%} | "
            f"Fora: {result_pred.get('prob_away', 0):.1%}",
        ])

    if value_bets:
        lines.extend(["", "*Value Bets*"])
        for vb in value_bets[:5]:
            lines.append(
                f"  {vb.selection} | EV: {vb.expected_value:.1%} | "
                f"Odd: {vb.market_odds:.2f} | Stake: {vb.stake_pct:.1f}%"
            )

    return _send_both("\n".join(lines))


def alert_result(home: str, away: str, league: str,
                  home_goals: int, away_goals: int,
                  home_corners: int | None = None,
                  away_corners: int | None = None,
                  pred_corners: float | None = None) -> bool:
    """Envia resultado de partida para o Telegram."""
    lines = [
        f"*RESULTADO: {home} x {away}*",
        f"Placar: {home_goals} - {away_goals}",
        f"Liga: {league}",
    ]

    if home_corners is not None:
        lines.append(f"Escanteios: {home_corners} - {away_corners}")
        if pred_corners:
            lines.append(f"Previsto: {pred_corners:.1f}")

    return _send_both("\n".join(lines))


def alert_value_bets(bets: list) -> bool:
    """Envia lista de value bets encontradas."""
    if not bets:
        return False

    lines = [
        f"*VALUE BETS ENCONTRADAS*",
        f"Total: {len(bets)} oportunidades",
        f"Data: {date.today().strftime('%d/%m/%Y')}",
        "",
    ]

    for vb in sorted(bets, key=lambda b: -b.edge)[:10]:
        lines.append(
            f"*{vb.home_team} x {vb.away_team}*"
        )
        lines.append(
            f"  {vb.selection} | "
            f"Prob: {vb.model_prob:.1%} | "
            f"Odd: {vb.market_odds:.2f} | "
            f"EV: {vb.expected_value:.1%} | "
            f"Stake: {vb.stake_pct:.1f}%"
        )
        lines.append("")

    return _send_both("\n".join(lines))


def alert_daily_summary(stats: dict) -> bool:
    """Envia resumo diario do sistema."""
    lines = [
        f"*RESUMO DIARIO - FOOTBALL AI*",
        f"Data: {date.today().strftime('%d/%m/%Y')}",
        "",
        "*Partidas no banco:*",
    ]

    for league, count in stats.get("matches_per_league", {}).items():
        lines.append(f"  {league}: {count}")

    lines.extend([
        "",
        f"Ultima atualizacao: {datetime.now().strftime('%H:%M')}",
    ])

    return _send_both("\n".join(lines))
