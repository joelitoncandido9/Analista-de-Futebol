#!/usr/bin/env python3
"""Ponto de entrada para producao — inicia o scheduler.

Uso:
    python start.py                    # Inicia scheduler em foreground
    nohup python start.py &            # Daemon (background)
    python start.py --once --collect   # Executa coleta unica e sai
    python start.py --once --predict   # Gera predictions unicas
"""
import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from config.settings import LOGS_DIR, LOG_LEVEL


def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{message}</cyan>",
        level=LOG_LEVEL,
        colorize=True,
    )
    logger.add(
        LOGS_DIR / "scheduler_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
    )


def cmd_once_collect():
    """Executa coleta unica."""
    logger.info("[Start] Coleta unica...")

    # 1. Coleta do dia via API-Football
    from collectors.api_football_collector import APIFootballCollector
    api = APIFootballCollector()
    fixtures = api.collect_today()
    logger.info(f"  Fixtures hoje: {len(fixtures)}")

    # 2. Merge
    from database.merge import merge_all
    merge_all()

    logger.info("[Start] Coleta unica concluida")


def cmd_once_predict():
    """Gera predictions unicas para os jogos do dia."""
    logger.info("[Start] Predictions unicas...")

    from collectors.api_football_collector import APIFootballCollector
    from models.corners.corners_predictor import CornersPredictor
    from models.results.dixon_coles import DixonColes
    from scheduler.alerts import alert_pre_match

    api = APIFootballCollector()

    today = datetime.now().strftime("%Y-%m-%d")
    fixtures = api.get_fixtures(today)

    # Filtrar so jogos das ligas que cobrimos
    from config.leagues import LEAGUES
    league_names = {l.name for l in LEAGUES}
    fixtures = [f for f in fixtures if f.get("league") in league_names]

    logger.info(f"  {len(fixtures)} jogos das ligas monitoradas")

    for fx in fixtures:
        home = fx.get("home_team", "")
        away = fx.get("away_team", "")
        league = fx.get("league", "")

        cp = CornersPredictor(league=league)
        corners = cp.predict(home, away, league, today)
        dc = DixonColes(league=league)
        result = dc.predict_score(home, away) if dc.load() else None

        alert_pre_match(home, away, league, corners, result)
        time.sleep(0.5)

    logger.info("[Start] Predictions concluidas")


def cmd_daemon():
    """Inicia scheduler em modo daemon (foreground)."""
    from scheduler.jobs import FootballScheduler

    logger.info("[Start] Iniciando Football AI Scheduler...")

    scheduler = FootballScheduler()

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info(f"[Start] Sinal {signum} recebido. Parando...")
        scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    scheduler.start()

    # Manter vivo
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.stop()
        logger.info("[Start] Parado pelo usuario.")


def main():
    parser = argparse.ArgumentParser(description="Football AI - Inicializacao")
    parser.add_argument("--once", action="store_true",
                        help="Executa uma unica vez e sai")
    parser.add_argument("--collect", action="store_true",
                        help="(com --once) Executa coleta")
    parser.add_argument("--predict", action="store_true",
                        help="(com --once) Executa predictions")
    args = parser.parse_args()

    setup_logging()

    if args.once and args.collect:
        cmd_once_collect()
    elif args.once and args.predict:
        cmd_once_predict()
    else:
        cmd_daemon()


if __name__ == "__main__":
    main()
