#!/usr/bin/env python3
"""Treina TODOS os modelos pendentes: novos mercados + novas ligas.

Uso:
    cd /home/palpites/football_ai && nohup python3 -m models.train_all_markets > logs/full_training.log 2>&1 &
"""
import os
import sys
import time
from datetime import datetime

# Ensure we're in the project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)
sys.path.insert(0, _project_root)

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add("/home/palpites/football_ai/logs/full_training_{time:YYYY-MM-DD}.log", rotation="1 day", level="INFO")

# ============================================================
# LEAGUES
# ============================================================
EURO_5 = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
NEW_4 = ["Championship", "Primeira Liga", "Eredivisie", "2. Bundesliga"]
ALL_10 = EURO_5 + NEW_4 + ["Brasileirao"]

# ============================================================
# GROUP A: ShotsOnTarget for Bundesliga, Serie A, Ligue 1
# ============================================================
def train_shots_on_target(leagues: list[str]):
    from models.shots_on_target.shots_on_target_trainer import ShotsOnTargetTrainer

    for league in leagues:
        logger.info(f"{'='*60}")
        logger.info(f"[ShotsOnTarget] Treinando {league}...")
        logger.info(f"{'='*60}")
        try:
            t = ShotsOnTargetTrainer(league=league)
            t.train()
            logger.info(f"[ShotsOnTarget] {league} CONCLUIDO")
        except Exception as e:
            logger.error(f"[ShotsOnTarget] {league} ERRO: {e}")

# ============================================================
# GROUP B: Fouls for all leagues
# ============================================================
def train_fouls(leagues: list[str]):
    from models.fouls.fouls_trainer import FoulsTrainer

    for league in leagues:
        logger.info(f"{'='*60}")
        logger.info(f"[Fouls] Treinando {league}...")
        logger.info(f"{'='*60}")
        try:
            t = FoulsTrainer(league=league)
            t.train()
            logger.info(f"[Fouls] {league} CONCLUIDO")
        except Exception as e:
            logger.error(f"[Fouls] {league} ERRO: {e}")

# ============================================================
# GROUP C: Cards for all leagues
# ============================================================
def train_cards(leagues: list[str]):
    from models.cards.cards_trainer import CardsTrainer

    for league in leagues:
        logger.info(f"{'='*60}")
        logger.info(f"[Cards] Treinando {league}...")
        logger.info(f"{'='*60}")
        try:
            t = CardsTrainer(league=league)
            t.train()
            logger.info(f"[Cards] {league} CONCLUIDO")
        except Exception as e:
            logger.error(f"[Cards] {league} ERRO: {e}")

# ============================================================
# GROUP D: TeamCorners (home + away) for all leagues
# ============================================================
def train_team_corners(leagues: list[str]):
    from models.team_corners.team_corners_trainer import (
        TeamCornersHomeTrainer,
        TeamCornersAwayTrainer,
    )

    for league in leagues:
        logger.info(f"{'='*60}")
        logger.info(f"[TeamCorners] Treinando {league}...")
        logger.info(f"{'='*60}")
        try:
            logger.info(f"[TeamCorners] {league} - Home...")
            ht = TeamCornersHomeTrainer(league=league)
            ht.train()
            logger.info(f"[TeamCorners] {league} - Away...")
            at = TeamCornersAwayTrainer(league=league)
            at.train()
            logger.info(f"[TeamCorners] {league} CONCLUIDO")
        except Exception as e:
            logger.error(f"[TeamCorners] {league} ERRO: {e}")

# ============================================================
# GROUP E: Corners + Shots for 4 new leagues
# ============================================================
def train_corners_shots(leagues: list[str]):
    from models.corners.corners_trainer import CornersTrainer
    from models.shots.shots_trainer import ShotsTrainer

    for league in leagues:
        logger.info(f"{'='*60}")
        logger.info(f"[Corners/Shots] Treinando {league}...")
        logger.info(f"{'='*60}")
        try:
            ct = CornersTrainer(league=league)
            ct.train()
            logger.info(f"[Corners] {league} CONCLUIDO")
        except Exception as e:
            logger.error(f"[Corners] {league} ERRO: {e}")

        try:
            st = ShotsTrainer(league=league)
            st.train()
            logger.info(f"[Shots] {league} CONCLUIDO")
        except Exception as e:
            logger.error(f"[Shots] {league} ERRO: {e}")

# ============================================================
# GROUP F: Dixon-Coles for 4 new leagues
# ============================================================
def train_dixon_coles(leagues: list[str]):
    from models.results.dixon_coles import DixonColes

    for league in leagues:
        logger.info(f"{'='*60}")
        logger.info(f"[Dixon-Coles] Treinando {league}...")
        logger.info(f"{'='*60}")
        try:
            dc = DixonColes(league=league)
            dc.train()
            logger.info(f"[Dixon-Coles] {league} CONCLUIDO")
        except Exception as e:
            logger.error(f"[Dixon-Coles] {league} ERRO: {e}")


# ============================================================
# MAIN
# ============================================================
def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("TREINAMENTO COMPLETO INICIADO")
    logger.info(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"10 ligas: {ALL_10}")
    logger.info("=" * 60)

    # --- FASE 1: ShotsOnTarget que faltam nas 5 europeias ---
    logger.info("\n\n>>> FASE 1: ShotsOnTarget para Bundesliga, Serie A, Ligue 1\n")
    train_shots_on_target(["Bundesliga", "Serie A", "Ligue 1"])

    # --- FASE 2: Fouls para todas as 10 ligas ---
    logger.info("\n\n>>> FASE 2: Fouls para todas as 10 ligas\n")
    train_fouls(ALL_10)

    # --- FASE 3: Cards para todas as 10 ligas ---
    logger.info("\n\n>>> FASE 3: Cards para todas as 10 ligas\n")
    train_cards(ALL_10)

    # --- FASE 4: TeamCorners para todas as 10 ligas ---
    logger.info("\n\n>>> FASE 4: TeamCorners para todas as 10 ligas\n")
    train_team_corners(ALL_10)

    # --- FASE 5: Corners + Shots para as 4 novas ligas ---
    logger.info("\n\n>>> FASE 5: Corners + Shots para Championship, Primeira Liga, Eredivisie, 2.Bundesliga\n")
    train_corners_shots(NEW_4)

    # --- FASE 6: Dixon-Coles para as 4 novas ligas ---
    logger.info("\n\n>>> FASE 6: Dixon-Coles para as 4 novas ligas\n")
    train_dixon_coles(NEW_4)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("TREINAMENTO COMPLETO FINALIZADO")
    logger.info(f"Tempo total: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
