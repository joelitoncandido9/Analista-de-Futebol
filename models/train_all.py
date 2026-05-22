#!/usr/bin/env python3
"""Pipeline de treinamento: treina todos os modelos e gera relatorio.

Uso:
    python models/train_all.py                              # Treina todas ligas
    python models/train_all.py --league premier_league       # Liga especifica
    python models/train_all.py --backtest-only               # So backtesting
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import MODELS_DIR, LOGS_DIR
from models.corners.corners_trainer import CornersTrainer
from models.shots.shots_trainer import ShotsTrainer
from models.results.dixon_coles import DixonColes
from models.backtesting import Backtester
from config.leagues import LEAGUES


def train_league(league_name: str, run_backtest: bool = True) -> dict:
    """Treina todos os modelos para uma liga.

    Returns:
        Dict com resultados.
    """
    results = {"league": league_name, "corners": {}, "shots": {}, "dixon_coles": {}}

    # 1. Escanteios
    logger.info(f"{'='*50}")
    logger.info(f"[{league_name}] Treinando modelo de escanteios...")
    logger.info(f"{'='*50}")
    try:
        ct = CornersTrainer(league=league_name)
        results["corners"] = ct.train(test_games=380)
    except Exception as e:
        logger.error(f"[{league_name}] Erro corners: {e}")
        results["corners"] = {"error": str(e)}

    # 2. Finalizacoes
    logger.info(f"{'='*50}")
    logger.info(f"[{league_name}] Treinando modelo de finalizacoes...")
    logger.info(f"{'='*50}")
    try:
        st = ShotsTrainer(league=league_name)
        results["shots"] = st.train(test_games=380)
    except Exception as e:
        logger.error(f"[{league_name}] Erro shots: {e}")
        results["shots"] = {"error": str(e)}

    # 3. Dixon-Coles
    logger.info(f"{'='*50}")
    logger.info(f"[{league_name}] Treinando Dixon-Coles...")
    logger.info(f"{'='*50}")
    try:
        dc = DixonColes(league=league_name)
        results["dixon_coles"] = dc.train()
    except Exception as e:
        logger.error(f"[{league_name}] Erro Dixon-Coles: {e}")
        results["dixon_coles"] = {"error": str(e)}

    # 4. Backtesting (opcional)
    if run_backtest:
        for model_type in ["corners", "shots"]:
            logger.info(f"[{league_name}] Backtesting {model_type}...")
            try:
                bt = Backtester(model_type=model_type, league=league_name)
                bt_metrics = bt.run(test_games=380, window=500)
                if bt_metrics:
                    results[f"backtest_{model_type}"] = bt_metrics
                    bt.save_results()
            except Exception as e:
                logger.error(f"[{league_name}] Erro backtest {model_type}: {e}")

    return results


def train_all_leagues(leagues: list[str] | None = None,
                      run_backtest: bool = True) -> dict:
    """Treina modelos para todas as ligas."""
    if leagues is None:
        leagues = [l.name for l in LEAGUES]

    all_results = {}
    for league_name in leagues:
        all_results[league_name] = train_league(league_name, run_backtest)

    return all_results


def generate_report(all_results: dict) -> str:
    """Gera relatorio textual dos resultados."""
    lines = []
    lines.append("=" * 70)
    lines.append("RELATORIO DE TREINAMENTO - FOOTBALL AI")
    lines.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    for league, results in all_results.items():
        lines.append(f"\n{'─'*50}")
        lines.append(f"  {league}")
        lines.append(f"{'─'*50}")

        # Escanteios
        c = results.get("corners", {})
        if c.get("mae_test"):
            lines.append(f"  Escanteios  | MAE: {c['mae_test']:.2f} | "
                          f"RMSE: {c['rmse_test']:.2f} | "
                          f"Teste: {c['n_test']} jogos")
        else:
            lines.append(f"  Escanteios  | {c.get('error', 'N/A')}")

        # Finalizacoes
        s = results.get("shots", {})
        if s.get("mae_test"):
            lines.append(f"  Finalizacoes | MAE: {s['mae_test']:.2f} | "
                          f"RMSE: {s['rmse_test']:.2f} | "
                          f"Teste: {s['n_test']} jogos")
        else:
            lines.append(f"  Finalizacoes | {s.get('error', 'N/A')}")

        # Dixon-Coles
        dc = results.get("dixon_coles", {})
        if dc.get("n_teams"):
            lines.append(f"  Dixon-Coles | {dc['n_teams']} times, "
                          f"{dc['n_matches']} partidas, "
                          f"Home Adv: {dc.get('home_adv', '?'):.2f}")
        else:
            lines.append(f"  Dixon-Coles | {dc.get('error', 'N/A')}")

        # Backtest corners
        btc = results.get("backtest_corners", {})
        if btc.get("mae"):
            lines.append(f"  Backtest C  | MAE: {btc['mae']:.2f} | "
                          f"Bias: {btc['bias']:.2f} | "
                          f"N: {btc['n_tests']}")
            for line_key, line_data in list(btc.get("line_accuracy", {}).items())[:3]:
                lines.append(f"    {line_key}: pred={line_data['predicted_rate']:.1%} "
                              f"real={line_data['actual_rate']:.1%}")

        # Backtest shots
        bts = results.get("backtest_shots", {})
        if bts.get("mae"):
            lines.append(f"  Backtest S  | MAE: {bts['mae']:.2f} | "
                          f"Bias: {bts['bias']:.2f} | "
                          f"N: {bts['n_tests']}")

    lines.append(f"\n{'='*70}")
    lines.append("FIM DO RELATORIO")
    lines.append(f"{'='*70}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Treina todos os modelos")
    parser.add_argument("--league", help="Liga especifica (ex: premier_league)")
    parser.add_argument("--backtest-only", action="store_true",
                        help="So executa backtesting")
    parser.add_argument("--no-backtest", action="store_true",
                        help="Pula backtesting")
    args = parser.parse_args()

    # Config logging
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{message}</cyan>", level="INFO")
    logger.add(LOGS_DIR / "training_{time:YYYY-MM-DD}.log", rotation="1 day", level="INFO")

    start = time.time()

    if args.league:
        # Converter nome amigavel para nome da liga
        league_name = args.league.replace("_", " ").title()
        # Mapear
        name_map = {
            "premier_league": "Premier League",
            "premier league": "Premier League",
            "la_liga": "La Liga",
            "la liga": "La Liga",
            "bundesliga": "Bundesliga",
            "serie_a": "Serie A",
            "serie a": "Serie A",
            "ligue_1": "Ligue 1",
            "ligue 1": "Ligue 1",
            "brasileirao": "Brasileirao",
        }
        league_name = name_map.get(args.league.lower(), league_name)

        if args.backtest_only:
            for mt in ["corners", "shots"]:
                bt = Backtester(model_type=mt, league=league_name)
                bt_metrics = bt.run()
                if bt_metrics:
                    bt.save_results()
            return

        results = {league_name: train_league(league_name, run_backtest=not args.no_backtest)}
    else:
        if args.backtest_only:
            for league in LEAGUES:
                for mt in ["corners", "shots"]:
                    bt = Backtester(model_type=mt, league=league.name)
                    btm = bt.run()
                    if btm:
                        bt.save_results()
            return

        results = train_all_leagues(run_backtest=not args.no_backtest)

    elapsed = time.time() - start

    # Salvar resultados
    results_path = MODELS_DIR / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Converter para serializavel
    serializable = _make_serializable(results)
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Gerar relatorio
    report = generate_report(results)
    report_path = MODELS_DIR / "training_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\nTempo total: {elapsed:.1f}s")
    print(f"Relatorio: {report_path}")
    print(f"Resultados: {results_path}")


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _make_serializable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
