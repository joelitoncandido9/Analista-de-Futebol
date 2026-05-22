#!/usr/bin/env python3
"""Football AI — Entry point principal.

Uso:
    python main.py setup              # Inicializa banco de dados
    python main.py collect --league all  # Coleta dados de todas as ligas
    python main.py collect --league premier_league --seasons 5
    python main.py status             # Status do banco
    python main.py today              # Coleta fixtures de hoje
"""
import argparse
import sys
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
        LOGS_DIR / "football_ai_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
    )


def cmd_setup():
    """Inicializa o banco de dados."""
    from database.schema import init_db
    init_db()
    print("\n✅ Banco pronto para uso")
    print("   Execute: python main.py collect --league all")


def cmd_collect(args):
    """Coleta dados historicos e/ou do dia."""
    # 1. Coleta dados historicos do football-data.co.uk
    if args.league == "all":
        print("\n📡 Coletando dados historicos (football-data.co.uk)...")
        from collectors.football_data_collector import FootballDataCollector
        collector = FootballDataCollector()
        results = collector.collect_all(max_seasons=args.seasons)
        for liga, count in results.items():
            print(f"  {liga}: {count} partidas")

    else:
        print(f"\n📡 Coletando {args.league}...")
        from collectors.football_data_collector import FootballDataCollector
        collector = FootballDataCollector()
        count = collector.collect_league(args.league, args.seasons)
        print(f"  {args.league}: {count} partidas")

    # 2. Coleta dados do Understat (xG, PPDA)
    print("\n📡 Coletando Understat (xG, PPDA)...")
    from collectors.understat_collector import UnderstatCollector
    understat = UnderstatCollector()
    if args.league == "all":
        results = understat.collect_all()
        for liga, count in results.items():
            print(f"  {liga}: {count} partidas")
    else:
        count = understat.collect_league(args.league)
        print(f"  {args.league}: {count} partidas")

    # 3. Se for coleta do dia, buscar fixtures de hoje
    if args.today:
        print("\n📡 Coletando fixtures de hoje (API-Football)...")
        from collectors.api_football_collector import APIFootballCollector
        api = APIFootballCollector()
        fixtures = api.collect_today()
        print(f"  {len(fixtures)} fixtures encontrados")

    # 4. Merge dados
    print("\n🔗 Merge de dados...")
    from database.merge import merge_all
    merge_all()


def cmd_merge():
    """Merge dados de diferentes fontes."""
    from database.merge import merge_all
    print("\n🔗 Merge de dados...")
    count = merge_all()
    print(f"  {count} registros atualizados")


def cmd_status():
    """Mostra status do banco de dados."""
    from database.schema import get_conn

    conn = get_conn()
    cur = conn.cursor()

    print("\n📊 STATUS DO BANCO")
    print("=" * 40)

    for table in ["matches", "teams", "players", "team_match_stats", "player_match_stats", "odds"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        print(f"  {table}: {count} registros")

    # Partidas por liga
    print("\n🏆 PARTIDAS POR LIGA")
    cur.execute("""
        SELECT league, COUNT(*) as qtd,
               MIN(match_date) as primeiro, MAX(match_date) as ultimo
        FROM matches GROUP BY league ORDER BY qtd DESC
    """)
    for row in cur.fetchall():
        print(f"  {row['league']}: {row['qtd']} ({row['primeiro']} a {row['ultimo']})")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Football AI - Sistema de Analise de Futebol")
    subparsers = parser.add_subparsers(dest="command", help="Comandos")

    # setup
    subparsers.add_parser("setup", help="Inicializa banco de dados")

    # collect
    collect_parser = subparsers.add_parser("collect", help="Coleta dados")
    collect_parser.add_argument("--league", default="all",
                                help="Liga (ex: premier_league) ou 'all'")
    collect_parser.add_argument("--seasons", type=int, default=None,
                                help="Numero de temporadas (default: todas)")
    collect_parser.add_argument("--today", action="store_true",
                                help="Coleta fixtures de hoje via API-Football")

    # status
    subparsers.add_parser("status", help="Status do banco")

    # merge
    subparsers.add_parser("merge", help="Merge dados de diferentes fontes")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging()

    cmds = {
        "setup": cmd_setup,
        "collect": lambda: cmd_collect(args),
        "status": cmd_status,
        "merge": cmd_merge,
    }

    fn = cmds.get(args.command)
    if fn:
        fn()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
