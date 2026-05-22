"""Indexador de partidas e jogadores no ChromaDB.

Converte dados do banco SQLite em documentos textuais,
gera embeddings e armazena no vectorstore para busca semantica.
"""
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

from config.settings import DB_PATH
from models.features import load_matches, league_averages
from rag.vectorstore import get_client, get_or_create_collection, _make_metadata
from database.schema import get_conn


def _match_to_doc(row: dict) -> tuple[str, str, dict]:
    """Converte uma partida em documento textual.

    Returns:
        (doc_id, text, metadata)
    """
    home = row.get("home_team", "?")
    away = row.get("away_team", "?")
    league = row.get("league", "?")
    date = str(row.get("match_date", "")[:10])
    hg = row.get("home_goals", "?")
    ag = row.get("away_goals", "?")
    hc = row.get("home_corners", "?")
    ac = row.get("away_corners", "?")
    hs = row.get("home_shots", "?")
    asht = row.get("away_shots", "?")
    hx = row.get("home_xg", "?")
    ax = row.get("away_xg", "?")

    if hg != "?" and ag != "?":
        result = f"{home} {hg}-{ag} {away}"
        winner = "home" if hg > ag else ("away" if ag > hg else "draw")
    else:
        result = f"{home} vs {away}"
        winner = "unknown"

    text = (
        f"Partida: {home} x {away} em {date} - {league}. "
        f"Resultado: {hg}-{ag}. "
        f"Escanteios: {home} {hc}, {away} {ac}. "
        f"Finalizacoes: {home} {hs}, {away} {asht}. "
        f"xG: {home} {hx}, {away} {ax}."
    )

    doc_id = f"match_{row.get('match_id', uuid.uuid4().hex)}"

    metadata = _make_metadata({
        "type": "match",
        "match_id": row.get("match_id"),
        "league": league,
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_goals": hg if hg != "?" else None,
        "away_goals": ag if ag != "?" else None,
        "winner": winner,
        "season": row.get("season"),
        "home_corners": hc if hc != "?" else None,
        "away_corners": ac if ac != "?" else None,
        "home_xg": hx if hx != "?" else None,
        "away_xg": ax if ax != "?" else None,
    })

    return doc_id, text, metadata


def _team_to_doc(team: str, league: str, stats: dict) -> tuple[str, str, dict]:
    """Converte perfil de time em documento."""
    text = (
        f"Time: {team} ({league}). "
        f"Media escanteios: {stats.get('avg_total_corners', '?'):.1f} por jogo "
        f"({stats.get('avg_home_corners', '?'):.1f} casa, {stats.get('avg_away_corners', '?'):.1f} fora). "
        f"Media finalizacoes: {stats.get('avg_total_shots', '?'):.1f} por jogo. "
        f"Media gols: {stats.get('avg_home_goals', '?'):.2f} casa, {stats.get('avg_away_goals', '?'):.2f} fora."
    )

    doc_id = f"team_{league}_{team}".replace(" ", "_").lower()

    metadata = _make_metadata({
        "type": "team",
        "team": team,
        "league": league,
    })

    return doc_id, text, metadata


def index_matches(limit: int = 5000) -> int:
    """Indexa partidas no ChromaDB.

    Args:
        limit: Maximo de partidas a indexar (mais recentes).

    Returns:
        Numero de documentos indexados.
    """
    logger.info("[Indexer] Carregando partidas do banco...")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT * FROM matches
           WHERE home_corners IS NOT NULL
           ORDER BY match_date DESC LIMIT ?""",
        (limit,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    if not rows:
        logger.warning("[Indexer] Nenhuma partida para indexar")
        return 0

    client = get_client()
    collection = get_or_create_collection(client, "matches")

    ids = []
    documents = []
    metadatas = []

    for row in rows:
        doc_id, text, metadata = _match_to_doc(row)
        ids.append(doc_id)
        documents.append(text)
        metadatas.append(metadata)

    # Upsert em lotes de 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info(f"[Indexer] {len(ids)} partidas indexadas em 'matches'")
    return len(ids)


def index_teams() -> int:
    """Indexa perfis de times no ChromaDB."""
    logger.info("[Indexer] Calculando medias por time...")

    df = load_matches()
    if df.empty:
        return 0

    avgs = league_averages(df)

    client = get_client()
    collection = get_or_create_collection(client, "teams")

    ids = []
    documents = []
    metadatas = []

    for league in df["league"].unique():
        league_df = df[df["league"] == league]
        for team in league_df["home_team"].unique():
            team_df = league_df[
                (league_df["home_team"] == team) | (league_df["away_team"] == team)
            ]
            stats = {
                "avg_total_corners": team_df["total_corners"].mean(),
                "avg_home_corners": team_df[team_df["home_team"] == team]["home_corners"].mean(),
                "avg_away_corners": team_df[team_df["away_team"] == team]["away_corners"].mean(),
                "avg_total_shots": team_df["total_shots"].mean(),
                "avg_home_goals": team_df[team_df["home_team"] == team]["home_goals"].mean(),
                "avg_away_goals": team_df[team_df["away_team"] == team]["away_goals"].mean(),
            }

            doc_id, text, metadata = _team_to_doc(team, league, stats)
            ids.append(doc_id)
            documents.append(text)
            metadatas.append(metadata)

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(ids=ids[i:end], documents=documents[i:end], metadatas=metadatas[i:end])

    logger.info(f"[Indexer] {len(ids)} times indexados em 'teams'")
    return len(ids)


def index_all(limit_matches: int = 5000) -> dict:
    """Indexa todos os dados no ChromaDB."""
    results = {
        "matches": index_matches(limit_matches),
        "teams": index_teams(),
    }
    logger.info(f"[Indexer] Total indexado: {sum(results.values())} documentos")
    return results
