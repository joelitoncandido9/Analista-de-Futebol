"""API REST do Football AI — FastAPI.

Endpoints para consultar dados, previsoes e agentes.
Executar: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.leagues import LEAGUES, LEAGUES_BY_NAME
from config.settings import MODELS_DIR
from database.schema import get_conn


def _serialize(obj: Any) -> Any:
    """Converte tipos numpy para tipos Python nativos (para JSON)."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _serialize(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

app = FastAPI(title="Football AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check com status do banco."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM matches")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT league) FROM matches")
        leagues = cur.fetchone()[0]
        conn.close()
        return {"status": "ok", "total_matches": total, "leagues": leagues}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ─── Leagues ───────────────────────────────────────────────────────────

@app.get("/leagues")
def list_leagues():
    """Lista as ligas configuradas."""
    return [
        {
            "name": l.name,
            "country": l.country,
            "api_football_id": l.api_football_id,
            "has_understat": l.understat_name is not None,
        }
        for l in LEAGUES
    ]


# ─── Matches ───────────────────────────────────────────────────────────

@app.get("/matches")
def list_matches(
    league: Optional[str] = Query(None, description="Filtrar por liga"),
    season: Optional[str] = Query(None, description="Filtrar por temporada"),
    team: Optional[str] = Query(None, description="Filtrar por time"),
    date_from: Optional[str] = Query(None, description="Data inicial (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Data final (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=1000),
):
    """Lista partidas com filtros."""
    conn = get_conn()
    cur = conn.cursor()

    conditions = []
    params = []

    if league:
        conditions.append("league = ?")
        params.append(league)
    if season:
        conditions.append("season = ?")
        params.append(season)
    if team:
        conditions.append("(home_team = ? OR away_team = ?)")
        params.extend([team, team])
    if date_from:
        conditions.append("match_date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("match_date <= ?")
        params.append(date_to)

    where = " AND ".join(conditions) if conditions else "1=1"
    cur.execute(
        f"SELECT * FROM matches WHERE {where} ORDER BY match_date DESC LIMIT ?",
        [*params, limit],
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


@app.get("/matches/{match_id}")
def get_match(match_id: str):
    """Detalhes de uma partida."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Partida nao encontrada")
    return dict(row)


# ─── Teams ─────────────────────────────────────────────────────────────

@app.get("/teams")
def list_teams(query: Optional[str] = Query(None, description="Buscar time")):
    """Lista times cadastrados."""
    conn = get_conn()
    cur = conn.cursor()
    if query:
        cur.execute(
            "SELECT DISTINCT name FROM teams WHERE name LIKE ? ORDER BY name LIMIT 100",
            (f"%{query}%",),
        )
    else:
        cur.execute("SELECT DISTINCT name FROM teams ORDER BY name LIMIT 200")
    rows = [r["name"] for r in cur.fetchall()]
    conn.close()
    return rows


@app.get("/teams/{team_name}")
def get_team(team_name: str, league: Optional[str] = Query(None)):
    """Estatisticas de um time."""
    from database.queries import get_team_recent_matches, get_team_avg_stats

    recent = get_team_recent_matches(team_name, league, n=10)

    # Medias de todas as ligas que o time joga
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT league FROM matches WHERE home_team = ? OR away_team = ?",
        (team_name, team_name),
    )
    leagues_played = [r["league"] for r in cur.fetchall()]
    conn.close()

    avgs = {}
    for lg in leagues_played:
        avg = get_team_avg_stats(team_name, lg, "", n=10)
        if avg and any(v is not None for v in avg.values()):
            avgs[lg] = {k: (v if v is not None else 0) for k, v in avg.items()}

    return {
        "name": team_name,
        "leagues": leagues_played,
        "recent_matches": recent,
        "averages": avgs,
    }


# ─── H2H ───────────────────────────────────────────────────────────────

@app.get("/h2h")
def head_to_head(team_a: str = Query(...), team_b: str = Query(...)):
    """Confronto direto entre dois times."""
    from database.queries import get_h2h_stats

    stats = get_h2h_stats(team_a, team_b)
    if stats.get("total", 0) == 0:
        raise HTTPException(404, "Nenhum confronto encontrado")
    return stats


# ─── Predictions ───────────────────────────────────────────────────────

@app.get("/predict/corners")
def predict_corners(
    home: str = Query(...),
    away: str = Query(...),
    league: str = Query(...),
    match_date: str = Query(default_factory=lambda: date.today().strftime("%Y-%m-%d")),
):
    """Preve total de escanteios para uma partida."""
    from models.corners.corners_predictor import CornersPredictor

    try:
        cp = CornersPredictor(league=league)
        result = cp.predict(home, away, league, match_date)
        if result is None:
            raise HTTPException(400, "Nao foi possivel gerar previsao")
        return _serialize(result)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/predict/shots")
def predict_shots(
    home: str = Query(...),
    away: str = Query(...),
    league: str = Query(...),
    match_date: str = Query(default_factory=lambda: date.today().strftime("%Y-%m-%d")),
):
    """Preve total de finalizacoes para uma partida."""
    from models.shots.shots_predictor import ShotsPredictor

    try:
        sp = ShotsPredictor(league=league)
        result = sp.predict(home, away, league, match_date)
        if result is None:
            raise HTTPException(400, "Nao foi possivel gerar previsao")
        return _serialize(result)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/predict/result")
def predict_result(
    home: str = Query(...),
    away: str = Query(...),
    league: str = Query(...),
):
    """Preve resultado (Dixon-Coles) para uma partida."""
    from models.results.dixon_coles import DixonColes

    try:
        dc = DixonColes(league=league)
        if not dc.load():
            raise HTTPException(400, f"Modelo Dixon-Coles nao encontrado para {league}")
        result = dc.predict_score(home, away)
        return _serialize(result)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/predict/pre-match")
def pre_match_report(
    home: str = Query(...),
    away: str = Query(...),
    league: str = Query(...),
    match_date: Optional[str] = Query(None),
):
    """Relatorio pre-jogo completo via agente reporter."""
    from agents.reporter import ReporterAgent

    try:
        agent = ReporterAgent()
        report = agent.pre_match(home, away, league, match_date)
        return {"home_team": home, "away_team": away, "league": league, "report": report}
    except Exception as e:
        raise HTTPException(400, str(e))


# ─── Agents ────────────────────────────────────────────────────────────

@app.post("/agents/analyst")
def analyst_query(
    question: str = Query(...),
    league: Optional[str] = Query(None),
):
    """Pergunta ao analista tatico."""
    from agents.analyst import AnalystAgent

    try:
        agent = AnalystAgent()
        response = agent.answer(question, league)
        return {"question": question, "response": response}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/agents/scout")
def scout_player(
    player_name: str = Query(...),
    league: Optional[str] = Query(None),
):
    """Analisa um jogador (scouting)."""
    from agents.scout import ScoutAgent

    try:
        agent = ScoutAgent()
        report = agent.scout(player_name, league)
        return {"player": player_name, "report": report}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/agents/post-match")
def post_match_report(
    home: str = Query(...),
    away: str = Query(...),
    league: str = Query(...),
    match_date: Optional[str] = Query(None),
):
    """Relatorio pos-jogo."""
    from agents.reporter import ReporterAgent

    try:
        agent = ReporterAgent()
        report = agent.post_match(home, away, league, match_date)
        return {"home_team": home, "away_team": away, "league": league, "report": report}
    except Exception as e:
        raise HTTPException(400, str(e))


# ─── Stats ─────────────────────────────────────────────────────────────

@app.get("/stats/database")
def database_stats():
    """Estatisticas do banco de dados."""
    conn = get_conn()
    cur = conn.cursor()

    stats = {}
    for table in ["matches", "teams", "players", "team_match_stats", "player_match_stats", "odds"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cur.fetchone()[0]

    cur.execute("""
        SELECT league, COUNT(*) as total,
               MIN(match_date) as first_date, MAX(match_date) as last_date,
               COUNT(CASE WHEN home_xg IS NOT NULL THEN 1 END) as with_xg
        FROM matches GROUP BY league ORDER BY total DESC
    """)
    stats["per_league"] = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT COUNT(DISTINCT source) as sources FROM matches")
    stats["sources"] = cur.fetchone()["sources"]

    conn.close()
    return stats


@app.get("/stats/models")
def model_stats():
    """Metricas dos modelos treinados."""
    import json

    results_path = MODELS_DIR / "training_results.json"
    if not results_path.exists():
        return {"status": "not_trained", "message": "Nenhum modelo treinado ainda"}

    with open(results_path) as f:
        results = json.load(f)

    return {"status": "trained", "results": results}


@app.get("/stats/leagues/{league_name}")
def league_stats(league_name: str):
    """Estatisticas detalhadas de uma liga."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) as total, MIN(match_date) as first, MAX(match_date) as last FROM matches WHERE league = ?",
        (league_name,),
    )
    general = dict(cur.fetchone())

    cur.execute(
        "SELECT season, COUNT(*) FROM matches WHERE league = ? GROUP BY season ORDER BY season",
        (league_name,),
    )
    general["per_season"] = [dict(r) for r in cur.fetchall()]

    cur.execute(
        """SELECT
            AVG(home_corners + away_corners) as avg_corners,
            AVG(home_shots + away_shots) as avg_shots,
            AVG(home_goals + away_goals) as avg_goals,
            AVG(CASE WHEN home_xg IS NOT NULL THEN home_xg + away_xg END) as avg_xg
           FROM matches WHERE league = ?""",
        (league_name,),
    )
    general["averages"] = dict(cur.fetchone())

    conn.close()

    if general["total"] == 0:
        raise HTTPException(404, f"Liga {league_name} sem dados")

    return general


# ─── Trigger ───────────────────────────────────────────────────────────

@app.post("/trigger/collect")
def trigger_collect():
    """Dispara coleta de dados do dia."""
    from collectors.api_football_collector import APIFootballCollector

    try:
        api = APIFootballCollector()
        fixtures = api.collect_today()
        from database.merge import merge_all
        merge_all()
        return {"status": "ok", "fixtures_collected": len(fixtures)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/trigger/predict")
def trigger_predict():
    """Dispara previsoes para jogos do dia."""
    from datetime import datetime
    from collectors.api_football_collector import APIFootballCollector
    from models.corners.corners_predictor import CornersPredictor
    from models.shots.shots_predictor import ShotsPredictor
    from models.results.dixon_coles import DixonColes
    from config.leagues import LEAGUES

    try:
        api = APIFootballCollector()
        today = datetime.now().strftime("%Y-%m-%d")
        fixtures = api.get_fixtures(today)

        league_names = {l.name for l in LEAGUES}
        fixtures = [f for f in fixtures if f.get("league") in league_names]

        predictions = []

        for fx in fixtures:
            home = fx.get("home_team", "")
            away = fx.get("away_team", "")
            league = fx.get("league", "")
            cp = CornersPredictor(league=league)
            sp = ShotsPredictor(league=league)
            corners = cp.predict(home, away, league, today)
            shots = sp.predict(home, away, league, today)
            dc = DixonColes(league=league)
            result = dc.predict_score(home, away) if dc.load() else None
            predictions.append({
                "home_team": home, "away_team": away, "league": league,
                "corners": _serialize(corners), "shots": _serialize(shots), "result": _serialize(result),
            })

        return {"status": "ok", "predictions": len(predictions)}
    except Exception as e:
        raise HTTPException(500, str(e))
