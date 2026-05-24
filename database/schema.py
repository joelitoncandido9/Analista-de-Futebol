"""Schema do banco SQLite."""
import sqlite3
from pathlib import Path

from config.settings import DB_PATH


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT UNIQUE,
    league TEXT,
    season TEXT,
    round INTEGER,
    match_date TEXT,
    home_team TEXT,
    away_team TEXT,
    home_goals INTEGER,
    away_goals INTEGER,
    home_xg REAL,
    away_xg REAL,
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    home_corners INTEGER,
    away_corners INTEGER,
    home_fouls INTEGER,
    away_fouls INTEGER,
    home_yellow INTEGER,
    away_yellow INTEGER,
    home_red INTEGER,
    away_red INTEGER,
    home_possession INTEGER,
    away_possession INTEGER,
    home_ppda REAL,
    away_ppda REAL,
    home_deep INTEGER,
    away_deep INTEGER,
    referee TEXT,
    venue TEXT,
    status TEXT,
    api_fixture_id TEXT,
    source TEXT DEFAULT 'football_data',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    short_name TEXT,
    country TEXT,
    api_team_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    team TEXT,
    position TEXT,
    nationality TEXT,
    api_player_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, team)
);

CREATE TABLE IF NOT EXISTS team_match_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT,
    team TEXT,
    league TEXT,
    season TEXT,
    goals INTEGER,
    xg REAL,
    shots INTEGER,
    shots_on_target INTEGER,
    corners INTEGER,
    fouls INTEGER,
    yellow INTEGER,
    red INTEGER,
    possession INTEGER,
    ppda REAL,
    deep_completions INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT,
    player TEXT,
    team TEXT,
    minutes_played INTEGER,
    goals INTEGER,
    assists INTEGER,
    shots INTEGER,
    shots_on_target INTEGER,
    key_passes INTEGER,
    passes_completed INTEGER,
    passes_attempted INTEGER,
    dribbles INTEGER,
    tackles INTEGER,
    fouls INTEGER,
    yellow INTEGER,
    red INTEGER,
    xg REAL,
    xa REAL,
    rating REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT,
    bookmaker TEXT,
    market TEXT,
    selection TEXT,
    odd_value REAL,
    timestamp TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS collection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    league TEXT,
    season TEXT,
    rows_collected INTEGER,
    status TEXT DEFAULT 'success',
    error TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_team_stats_match ON team_match_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_match_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_id);

CREATE TABLE IF NOT EXISTS market_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT '',
    threshold REAL NOT NULL,
    accuracy REAL NOT NULL,
    n_samples INTEGER NOT NULL,
    calibrated_bucket TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(market, direction)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fixture_id TEXT,
    home_team TEXT,
    away_team TEXT,
    league TEXT,
    match_date TEXT,
    market TEXT,
    line REAL,
    direction TEXT,
    probability REAL,
    predicted_value REAL,
    actual_value REAL,
    was_correct INTEGER,
    source TEXT NOT NULL DEFAULT 'model',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(fixture_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(match_date);
CREATE INDEX IF NOT EXISTS idx_predictions_source ON predictions(source);
CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique ON predictions(fixture_id, market, direction, line, source);
"""


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Retorna conexao com o banco."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path | None = None):
    """Cria todas as tabelas se nao existirem."""
    conn = get_conn(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
    print(f"✅ Banco inicializado: {db_path or DB_PATH}")
