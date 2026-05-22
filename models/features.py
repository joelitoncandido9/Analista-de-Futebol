"""Feature engineering para modelos de escanteios, finalizacoes e resultados.

Gera features a partir do banco SQLite:
- Medias moveis das ultimas N partidas (escanteios, finalizacoes, xG, gols)
- Forca de ataque/defesa
- Vantagem de casa (media historica por liga)
- Momentum recente (pontos ultimos 5 jogos)
- Confronto direto (H2H)
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DB_PATH


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_matches(league: str | None = None,
                 min_date: str | None = None,
                 max_date: str | None = None,
                 require_corners: bool = True) -> pd.DataFrame:
    """Carrega partidas do banco como DataFrame, ordenadas por data.

    Args:
        require_corners: Se True (padrão), só retorna partidas com escanteios.
                         Para Dixon-Coles (só precisa de gols), usar False.
    """
    conn = _conn()
    query = """SELECT id, match_id, league, season, match_date,
                      home_team, away_team,
                      home_goals, away_goals,
                      home_xg, away_xg,
                      home_shots, away_shots,
                      home_shots_on_target, away_shots_on_target,
                      home_corners, away_corners,
                      home_fouls, away_fouls,
                      home_yellow, away_yellow,
                      home_red, away_red,
                      home_possession, away_possession,
                      home_ppda, away_ppda,
                      home_deep, away_deep
               FROM matches
               WHERE 1=1"""

    if require_corners:
        query += " AND home_corners IS NOT NULL"

    params = []
    if league:
        query += " AND league = ?"
        params.append(league)
    if min_date:
        query += " AND match_date >= ?"
        params.append(min_date)
    if max_date:
        query += " AND match_date <= ?"
        params.append(max_date)

    query += " ORDER BY match_date ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return df

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["total_corners"] = df["home_corners"] + df["away_corners"]
    df["total_shots"] = df["home_shots"] + df["away_shots"]
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    return df


def _rolling_avg(team_df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Media movel de uma coluna para um time, excluindo o jogo atual."""
    return team_df[col].shift(1).rolling(window, min_periods=1).mean()


def build_team_features(team: str, match_date: pd.Timestamp,
                        df: pd.DataFrame, windows: list[int] | None = None
                        ) -> dict:
    """Calcula features rolling para um time ate uma data especifica."""
    if windows is None:
        windows = [5, 10]

    team_df = df[
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["match_date"] < match_date)
    ].copy()

    if team_df.empty:
        return _default_team_features(team, windows)

    team_df["is_home"] = team_df["home_team"] == team
    team_df["goals_for"] = np.where(team_df["is_home"],
                                    team_df["home_goals"], team_df["away_goals"])
    team_df["goals_against"] = np.where(team_df["is_home"],
                                        team_df["away_goals"], team_df["home_goals"])
    team_df["corners_for"] = np.where(team_df["is_home"],
                                       team_df["home_corners"], team_df["away_corners"])
    team_df["corners_against"] = np.where(team_df["is_home"],
                                           team_df["away_corners"], team_df["home_corners"])
    team_df["shots_for"] = np.where(team_df["is_home"],
                                     team_df["home_shots"], team_df["away_shots"])
    team_df["shots_against"] = np.where(team_df["is_home"],
                                         team_df["away_shots"], team_df["home_shots"])
    team_df["xg_for"] = np.where(team_df["is_home"],
                                  team_df["home_xg"].fillna(0), team_df["away_xg"].fillna(0))
    team_df["xg_against"] = np.where(team_df["is_home"],
                                      team_df["away_xg"].fillna(0), team_df["home_xg"].fillna(0))
    team_df["points"] = np.where(
        team_df["goals_for"] > team_df["goals_against"], 3,
        np.where(team_df["goals_for"] == team_df["goals_against"], 1, 0),
    )

    features = {}
    # Medias rolling por janela
    for w in windows:
        features[f"corners_for_avg_{w}"] = _rolling_avg(team_df, "corners_for", w).iloc[-1] if len(team_df) >= 1 else np.nan
        features[f"corners_against_avg_{w}"] = _rolling_avg(team_df, "corners_against", w).iloc[-1] if len(team_df) >= 1 else np.nan
        features[f"shots_for_avg_{w}"] = _rolling_avg(team_df, "shots_for", w).iloc[-1] if len(team_df) >= 1 else np.nan
        features[f"shots_against_avg_{w}"] = _rolling_avg(team_df, "shots_against", w).iloc[-1] if len(team_df) >= 1 else np.nan
        features[f"goals_for_avg_{w}"] = _rolling_avg(team_df, "goals_for", w).iloc[-1] if len(team_df) >= 1 else np.nan
        features[f"goals_against_avg_{w}"] = _rolling_avg(team_df, "goals_against", w).iloc[-1] if len(team_df) >= 1 else np.nan
        features[f"xg_for_avg_{w}"] = _rolling_avg(team_df, "xg_for", w).iloc[-1] if len(team_df) >= 1 else np.nan
        features[f"xg_against_avg_{w}"] = _rolling_avg(team_df, "xg_against", w).iloc[-1] if len(team_df) >= 1 else np.nan
        features[f"points_avg_{w}"] = _rolling_avg(team_df, "points", w).iloc[-1] if len(team_df) >= 1 else np.nan

    # Total de jogos como historico (pra confianca)
    features["team_total_games"] = len(team_df)

    return features


def _default_team_features(team: str, windows: list[int]) -> dict:
    """Valores padrao (NaN) para time sem historico."""
    features = {}
    for w in windows:
        for stat in ["corners_for_avg", "corners_against_avg",
                      "shots_for_avg", "shots_against_avg",
                      "goals_for_avg", "goals_against_avg",
                      "xg_for_avg", "xg_against_avg",
                      "points_avg"]:
            features[f"{stat}_{w}"] = np.nan
    features["team_total_games"] = 0
    return features


def build_match_features(row: pd.Series, df: pd.DataFrame) -> dict:
    """Gera feature vector completo para uma partida."""
    match_date = row["match_date"]
    league = row["league"]
    home = row["home_team"]
    away = row["away_team"]

    league_df = df[df["league"] == league]

    home_feats = build_team_features(home, match_date, league_df)
    away_feats = build_team_features(away, match_date, league_df)

    features = {}
    for w in [5, 10]:
        # Escanteios
        features[f"home_corners_for_avg_{w}"] = home_feats.get(f"corners_for_avg_{w}")
        features[f"away_corners_for_avg_{w}"] = away_feats.get(f"corners_for_avg_{w}")
        features[f"home_corners_against_avg_{w}"] = home_feats.get(f"corners_against_avg_{w}")
        features[f"away_corners_against_avg_{w}"] = away_feats.get(f"corners_against_avg_{w}")

        # Finalizacoes
        features[f"home_shots_for_avg_{w}"] = home_feats.get(f"shots_for_avg_{w}")
        features[f"away_shots_for_avg_{w}"] = away_feats.get(f"shots_for_avg_{w}")
        features[f"home_shots_against_avg_{w}"] = home_feats.get(f"shots_against_avg_{w}")
        features[f"away_shots_against_avg_{w}"] = away_feats.get(f"shots_against_avg_{w}")

        # Gols
        features[f"home_goals_for_avg_{w}"] = home_feats.get(f"goals_for_avg_{w}")
        features[f"away_goals_for_avg_{w}"] = away_feats.get(f"goals_for_avg_{w}")
        features[f"home_goals_against_avg_{w}"] = home_feats.get(f"goals_against_avg_{w}")
        features[f"away_goals_against_avg_{w}"] = away_feats.get(f"goals_against_avg_{w}")

        # xG
        features[f"home_xg_for_avg_{w}"] = home_feats.get(f"xg_for_avg_{w}")
        features[f"away_xg_for_avg_{w}"] = away_feats.get(f"xg_for_avg_{w}")

        # Pontos (forma)
        features[f"home_points_avg_{w}"] = home_feats.get(f"points_avg_{w}")
        features[f"away_points_avg_{w}"] = away_feats.get(f"points_avg_{w}")

    # Diferenca entre os times (home - away)
    for w in [5, 10]:
        for stat in ["corners_for_avg", "corners_against_avg",
                      "shots_for_avg", "shots_against_avg",
                      "goals_for_avg", "goals_against_avg",
                      "points_avg"]:
            h = home_feats.get(f"{stat}_{w}")
            a = away_feats.get(f"{stat}_{w}")
            if h is not None and a is not None and not (np.isnan(h) if isinstance(h, float) else False) and not (np.isnan(a) if isinstance(a, float) else False):
                features[f"diff_{stat}_{w}"] = h - a
            else:
                features[f"diff_{stat}_{w}"] = np.nan

    # Historico dos times
    features["home_total_games"] = home_feats.get("team_total_games", 0)
    features["away_total_games"] = away_feats.get("team_total_games", 0)

    return features


def build_dataset(league: str | None = None,
                  min_date: str | None = None,
                  max_date: str | None = None,
                  target_col: str = "total_corners",
                  min_data_games: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Gera dataset completo com features + target para treinamento.

    Returns:
        (X_df, y_series) onde X tem as features e y o target.
    """
    df = load_matches(league, min_date, max_date)
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    rows = []
    targets = []
    for idx, row in df.iterrows():
        feats = build_match_features(row, df)
        if feats["home_total_games"] < min_data_games or feats["away_total_games"] < min_data_games:
            continue
        target_val = row.get(target_col)
        if target_val is None or (isinstance(target_val, float) and np.isnan(target_val)):
            continue
        rows.append(feats)
        targets.append(target_val)

    X = pd.DataFrame(rows)
    y = pd.Series(targets, name=target_col)
    return X, y


def temporal_split(X: pd.DataFrame, y: pd.Series,
                   test_games: int = 380) -> tuple:
    """Split temporal: ultimos N jogos sao teste, resto treino.

    Como os dados sao ordenados por data no dataset,
    podemos usar o indice diretamente.
    """
    if len(X) <= test_games:
        return X, y, X, y  # Poucos dados, tudo treino

    split = len(X) - test_games
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test


def league_averages(df: pd.DataFrame) -> dict:
    """Medias historicas de escanteios/finalizacoes por liga."""
    avgs = {}
    for league in df["league"].unique():
        ld = df[df["league"] == league]
        avgs[league] = {
            "avg_total_corners": ld["total_corners"].mean(),
            "avg_home_corners": ld["home_corners"].mean(),
            "avg_away_corners": ld["away_corners"].mean(),
            "avg_total_shots": ld["total_shots"].mean(),
            "avg_home_shots": ld["home_shots"].mean(),
            "avg_away_shots": ld["away_shots"].mean(),
            "avg_home_goals": ld["home_goals"].mean(),
            "avg_away_goals": ld["away_goals"].mean(),
        }
    return avgs
