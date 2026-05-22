"""Queries reutilizaveis para o banco."""
from database.schema import get_conn


def get_matches_by_league(league: str, season: str | None = None, limit: int = 100):
    """Retorna partidas de uma liga."""
    conn = get_conn()
    cur = conn.cursor()
    if season:
        cur.execute(
            "SELECT * FROM matches WHERE league = ? AND season = ? ORDER BY match_date DESC LIMIT ?",
            (league, season, limit),
        )
    else:
        cur.execute(
            "SELECT * FROM matches WHERE league = ? ORDER BY match_date DESC LIMIT ?",
            (league, limit),
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_player_matches(player: str, limit: int = 20):
    """Retorna ultimas partidas de um jogador."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT pm.*, m.match_date, m.home_team, m.away_team
           FROM player_match_stats pm
           JOIN matches m ON pm.match_id = m.match_id
           WHERE pm.player = ?
           ORDER BY m.match_date DESC LIMIT ?""",
        (player, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_team_recent_matches(team: str, league: str | None = None, n: int = 10):
    """Retorna ultimos N jogos de um time."""
    conn = get_conn()
    cur = conn.cursor()
    if league:
        cur.execute(
            """SELECT * FROM matches
               WHERE (home_team = ? OR away_team = ?) AND league = ?
               ORDER BY match_date DESC LIMIT ?""",
            (team, team, league, n),
        )
    else:
        cur.execute(
            """SELECT * FROM matches
               WHERE home_team = ? OR away_team = ?
               ORDER BY match_date DESC LIMIT ?""",
            (team, team, n),
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_team_avg_stats(team: str, league: str, season: str, n: int = 10):
    """Media das estatisticas de um time nos ultimos N jogos."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT
            AVG(CASE WHEN home_team = ? THEN home_corners ELSE away_corners END) as avg_corners_for,
            AVG(CASE WHEN home_team = ? THEN away_corners ELSE home_corners END) as avg_corners_against,
            AVG(CASE WHEN home_team = ? THEN home_shots ELSE away_shots END) as avg_shots_for,
            AVG(CASE WHEN home_team = ? THEN away_shots ELSE home_shots END) as avg_shots_against,
            AVG(CASE WHEN home_team = ? THEN home_xg ELSE away_xg END) as avg_xg_for,
            AVG(CASE WHEN home_team = ? THEN away_xg ELSE home_xg END) as avg_xg_against
           FROM matches
           WHERE (home_team = ? OR away_team = ?) AND league = ? AND season = ?
           ORDER BY match_date DESC LIMIT ?""",
        (team, team, team, team, team, team, team, team, league, season, n),
    )
    row = dict(cur.fetchone())
    conn.close()
    return row


def get_h2h_stats(team_a: str, team_b: str):
    """Estatisticas de confronto direto."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT
            COUNT(*) as total,
            SUM(CASE WHEN home_team = ? AND home_goals > away_goals THEN 1
                     WHEN away_team = ? AND away_goals > home_goals THEN 1 ELSE 0 END) as wins_a,
            SUM(CASE WHEN home_goals = away_goals THEN 1 ELSE 0 END) as draws,
            SUM(CASE WHEN home_team = ? AND home_goals < away_goals THEN 1
                     WHEN away_team = ? AND away_goals < home_goals THEN 1 ELSE 0 END) as wins_b,
            AVG(home_corners + away_corners) as avg_corners_total,
            AVG(home_shots + away_shots) as avg_shots_total
           FROM matches
           WHERE (home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?)""",
        (team_a, team_a, team_a, team_a, team_a, team_b, team_b, team_a),
    )
    row = dict(cur.fetchone())
    conn.close()
    return row


def save_matches(matches: list[dict]):
    """Salva lista de partidas no banco (upsert)."""
    conn = get_conn()
    cur = conn.cursor()
    saved = 0
    for m in matches:
        try:
            cur.execute(
                """INSERT OR REPLACE INTO matches
                   (match_id, league, season, round, match_date,
                    home_team, away_team, home_goals, away_goals,
                    home_xg, away_xg,
                    home_shots, away_shots,
                    home_shots_on_target, away_shots_on_target,
                    home_corners, away_corners,
                    home_fouls, away_fouls,
                    home_yellow, away_yellow,
                    home_red, away_red,
                    home_possession, away_possession,
                    home_ppda, away_ppda,
                    home_deep, away_deep,
                    referee, venue, status, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    m.get("match_id"), m.get("league"), m.get("season"),
                    m.get("round"), m.get("match_date"),
                    m.get("home_team"), m.get("away_team"),
                    m.get("home_goals"), m.get("away_goals"),
                    m.get("home_xg"), m.get("away_xg"),
                    m.get("home_shots"), m.get("away_shots"),
                    m.get("home_shots_on_target"), m.get("away_shots_on_target"),
                    m.get("home_corners"), m.get("away_corners"),
                    m.get("home_fouls"), m.get("away_fouls"),
                    m.get("home_yellow"), m.get("away_yellow"),
                    m.get("home_red"), m.get("away_red"),
                    m.get("home_possession"), m.get("away_possession"),
                    m.get("home_ppda"), m.get("away_ppda"),
                    m.get("home_deep"), m.get("away_deep"),
                    m.get("referee"), m.get("venue"),
                    m.get("status"), m.get("source", "football_data"),
                ),
            )
            saved += 1
        except Exception as e:
            print(f"  ⚠️ Erro salvando {m.get('match_id')}: {e}")
    conn.commit()
    conn.close()
    return saved


def save_predictions(predictions: list[dict]):
    """Salva previsões no banco para comparação futura."""
    conn = get_conn()
    cur = conn.cursor()
    saved = 0
    for p in predictions:
        try:
            cur.execute(
                """INSERT OR REPLACE INTO predictions
                   (fixture_id, home_team, away_team, league, match_date,
                    market, line, direction, probability,
                    predicted_value, actual_value, was_correct)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    p.get("fixture_id"),
                    p.get("home_team"),
                    p.get("away_team"),
                    p.get("league"),
                    p.get("match_date"),
                    p.get("market"),
                    p.get("line"),
                    p.get("direction"),
                    p.get("probability"),
                    p.get("predicted_value"),
                    p.get("actual_value"),
                    p.get("was_correct"),
                ),
            )
            saved += 1
        except Exception as e:
            print(f"  ⚠️ Erro salvando predição {p.get('fixture_id')}: {e}")
    conn.commit()
    conn.close()
    return saved


def evaluate_predictions():
    """Compara previsões com resultados reais e retorna estatísticas de acerto."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE predictions
        SET actual_value = (
            CASE WHEN market = 'total_corners' THEN home_corners + away_corners
                 WHEN market = 'total_shots' THEN home_shots + away_shots
            END
        ),
        was_correct = (
            CASE
                WHEN direction = 'over' AND
                     (CASE WHEN market = 'total_corners' THEN home_corners + away_corners
                           WHEN market = 'total_shots' THEN home_shots + away_shots
                     END) > line THEN 1
                WHEN direction = 'under' AND
                     (CASE WHEN market = 'total_corners' THEN home_corners + away_corners
                           WHEN market = 'total_shots' THEN home_shots + away_shots
                     END) < line THEN 1
                WHEN actual_value IS NOT NULL THEN 0
            END
        )
        FROM matches
        WHERE predictions.fixture_id = 'api_' || matches.match_id
        AND predictions.actual_value IS NULL
        AND matches.home_corners IS NOT NULL
    """)
    updated = cur.rowcount

    cur.execute("""
        SELECT market, direction, line,
               COUNT(*) as total,
               SUM(was_correct) as hits,
               ROUND(AVG(was_correct), 3) as accuracy
        FROM predictions
        WHERE was_correct IS NOT NULL
        GROUP BY market, direction, line
        ORDER BY accuracy DESC
    """)
    stats = [dict(r) for r in cur.fetchall()]
    conn.commit()
    conn.close()
    return {"evaluated": updated, "stats": stats}
