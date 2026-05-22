"""Coletor de dados da API-Football.

Usa 3 keys rotacionadas para buscar fixtures, estatisticas, lineups e eventos
das partidas do dia e proximas rodadas.
"""
import time
from datetime import datetime, timedelta

import requests
from loguru import logger

from .base_collector import BaseCollector
from config.leagues import LEAGUES, get_league_by_api_id
from config.settings import API_FOOTBALL_KEYS, API_FOOTBALL_URL


class APIFootballCollector(BaseCollector):
    """Coleta fixtures, estatisticas e eventos da API-Football."""

    def __init__(self):
        super().__init__("API-Football", rate_per_min=10)
        self._key_index = 0

    def _next_key(self) -> str:
        """Rotaciona entre as keys disponiveis."""
        key = API_FOOTBALL_KEYS[self._key_index % len(API_FOOTBALL_KEYS)]
        self._key_index += 1
        return key

    def _call(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Chamada a API-Football com rotacao de keys e retry."""
        url = f"{API_FOOTBALL_URL}{endpoint}"

        for attempt in range(3):
            key = self._next_key()
            headers = {
                "x-apisports-key": key,
                "Accept": "application/json",
            }
            self._rate_limit()

            try:
                resp = requests.get(url, headers=headers, params=params, timeout=20)
                if resp.status_code == 429:
                    logger.warning(f"  Rate limit key, tentando proxima...")
                    continue
                if resp.status_code != 200:
                    logger.warning(f"  HTTP {resp.status_code} em {endpoint}")
                    time.sleep(2 ** attempt)
                    continue

                data = resp.json()
                if data.get("errors"):
                    logger.warning(f"  Erro API: {data['errors']}")
                    return None
                return data

            except Exception as e:
                logger.warning(f"  Erro: {e}")
                time.sleep(2 ** attempt)

        return None

    def get_fixtures(self, date: str | None = None) -> list[dict]:
        """Busca fixtures de todas as ligas para uma data."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        result = self._call("/fixtures", {"date": date})
        if not result or not result.get("response"):
            return []

        fixtures = []
        for r in result["response"]:
            league = r["league"]
            lid = league["id"]
            matched = get_league_by_api_id(lid)
            if not matched:
                continue

            fixture = r["fixture"]
            teams = r["teams"]

            fixtures.append({
                "fixture_id": str(fixture["id"]),
                "date": date,
                "league": matched.name,
                "home_team": teams["home"]["name"],
                "away_team": teams["away"]["name"],
                "venue": fixture.get("venue", {}).get("name", ""),
                "status": fixture["status"]["short"],
                "home_goals": teams["home"].get("goals") or fixture["goals"].get("home"),
                "away_goals": teams["away"].get("goals") or fixture["goals"].get("away"),
                "match_date": fixture.get("date", ""),
            })

        return fixtures

    def get_fixture_stats(self, fixture_id: str) -> dict | None:
        """Busca estatisticas detalhadas de uma partida."""
        result = self._call(f"/fixtures/statistics", {"fixture": fixture_id})
        if not result or not result.get("response"):
            return None

        stats = {}
        for team_stats in result["response"]:
            team = team_stats["team"]["name"]
            for stat in team_stats["statistics"]:
                key = stat["type"].lower().replace(" ", "_")
                value = stat.get("value")
                stats[f"{team}_{key}"] = value

        return stats

    def get_fixture_events(self, fixture_id: str) -> list[dict]:
        """Busca eventos (gols, cartoes, substituicoes) de uma partida."""
        result = self._call(f"/fixtures/events", {"fixture": fixture_id})
        if not result or not result.get("response"):
            return []
        return result["response"]

    def get_fixture_lineups(self, fixture_id: str) -> list[dict]:
        """Busca escalacoes de uma partida."""
        result = self._call(f"/fixtures/lineups", {"fixture": fixture_id})
        if not result or not result.get("response"):
            return []
        return result["response"]

    def get_player_stats(self, fixture_id: str) -> list[dict]:
        """Busca estatisticas individuais dos jogadores."""
        result = self._call(f"/fixtures/players", {"fixture": fixture_id})
        if not result or not result.get("response"):
            return []
        players = []
        for team_data in result["response"]:
            team = team_data["team"]["name"]
            for p in team_data.get("players", []):
                player = p["player"]
                stats = p.get("statistics", [{}])[0] if p.get("statistics") else {}
                players.append({
                    "player_name": player.get("name", ""),
                    "team": team,
                    "position": stats.get("games", {}).get("position", ""),
                    "rating": stats.get("games", {}).get("rating"),
                    "minutes": stats.get("games", {}).get("minutes"),
                    "goals": stats.get("goals", {}).get("total"),
                    "assists": stats.get("goals", {}).get("assists"),
                    "shots": stats.get("shots", {}).get("total"),
                    "shots_on_target": stats.get("shots", {}).get("on"),
                    "key_passes": stats.get("passes", {}).get("key"),
                    "passes_accuracy": stats.get("passes", {}).get("accuracy"),
                    "tackles": stats.get("tackles", {}).get("total"),
                    "fouls": stats.get("fouls", {}).get("committed"),
                    "yellow": 1 if stats.get("cards", {}).get("yellow") == "1" else 0,
                    "red": 1 if stats.get("cards", {}).get("red") == "1" else 0,
                    "dribbles": stats.get("dribbles", {}).get("attempts"),
                    "dribbles_success": stats.get("dribbles", {}).get("success"),
                })
        return players

    def collect_today(self) -> list[dict]:
        """Coleta fixtures de hoje + estatisticas das partidas encerradas."""
        self.log_start("coleta do dia")
        today = datetime.now().strftime("%Y-%m-%d")

        # 1. Buscar fixtures do dia
        fixtures = self.get_fixtures(today)
        if not fixtures:
            logger.info("Nenhum jogo hoje")
            return []

        logger.info(f"  {len(fixtures)} jogos encontrados")

        # 2. Para cada jogo encerrado, buscar estatisticas
        from database.queries import save_matches

        matches = []
        for fx in fixtures:
            if fx["status"] in ("FT", "AET", "PEN"):
                stats = self.get_fixture_stats(fx["fixture_id"])
                if stats:
                    # Mapear estatisticas para nosso formato
                    match = {
                        "match_id": f"api_{fx['fixture_id']}",
                        "league": fx["league"],
                        "season": "2025/2026",
                        "match_date": fx["match_date"],
                        "home_team": fx["home_team"],
                        "away_team": fx["away_team"],
                        "home_goals": fx["home_goals"],
                        "away_goals": fx["away_goals"],
                        "status": "finished",
                        "source": "api_football",
                    }

                    # Extrair estatisticas do formato API
                    for team_key in ["home_team", "away_team"]:
                        team = fx[team_key]
                        prefix = team_key.split("_")[0]

                        match[f"{prefix}_shots"] = _find_stat(stats, team, "Shots on Goal") or _find_stat(stats, team, "Total Shots")
                        match[f"{prefix}_shots_on_target"] = _find_stat(stats, team, "Shots on Goal")
                        match[f"{prefix}_possession"] = _find_stat(stats, team, "Ball Possession")
                        match[f"{prefix}_yellow"] = _find_stat(stats, team, "Yellow Cards")
                        match[f"{prefix}_red"] = _find_stat(stats, team, "Red Cards")
                        match[f"{prefix}_fouls"] = _find_stat(stats, team, "Fouls")
                        match[f"{prefix}_corners"] = _find_stat(stats, team, "Corner Kicks")

                    matches.append(match)

                time.sleep(1)

        if matches:
            saved = save_matches(matches)
            logger.info(f"  {saved} partidas salvas")
        else:
            logger.info("  Nenhuma partida encerrada hoje")

        return fixtures

    def collect_upcoming(self, days: int = 7) -> list[dict]:
        """Coleta fixtures dos proximos dias."""
        fixtures = []
        for d in range(1, days + 1):
            date = (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")
            day_fixtures = self.get_fixtures(date)
            fixtures.extend(day_fixtures)
        logger.info(f"  {len(fixtures)} jogos futuros encontrados")
        return fixtures


def _find_stat(stats: dict, team: str, stat_name: str) -> int | None:
    """Busca valor de estatistica no dict retornado pela API."""
    key = f"{team}_{stat_name.lower().replace(' ', '_')}"
    val = stats.get(key)
    if val is None:
        return None
    if isinstance(val, str):
        if "%" in val:
            try:
                return int(val.replace("%", ""))
            except ValueError:
                return None
        try:
            return int(val)
        except ValueError:
            return None
    if isinstance(val, (int, float)):
        return int(val)
    return None
