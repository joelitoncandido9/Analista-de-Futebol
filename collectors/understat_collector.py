"""Coletor de dados do Understat via soccerdata.

Fornece xG, xA, PPDA, deep completions para temporada atual.
"""
from loguru import logger

from .base_collector import BaseCollector
from config.leagues import LEAGUES


class UnderstatCollector(BaseCollector):
    """Coleta xG, xA, PPDA, deep completions do Understat."""

    def __init__(self):
        super().__init__("Understat", rate_per_min=30)

    def collect_league(self, league_name: str, season: str = "2425") -> int:
        """Coleta dados de uma liga do Understat."""
        from config.leagues import get_league
        from database.queries import save_matches

        league = get_league(league_name)
        if not league.understat_name:
            logger.warning(f"  Understat nao cobre {league.name}, pulando")
            return 0
        self.log_start(f"coleta Understat {league.name}")

        try:
            from soccerdata import Understat

            understat = Understat(leagues=league.understat_name, seasons=season)

            # Schedule com resultados
            sched = understat.read_schedule()
            if sched is None or sched.empty:
                logger.warning(f"  Sem schedule para {league.name}")
                return 0

            # Team match stats (xG, PPDA, deep completions)
            stats_df = understat.read_team_match_stats()
            if stats_df is None or stats_df.empty:
                logger.warning(f"  Sem stats para {league.name}")
                return 0

            # Merge schedule + stats
            matches = []
            for (league_key, season_key, game_key), game in sched.iterrows():
                match_id = f"understat_{league.name}_{game_key}"
                match_id = match_id.replace(" ", "_").replace("-", "_")

                # Buscar stats correspondentes
                stats_row = None
                try:
                    if (league_key, season_key, game_key) in stats_df.index:
                        stats_row = stats_df.loc[(league_key, season_key, game_key)]
                except:
                    pass

                match = {
                    "match_id": match_id,
                    "league": league.name,
                    "season": f"20{season[:2]}/20{season[2:]}",
                    "round": None,
                    "match_date": str(game.get("date", "")),
                    "home_team": str(game.get("home_team", "")),
                    "away_team": str(game.get("away_team", "")),
                    "home_goals": self._int(game, "home_goals"),
                    "away_goals": self._int(game, "away_goals"),
                    "home_xg": self._float(game, "home_xg"),
                    "away_xg": self._float(game, "away_xg"),
                    "home_shots": None,
                    "away_shots": None,
                    "home_shots_on_target": None,
                    "away_shots_on_target": None,
                    "home_corners": None,
                    "away_corners": None,
                    "home_fouls": None,
                    "away_fouls": None,
                    "home_yellow": None,
                    "away_yellow": None,
                    "home_red": None,
                    "away_red": None,
                    "home_possession": None,
                    "away_possession": None,
                    "home_ppda": self._float(stats_row, "home_ppda") if stats_row is not None else None,
                    "away_ppda": self._float(stats_row, "away_ppda") if stats_row is not None else None,
                    "home_deep": self._int(stats_row, "home_deep_completions") if stats_row is not None else None,
                    "away_deep": self._int(stats_row, "away_deep_completions") if stats_row is not None else None,
                    "referee": None,
                    "venue": None,
                    "status": "finished" if game.get("is_result") else "scheduled",
                    "source": "understat",
                }
                matches.append(match)

            # Salvar no banco
            saved = save_matches(matches)
            self.log_success(f"Understat {league.name}", saved)
            return saved

        except Exception as e:
            self.log_error(f"Understat {league.name}", e)
            return 0

    def collect_all(self, season: str = "2425") -> dict[str, int]:
        """Coleta todas as 6 ligas do Understat."""
        results = {}
        for league in LEAGUES:
            count = self.collect_league(league.name, season)
            results[league.name] = count
        return results

    def _int(self, row, col: str) -> int | None:
        if row is None:
            return None
        try:
            v = row.get(col) if hasattr(row, "get") else None
            if v is None:
                return None
            return int(v)
        except (ValueError, TypeError, AttributeError):
            return None

    def _float(self, row, col: str) -> float | None:
        if row is None:
            return None
        try:
            v = row.get(col) if hasattr(row, "get") else None
            if v is None:
                return None
            return round(float(v), 2)
        except (ValueError, TypeError, AttributeError):
            return None
