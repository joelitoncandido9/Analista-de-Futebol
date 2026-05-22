"""Coletor de dados historicos da football-data.co.uk.

Baixa CSVs com escanteios, finalizacoes, cartoes, odds — 25 temporadas de 6 ligas.
"""
import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from .base_collector import BaseCollector
from config.leagues import LEAGUES, FD_CODE_TO_LEAGUE
from config.settings import RAW_DIR


# Mapeamento: codigo football-data -> codigo interno para colunas
SEASON_CODES = [
    "0001", "0102", "0203", "0304", "0405", "0506", "0607", "0708", "0809", "0910",
    "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425",
]

FD_BASE = "https://www.football-data.co.uk/mmz4281"

# Mapeamento de colunas do CSV para nossos campos
COLUMN_MAP = {
    "HC": "home_corners",
    "AC": "away_corners",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HTHG": "home_ht_goals",
    "HTAG": "away_ht_goals",
    "Referee": "referee",
}


class FootballDataCollector(BaseCollector):
    """Coleta dados historicos de partidas do football-data.co.uk."""

    def __init__(self):
        super().__init__("FootballData", rate_per_min=30)

    def _parse_season(self, season_code: str) -> str:
        """Converte 2324 -> 2023/2024."""
        y1 = int(season_code[:2])
        y2 = int(season_code[2:])
        # heuristic: se for tipo 2425, é 2024/2025
        if y1 > 90:
            cent = 1900
        else:
            cent = 2000
        return f"{cent + y1}/{cent + y2}"

    def download_league(self, league_code: str, season_code: str) -> pd.DataFrame | None:
        """Baixa CSV de uma liga/temporada e retorna DataFrame."""
        # Codigo correto: Brasileirao usa B1, nao BR1
        csv_code = league_code
        url = f"{FD_BASE}/{season_code}/{csv_code}.csv"

        try:
            resp = requests.get(url, timeout=30, allow_redirects=True)
            if resp.status_code != 200:
                return None

            # Tentar com encoding latino padrao
            df = pd.read_csv(io.StringIO(resp.text), encoding="latin1")

            # Renomear colunas
            df.rename(columns=COLUMN_MAP, inplace=True)

            # Normalizar nomes de times
            for col in ["HomeTeam", "AwayTeam"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

            return df

        except Exception as e:
            logger.warning(f"Erro baixando {url}: {e}")
            return None

    def parse_matches(self, df: pd.DataFrame, league: str, season_code: str) -> list[dict]:
        """Converte DataFrame do CSV para lista de dicts no formato do banco."""
        season = self._parse_season(season_code)
        matches = []

        for _, row in df.iterrows():
            match_date = None
            if "Date" in df.columns:
                date_str = str(row.get("Date", "")).strip()
                if date_str and date_str != "nan":
                    try:
                        # Trata formatos: dd/mm/yyyy ou dd/mm/yy
                        parts = date_str.split("/")
                        if len(parts) == 3:
                            day, month, year = parts
                            if len(year) == 2:
                                year = f"20{year}"
                            match_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    except:
                        pass

            home = str(row.get("HomeTeam", ""))
            away = str(row.get("AwayTeam", ""))

            if not home or not away:
                continue

            match_id = f"fd_{league}_{season_code}_{home}_{away}"
            match_id = re.sub(r"[^a-zA-Z0-9_-]", "_", match_id)

            match = {
                "match_id": match_id,
                "league": league,
                "season": season,
                "round": None,
                "match_date": match_date,
                "home_team": home,
                "away_team": away,
                "home_goals": self._int(row, "home_goals"),
                "away_goals": self._int(row, "away_goals"),
                "home_shots": self._int(row, "home_shots"),
                "away_shots": self._int(row, "away_shots"),
                "home_shots_on_target": self._int(row, "home_shots_on_target"),
                "away_shots_on_target": self._int(row, "away_shots_on_target"),
                "home_corners": self._int(row, "home_corners"),
                "away_corners": self._int(row, "away_corners"),
                "home_fouls": self._int(row, "home_fouls"),
                "away_fouls": self._int(row, "away_fouls"),
                "home_yellow": self._int(row, "home_yellow"),
                "away_yellow": self._int(row, "away_yellow"),
                "home_red": self._int(row, "home_red"),
                "away_red": self._int(row, "away_red"),
                "home_xg": None,
                "away_xg": None,
                "home_possession": None,
                "away_possession": None,
                "home_ppda": None,
                "away_ppda": None,
                "home_deep": None,
                "away_deep": None,
                "referee": str(row.get("referee", "")) if "referee" in df.columns else None,
                "venue": None,
                "status": "finished",
                "source": "football_data",
            }
            matches.append(match)

        return matches

    def _int(self, row, col: str) -> int | None:
        val = row.get(col)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    def collect_league(self, league_name: str, max_seasons: int | None = None) -> int:
        """Coleta todas as temporadas de uma liga."""
        from config.leagues import get_league
        from database.queries import save_matches

        league = get_league(league_name)
        codes = SEASON_CODES
        if max_seasons:
            codes = codes[-max_seasons:]

        total = 0
        self.log_start(f"coleta {league.name} ({len(codes)} temporadas)")

        for season_code in codes:
            df = self.download_league(league.football_data_code, season_code)
            if df is None or df.empty:
                continue

            matches = self.parse_matches(df, league.name, season_code)
            if matches:
                saved = save_matches(matches)
                total += saved
                logger.info(f"  {season_code}: {saved} partidas")

        self.log_success(f"coleta {league.name}", total)
        return total

    def collect_all(self, max_seasons: int | None = None) -> dict[str, int]:
        """Coleta todas as 6 ligas."""
        from config.leagues import LEAGUES

        results = {}
        for league in LEAGUES:
            count = self.collect_league(league.name, max_seasons)
            results[league.name] = count
        return results
