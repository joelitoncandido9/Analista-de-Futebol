"""Configuracao das 6 ligas monitoradas."""
from dataclasses import dataclass


@dataclass
class League:
    name: str
    country: str
    api_football_id: int
    football_data_code: str       # codigo usado no football-data.co.uk
    sport_key: str                # chave para the-odds-api
    understat_name: str           # nome usado no soccerdata Understat
    fbref_name: str | None = None # nome FBref (caso funcione no futuro)


LEAGUES: list[League] = [
    League(
        name="Premier League",
        country="Inglaterra",
        api_football_id=39,
        football_data_code="E0",
        sport_key="soccer_epl",
        understat_name="ENG-Premier League",
    ),
    League(
        name="La Liga",
        country="Espanha",
        api_football_id=140,
        football_data_code="SP1",
        sport_key="soccer_spain_la_liga",
        understat_name="ESP-La Liga",
    ),
    League(
        name="Bundesliga",
        country="Alemanha",
        api_football_id=78,
        football_data_code="D1",
        sport_key="soccer_germany_bundesliga",
        understat_name="GER-Bundesliga",
    ),
    League(
        name="Serie A",
        country="Italia",
        api_football_id=135,
        football_data_code="I1",
        sport_key="soccer_italy_serie_a",
        understat_name="ITA-Serie A",
    ),
    League(
        name="Ligue 1",
        country="Franca",
        api_football_id=61,
        football_data_code="F1",
        sport_key="soccer_france_ligue_1",
        understat_name="FRA-Ligue 1",
    ),
    League(
        name="Brasileirao",
        country="Brasil",
        api_football_id=71,
        football_data_code="B1",
        sport_key="soccer_brazil_campeonato",
        understat_name=None,  # Understat nao cobre Brasileirao
    ),
]

LEAGUES_BY_NAME = {l.name: l for l in LEAGUES}
LEAGUES_BY_API_ID = {l.api_football_id: l for l in LEAGUES}


def get_league(name: str) -> League:
    if name in LEAGUES_BY_NAME:
        return LEAGUES_BY_NAME[name]
    for l in LEAGUES:
        if name.lower() in l.name.lower():
            return l
    raise ValueError(f"Liga nao encontrada: {name}")


def get_league_by_api_id(api_id: int) -> League | None:
    return LEAGUES_BY_API_ID.get(api_id)


# Mapeamento football-data-co.uk: codigo -> nome da liga
FD_CODE_TO_LEAGUE = {l.football_data_code: l.name for l in LEAGUES}
