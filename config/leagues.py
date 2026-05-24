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
    fbref_name: str | None = None   # nome FBref (caso funcione no futuro)
    bsd_id: int | None = None       # ID na BSD (Bzzoiro Sports Data)


LEAGUES: list[League] = [
    League(
        name="Premier League",
        country="Inglaterra",
        api_football_id=39,
        football_data_code="E0",
        sport_key="soccer_epl",
        understat_name="ENG-Premier League",
        bsd_id=1,
    ),
    League(
        name="La Liga",
        country="Espanha",
        api_football_id=140,
        football_data_code="SP1",
        sport_key="soccer_spain_la_liga",
        understat_name="ESP-La Liga",
        bsd_id=3,
    ),
    League(
        name="Bundesliga",
        country="Alemanha",
        api_football_id=78,
        football_data_code="D1",
        sport_key="soccer_germany_bundesliga",
        understat_name="GER-Bundesliga",
        bsd_id=5,
    ),
    League(
        name="Serie A",
        country="Italia",
        api_football_id=135,
        football_data_code="I1",
        sport_key="soccer_italy_serie_a",
        understat_name="ITA-Serie A",
        bsd_id=4,
    ),
    League(
        name="Ligue 1",
        country="Franca",
        api_football_id=61,
        football_data_code="F1",
        sport_key="soccer_france_ligue_one",
        understat_name="FRA-Ligue 1",
        bsd_id=6,
    ),
    League(
        name="Brasileirao",
        country="Brasil",
        api_football_id=71,
        football_data_code="B1",
        sport_key="soccer_brazil_campeonato",
        understat_name=None,
        bsd_id=9,
    ),
    League(
        name="Championship",
        country="Inglaterra",
        api_football_id=40,
        football_data_code="E1",
        sport_key="soccer_efl_champ",
        understat_name=None,
        bsd_id=12,
    ),
    League(
        name="Primeira Liga",
        country="Portugal",
        api_football_id=94,
        football_data_code="P1",
        sport_key="soccer_portugal_primeira_liga",
        understat_name=None,
        bsd_id=2,
    ),
    League(
        name="Eredivisie",
        country="Holanda",
        api_football_id=88,
        football_data_code="N1",
        sport_key="soccer_netherlands_eredivisie",
        understat_name=None,
        bsd_id=10,
    ),
    League(
        name="2. Bundesliga",
        country="Alemanha",
        api_football_id=79,
        football_data_code="D2",
        sport_key="soccer_germany_bundesliga2",
        understat_name=None,
        bsd_id=None,
    ),
]

LEAGUES_BY_NAME = {l.name: l for l in LEAGUES}
LEAGUES_BY_API_ID = {l.api_football_id: l for l in LEAGUES}
LEAGUES_BY_BSD_ID = {l.bsd_id: l for l in LEAGUES if l.bsd_id}


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
