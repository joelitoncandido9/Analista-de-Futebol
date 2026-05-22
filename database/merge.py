"""Merge de dados de diferentes fontes no banco."""
import re
from loguru import logger
from database.schema import get_conn


def _normalize(name: str) -> str:
    """Normaliza nome de time para matching entre fontes."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ']", "", name)
    # Abreviacoes comuns
    replacements = {
        "manchester united": "man united",
        "manchester city": "man city",
        "man utd": "man united",
        "man city": "man city",
        "wolverhampton": "wolves",
        "wolverhampton wanderers": "wolves",
        "leicester city": "leicester",
        "tottenham": "spurs",
        "tottenham hotspur": "spurs",
        "newcastle utd": "newcastle",
        "newcastle united": "newcastle",
        "brighton and hove albion": "brighton",
        "nottingham forest": "nott'm forest",
        "west ham united": "west ham",
        "west bromwich albion": "west brom",
        "huddersfield town": "huddersfield",
        "swansea city": "swansea",
        "stoke city": "stoke",
        "cardiff city": "cardiff",
        "norwich city": "norwich",
        "leeds united": "leeds",
        "sheffield united": "sheffield utd",
        "sheffield wednesday": "sheffield wed",
        "hull city": "hull",
        "derby county": "derby",
        "birmingham city": "birmingham",
        "bolton wanderers": "bolton",
        "reading fc": "reading",
        "blackburn rovers": "blackburn",
        "queens park rangers": "qpr",
        "rotherham united": "rotherham",
        "middlesbrough fc": "middlesbrough",
        "borussia mönchengladbach": "borussia mg",
        "borussia monchengladbach": "borussia mg",
        "borussia dortmund": "borussia dortmund",
        "bayer leverkusen": "bayer leverkusen",
        "bayer 04 leverkusen": "bayer leverkusen",
        "1. fc köln": "koln",
        "fc köln": "koln",
        "eintracht frankfurt": "eintracht",
        "rb leipzig": "leipzig",
        "red bull leipzig": "leipzig",
        "fc bayern münchen": "bayern munich",
        "bayern münchen": "bayern munich",
        "fc bayern munich": "bayern munich",
        "vfb stuttgart": "stuttgart",
        "vfl wolfsburg": "wolfsburg",
        "vfl bochum": "bochum",
        "1. fc heidenheim": "heidenheim",
        "1. fc union berlin": "union berlin",
        "fc union berlin": "union berlin",
        "1. fc kaiserslautern": "kaiserslautern",
        "sc freiburg": "freiburg",
        "sv darmstadt 98": "darmstadt",
        "fc augsburg": "augsburg",
        "1. fsv mainz 05": "mainz",
        "fsv mainz 05": "mainz",
        "werder bremen": "werder bremen",
        "tsg 1899 hoffenheim": "hoffenheim",
        "atlético madrid": "atletico madrid",
        "atletico de madrid": "atletico madrid",
        "real betis": "betis",
        "real betis balompié": "betis",
        "real betis balompie": "betis",
        "athletic club": "athletic bilbao",
        "athletic de bilbao": "athletic bilbao",
        "valencia cf": "valencia",
        "deportivo alavés": "alaves",
        "deportivo alaves": "alaves",
        "cd alavés": "alaves",
        "rc celta de vigo": "celta vigo",
        "celta de vigo": "celta vigo",
        "ca osasuna": "osasuna",
        "fc barcelona": "barcelona",
        "real madrid": "real madrid",
        "real madrid cf": "real madrid",
        "sevilla fc": "sevilla",
        "villareal cf": "villareal",
        "fc porto": "porto",
        "sl benfica": "benfica",
        "sporting cp": "sporting lisbon",
        "fc twente": "twente",
        "psv eindhoven": "psv",
        "ajax amsterdam": "ajax",
        "fc emmen": "emmen",
        "nec nijmegen": "nec",
        "fc groningen": "groningen",
        "fc utrecht": "utrecht",
        "go ahead eagles": "go ahead",
        "associacao atletica internacional": "internacional",
        "botafogo de futebol e regatas": "botafogo",
        "botafogo fr": "botafogo",
        "club de regatas vasco da gama": "vasco",
        "club de regatas do flamengo": "flamengo",
        "santos fc": "santos",
        "santos fc sp": "santos",
        "sao paulo fc": "sao paulo",
        "sao paulo": "sao paulo",
        "sport club corinthians paulista": "corinthians",
        "corinthians": "corinthians",
        "sport club internacional": "internacional",
        "cruzeiro ec": "cruzeiro",
        "cruzeiro": "cruzeiro",
        "gremio fb pa": "gremio",
        "gremio": "gremio",
        "fluminense fc": "fluminense",
        "fluminense": "fluminense",
        "ec bahia": "bahia",
        "esporte clube bahia": "bahia",
        "ec vitoria": "vitoria",
        "atletico mg": "atletico mineiro",
        "atletico mineiro": "atletico mineiro",
        "atletico pr": "athletico paranaense",
        "athletico paranaense": "athletico paranaense",
        "atletico go": "atletico goianiense",
        "atletico goianiense": "atletico goianiense",
        "cuiaba ec": "cuiaba",
        "cuiaba": "cuiaba",
        "fortaleza ec": "fortaleza",
        "fortaleza": "fortaleza",
        "rb bragantino": "bragantino",
        "red bull bragantino": "bragantino",
        "jr": "juniors",
        "junior": "juniors",
    }
    # Mapeamentos genericos (prefix matching) - roda ANTES do replacements dict
    prefix_rep = {
        # Bundesliga
        "bayer leverkusen": "leverkusen",
        "borussia dortmund": "dortmund",
        "borussia m.gladbach": "m\'gladbach",
        "borussia mönchengladbach": "m\'gladbach",
        "borussia monchengladbach": "m\'gladbach",
        "eintracht frankfurt": "ein frankfurt",
        "fc heidenheim": "heidenheim",
        "mainz 05": "mainz",
        "rasenballsport leipzig": "rb leipzig",
        "st. pauli": "st pauli",
        "vfb stuttgart": "stuttgart",
        "1. fc heidenheim": "heidenheim",
        "1. fc union berlin": "union berlin",
        "fc union berlin": "union berlin",
        # La Liga
        "athletic club": "ath bilbao",
        "atletico madrid": "ath madrid",
        "celta vigo": "celta",
        "espanyol": "espanol",
        "rayo vallecano": "vallecano",
        "real betis": "betis",
        "real sociedad": "sociedad",
        "real valladolid": "valladolid",
        # Premier League
        "manchester united": "man united",
        "manchester city": "man city",
        "wolverhampton wanderers": "wolves",
        "newcastle united": "newcastle",
        "nottingham forest": "nott'm forest",
        "brighton and hove albion": "brighton",
        "tottenham hotspur": "tottenham",
        "west ham united": "west ham",
        "leicester city": "leicester",
        # Ligue 1
        "paris saint germain": "psg",
        "paris st germain": "psg",
        "as monaco": "monaco",
        "olympique lyonnais": "lyon",
        "olympique de lyon": "lyon",
        "olympique marseille": "marseille",
        "olympique de marseille": "marseille",
        "as saint étienne": "st etienne",
        "as saint-etienne": "st etienne",
        "fc nantes": "nantes",
        "fc lorient": "lorient",
        "rc lens": "lens",
        "rc strasbourg": "strasbourg",
        "montpellier hsc": "montpellier",
        "fc metz": "metz",
        "fc toulouse": "toulouse",
        "stade brest 29": "brest",
        "stade de reims": "reims",
        "stade rennais": "rennes",
        "ac ajaccio": "ajaccio",
        "es metz": "metz",
        "fc nantes": "nantes",
        "le havre ac": "le havre",
        "angters sco": "angers",
        "ogc nice": "nice",
    }
    for old, new in prefix_rep.items():
        if old in name:
            name = name.replace(old, new)

    for old, new in sorted(replacements.items(), key=lambda x: -len(x[0])):
        if old in name:
            name = name.replace(old, new)
    return name.strip()


def match_teams(t1: str, t2: str) -> bool:
    """Verifica se dois nomes de time se referem ao mesmo clube."""
    n1 = _normalize(t1)
    n2 = _normalize(t2)
    if n1 == n2:
        return True
    if len(n1) >= 5 and len(n2) >= 5:
        if n1[:5] == n2[:5]:
            return True
    return False


def merge_understat_into_football_data():
    """Atualiza registros football_data com dados de xG/PPDA do Understat."""
    conn = get_conn()
    cur = conn.cursor()

    # Buscar registros Understat que tem xG
    cur.execute(
        """SELECT match_id, league, match_date, home_team, away_team,
                  home_xg, away_xg, home_ppda, away_ppda,
                  home_deep, away_deep
           FROM matches WHERE source = 'understat'
           AND home_xg IS NOT NULL"""
    )
    understat_rows = [dict(r) for r in cur.fetchall()]

    updated = 0
    for u in understat_rows:
        # Usar apenas a data (YYYY-MM-DD), sem horario
        date_key = u["match_date"][:10] if u["match_date"] else ""

        # Buscar football_data ou merged correspondente pela mesma data
        cur.execute(
            """SELECT match_id, home_team, away_team FROM matches
               WHERE source IN ('football_data', 'merged')
               AND league = ? AND match_date = ?
               ORDER BY match_date""",
            (u["league"], date_key),
        )
        fd_rows = [dict(r) for r in cur.fetchall()]

        for fd in fd_rows:
            if match_teams(u["home_team"], fd["home_team"]) and \
               match_teams(u["away_team"], fd["away_team"]):
                # Update: adicionar xG, PPDA, deep ao registro football_data
                cur.execute(
                    """UPDATE matches SET
                        home_xg = COALESCE(?, home_xg),
                        away_xg = COALESCE(?, away_xg),
                        home_ppda = COALESCE(?, home_ppda),
                        away_ppda = COALESCE(?, away_ppda),
                        home_deep = COALESCE(?, home_deep),
                        away_deep = COALESCE(?, away_deep),
                        source = 'merged'
                       WHERE match_id = ?""",
                    (
                        u.get("home_xg"), u.get("away_xg"),
                        u.get("home_ppda"), u.get("away_ppda"),
                        u.get("home_deep"), u.get("away_deep"),
                        fd["match_id"],
                    ),
                )
                updated += cur.rowcount
                # Remover o registro Understat duplicado
                cur.execute("DELETE FROM matches WHERE match_id = ?", (u["match_id"],))
                break

    conn.commit()
    conn.close()
    logger.info(f"Merge concluido: {updated} registros atualizados com xG/PPDA")
    return updated


def merge_all():
    """Executa todas as etapas de merge."""
    u = merge_understat_into_football_data()
    total = u
    logger.info(f"Merge finalizado: {total} registros atualizados")
    return total
