"""Coletor de odds da Odds-API.io para corners, bookings, shots.

Usa a API REST da Odds-API.io (api.odds-api.io/v3) para buscar odds
de mercados alternativos que a BSD nao cobre: escanteios, cartoes,
finalizacoes, etc.
"""
from datetime import datetime
from collections import defaultdict

from loguru import logger

from .base_collector import BaseCollector
from config.settings import ODDSAPI_IO_KEYS, ODDSAPI_IO_URL


class OddsApiIOCollector(BaseCollector):
    """Coleta odds de mercados via Odds-API.io (Unibet como fonte primaria)."""

    BOOKMAKER = "Unibet"

    # Rotacao de chaves: 3 × 100 req/h = 300 req/h disponiveis
    _key_idx = 0

    # Mercados alvo que a BSD nao cobre
    TARGET_MARKETS = {
        "Corners Totals": {"market": "corners", "subtype": "over_under"},
        "Corners Totals Home": {"market": "home_corners", "subtype": "over_under"},
        "Corners Totals Away": {"market": "away_corners", "subtype": "over_under"},
        "Bookings Totals": {"market": "yellow_cards", "subtype": "over_under"},
        "Total Shots": {"market": "total_shots", "subtype": "over_under"},
        "Total Shots on Target": {"market": "total_shots_on_target", "subtype": "over_under"},
    }

    def __init__(self):
        super().__init__("OddsAPI", rate_per_min=10)  # conservador: 10 req/min = 600/h
        self.base_url = ODDSAPI_IO_URL
        self._keys = ODDSAPI_IO_KEYS
        if not self._keys:
            logger.error("ODDSAPI_IO_KEY nao configurado")

    def _next_key(self) -> str:
        """Rotaciona entre as chaves disponiveis."""
        k = self._keys[OddsApiIOCollector._key_idx % len(self._keys)]
        OddsApiIOCollector._key_idx += 1
        return k

    def _call(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        """Chamada REST a Odds-API.io."""
        url = f"{self.base_url}{endpoint}"
        p = params or {}
        p["apiKey"] = self._next_key()
        resp = self._request(url, params=p, retries=3, timeout=20)
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception as e:
            logger.error(f"Erro parseando JSON de {url}: {e}")
            return None

    def get_leagues(self, sport: str = "football") -> list[dict]:
        """Lista ligas disponiveis."""
        data = self._call("/leagues", {"sport": sport})
        return data if isinstance(data, list) else []

    def get_events(
        self, league_slug: str | None = None, sport: str = "football"
    ) -> list[dict]:
        """Busca eventos pendentes de uma liga."""
        params = {"sport": sport, "status": "pending"}
        if league_slug:
            params["league"] = league_slug
        data = self._call("/events", params)
        return data if isinstance(data, list) else []

    def get_event_odds(self, event_id: int, bookmaker: str | None = None) -> dict | None:
        """Busca odds de um evento para um bookmaker especifico."""
        bm = bookmaker or self.BOOKMAKER
        data = self._call(f"/odds", {"eventId": event_id, "bookmakers": bm})
        if isinstance(data, dict) and "error" not in data:
            return data
        return None

    # ------------------------------------------------------------------
    # Extracao de mercados alvo
    # ------------------------------------------------------------------

    def extract_value_odds(self, event_id: int) -> list[dict]:
        """Extrai odds de corners, bookings, shots de um evento.

        Retorna lista de dicts com:
            event_id, home, away, league, market, line, direction, odd, source_bookmaker
        """
        odds_data = self.get_event_odds(event_id)
        if not odds_data:
            return []

        home = odds_data.get("home", "")
        away = odds_data.get("away", "")
        league = ""
        lg = odds_data.get("league", {})
        if isinstance(lg, dict):
            league = lg.get("name", "")

        results = []
        bm = odds_data.get("bookmakers", {})

        for bk_name, markets in bm.items():
            for m in markets:
                mkt_name = m.get("name", "")
                if mkt_name not in self.TARGET_MARKETS:
                    continue

                meta = self.TARGET_MARKETS[mkt_name]
                odds_list = m.get("odds", [])

                for odd_entry in odds_list:
                    hdp = odd_entry.get("hdp")
                    over_val = odd_entry.get("over", "N/A")
                    under_val = odd_entry.get("under", "N/A")

                    for direction, raw_odd in [("over", over_val), ("under", under_val)]:
                        if raw_odd in (None, "N/A", ""):
                            continue
                        try:
                            odd = float(raw_odd)
                        except (ValueError, TypeError):
                            continue

                        results.append({
                            "event_id": event_id,
                            "home_team": home,
                            "away_team": away,
                            "league": league,
                            "market": meta["market"],
                            "line": float(hdp) if hdp is not None else None,
                            "direction": direction,
                            "odd": odd,
                            "source_bookmaker": bk_name,
                            "market_label": mkt_name,
                        })
        return results

    # ------------------------------------------------------------------
    # Coleta principal
    # ------------------------------------------------------------------

    def collect_value_odds(
        self, league_slugs: list[str] | None = None
    ) -> list[dict]:
        """Coleta odds de valor para todas as ligas monitoradas."""
        self.log_start("coleta de odds (corners/bookings/shots)")

        if not league_slugs:
            # Usar ligas comuns
            league_slugs = [
                "brazil-brasileiro-serie-a", "brazil-brasileiro-serie-b",
                "international-clubs-copa-libertadores",
                "international-clubs-uefa-champions-league",
                "england-premier-league", "spain-la-liga",
                "germany-bundesliga", "italy-serie-a", "france-ligue-1",
                "portugal-primeira-liga",
            ]

        all_odds = []
        for slug in league_slugs:
            events = self.get_events(league_slug=slug)
            if not events:
                logger.debug(f"  Sem eventos para {slug}")
                continue

            logger.info(f"  {slug}: {len(events)} eventos")
            for ev in events:
                eid = ev.get("id")
                if not eid:
                    continue
                odds = self.extract_value_odds(eid)
                if odds:
                    all_odds.extend(odds)
                    logger.debug(
                        f"    {ev.get('home','?')} x {ev.get('away','?')}: "
                        f"{len(odds)} odds"
                    )

        logger.info(f"  Total: {len(all_odds)} odds coletadas")
        return all_odds


def get_league_stats(cur) -> dict:
    """Retorna medias historicas por liga para mercados alternativos."""
    cur.execute("""
        SELECT league,
               ROUND(AVG(home_corners + away_corners), 1) as avg_corners,
               ROUND(AVG(home_shots + away_shots), 1) as avg_shots,
               ROUND(AVG(home_shots_on_target + away_shots_on_target), 1) as avg_sot,
               ROUND(AVG(home_yellow + away_yellow), 1) as avg_yellow,
               ROUND(AVG(home_fouls + away_fouls), 1) as avg_fouls,
               COUNT(*) as n
        FROM matches
        WHERE source = 'bsd'
          AND home_corners IS NOT NULL
        GROUP BY league
    """)
    return {row["league"]: dict(row) for row in cur.fetchall()}
