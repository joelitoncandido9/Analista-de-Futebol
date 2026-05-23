"""Coletor de odds da The-Odds-API.

Usa 6-7 keys rotacionadas para buscar odds de mercados (h2h, btts, totals,
alternate_totals_corners, alternate_totals_cards) para as ligas monitoradas.

Custo: 1 regiao (uk) x N mercados = N creditos por liga por chamada.
Limite: ~500 creditos/mes por key, 6-7 keys = ~3000 creditos/mes.
"""
import time
from datetime import datetime

import requests
from loguru import logger

from .base_collector import BaseCollector
from config.leagues import LEAGUES
from config.settings import THE_ODDS_API_KEYS, THE_ODDS_API_URL


class TheOddsAPICollector(BaseCollector):
    """Coleta odds de mercado da The-Odds-API."""

    def __init__(self, regions: str = "eu,uk",
                 markets: str = "h2h,totals"):
        # Usamos 'eu,uk' para ter Pinnacle (h2h) + William Hill (totals 2.5/3.5).
        # h2h e totals sao os unicos mercados suportados para soccer.
        # btts, alternate_totals_corners, alternate_totals_cards retornam 422.
        super().__init__("TheOddsAPI", rate_per_min=10)
        self._key_index = 0
        self._valid_key: str | None = None
        self.regions = regions
        self.markets = markets

    def _next_key(self) -> str:
        """Rotaciona entre as keys disponiveis ou usa a cacheada."""
        if self._valid_key:
            return self._valid_key
        key = THE_ODDS_API_KEYS[self._key_index % len(THE_ODDS_API_KEYS)]
        self._key_index += 1
        return key

    def _call(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        """Chamada a The-Odds-API com cache de key valida e retry."""
        url = f"{THE_ODDS_API_URL}{endpoint}"

        # Se ja temos uma key valida cacheada, tentar ela primeiro
        keys_to_try = list(THE_ODDS_API_KEYS)
        if self._valid_key:
            keys_to_try.insert(0, self._valid_key)

        for attempt in range(min(3, len(keys_to_try))):
            key = keys_to_try[attempt] if not self._valid_key else self._valid_key
            req_params = dict(params or {})
            req_params["apiKey"] = key
            self._rate_limit()

            try:
                resp = requests.get(url, params=req_params, timeout=20)
                if resp.status_code == 401:
                    logger.warning(f"  Key invalida {key[:8]}..., tentando proxima...")
                    if key == self._valid_key:
                        self._valid_key = None  # Invalidar cache
                    continue
                if resp.status_code == 429:
                    logger.warning(f"  Rate limit key, tentando proxima...")
                    continue
                if resp.status_code == 422:
                    logger.warning(f"  Erro 422 para {endpoint}: {resp.text[:200]}")
                    return None
                if resp.status_code != 200:
                    logger.warning(f"  HTTP {resp.status_code} em {endpoint}")
                    continue

                # Cachear a key valida
                if not self._valid_key:
                    self._valid_key = key
                    logger.info(f"  Key cacheada: {key[:8]}...")
                return resp.json()

            except Exception as e:
                logger.warning(f"  Erro: {e}")

        return None

    def get_sports(self) -> list[dict]:
        """Lista todos os esportes disponiveis (custo 0)."""
        result = self._call("/sports")
        return result if isinstance(result, list) else []

    def get_odds_for_sport(self, sport_key: str) -> list[dict]:
        """Busca odds para uma liga.

        Args:
            sport_key: Chave do esporte/liga (ex: 'soccer_epl').

        Returns:
            Lista de fixtures com odds.
        """
        params = {
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        result = self._call(f"/sports/{sport_key}/odds", params)
        if result is None:
            return []
        if isinstance(result, dict) and result.get("error"):
            logger.warning(f"  Erro API para {sport_key}: {result.get('error')}")
            return []
        return result if isinstance(result, list) else []

    def collect_league_odds(self, league) -> list[dict]:
        """Coleta odds para uma liga e salva no banco.

        Args:
            league: League config object.

        Returns:
            Lista de fixtures com odds retornadas pela API.
        """
        self.log_start(f"odds para {league.name} ({league.sport_key})")

        fixtures = self.get_odds_for_sport(league.sport_key)
        if not fixtures:
            logger.info(f"  {league.name}: nenhuma fixture com odds")
            return []

        logger.info(f"  {league.name}: {len(fixtures)} fixtures com odds")

        # Salvar no banco
        odds_rows = []
        for fx in fixtures:
            match_id = f"odds_{fx.get('id', '')}"
            commence_time = fx.get("commence_time", "")

            for bm in fx.get("bookmakers", []):
                bookmaker = bm.get("title", bm.get("key", "unknown"))
                for market in bm.get("markets", []):
                    market_key = market.get("key", "unknown")
                    for outcome in market.get("outcomes", []):
                        odds_rows.append({
                            "match_id": match_id,
                            "bookmaker": bookmaker,
                            "market": market_key,
                            "selection": outcome.get("name", ""),
                            "odd_value": outcome.get("price", 0),
                            "timestamp": commence_time,
                        })

        if odds_rows:
            from database.queries import save_odds
            saved = save_odds(odds_rows)
            logger.info(f"  {league.name}: {saved} odds salvas ({len(fixtures)} fixtures)")
        else:
            logger.info(f"  {league.name}: nenhuma odd extraida")

        return fixtures

    def collect_all_odds(self) -> dict[str, int]:
        """Coleta odds para todas as ligas configuradas.

        Returns:
            Dict: league_name -> numero de fixtures com odds.
        """
        results = {}
        for league in LEAGUES:
            try:
                fixtures = self.collect_league_odds(league)
                results[league.name] = len(fixtures)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"  Erro coletando odds para {league.name}: {e}")
                results[league.name] = 0
        return results

    def get_formatted_odds_for_fixture(self, fixtures_with_odds: list[dict],
                                        home_team: str, away_team: str) -> dict:
        """Extrai odds de um fixture especifico no formato que ValueBetDetector espera.

        Returns:
            Dict com mercados: h2h, totals, btts, corners, cards
            Cada mercado eh um dict de {selecao: odd}.
        """
        from database.merge import match_teams

        for fx in fixtures_with_odds:
            api_home = fx.get("home_team", "")
            api_away = fx.get("away_team", "")
            if match_teams(api_home, home_team) and match_teams(api_away, away_team):
                return self._extract_market_odds(fx)

        return {}

    def _extract_market_odds(self, fx: dict) -> dict:
        """Extrai odds usando Pinnacle como referencia principal.

        - h2h: usa Pinnacle (regiao eu)
        - totals: tenta Pinnacle primeiro; se a linha nao casar com as
          linhas do modelo (2.5, 3.5), busca em bookmakers UK (William Hill, etc.)
          que usam linhas padrao.
        Outros mercados (btts, corners, cards) nao estao disponiveis na API.
        """
        result: dict[str, dict[str, float]] = {}
        bookmakers = fx.get("bookmakers", [])
        if not bookmakers:
            return result

        def _find_bm(name_filter: str = "") -> dict | None:
            for bm in bookmakers:
                if name_filter and name_filter not in (bm.get("title") or "").lower():
                    continue
                return bm
            return None

        def _get_odds(bm: dict, market_key: str, name_transform=None) -> dict[str, float]:
            odds: dict[str, float] = {}
            for market in bm.get("markets", []):
                if market.get("key") != market_key:
                    continue
                for o in market.get("outcomes", []):
                    key = name_transform(o) if name_transform else o["name"]
                    odds[key] = float(o["price"])
            return odds

        def _transform_total(o: dict) -> str:
            return f"{o['name'].lower()}_{o.get('point', 0)}"

        # h2h: sempre usar Pinnacle
        pinnacle = _find_bm("pinnacle")
        if pinnacle:
            h2h = _get_odds(pinnacle, "h2h")
            if h2h:
                result["h2h"] = h2h

        # totals: Pinnacle primeiro (linhas diferentes), depois busca linhas 2.5/3.5
        if pinnacle:
            totals = _get_odds(pinnacle, "totals", _transform_total)
            if totals:
                result["totals"] = totals  # fallback caso nao ache 2.5/3.5

        # Buscar bookmaker com linhas padrao (2.5, 3.5) compativeis com o modelo
        for bm in bookmakers:
            if bm is pinnacle:
                continue
            totals = _get_odds(bm, "totals", _transform_total)
            if totals:
                lines = {float(k.split("_")[1]) for k in totals if "_" in k}
                if any(ln in (2.5, 3.5) for ln in lines):
                    result["totals"] = totals
                    break

        return result
