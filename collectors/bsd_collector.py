"""Coletor de dados da BSD (Bzzoiro Sports Data) API v2.

Usa API REST para buscar eventos, estatisticas, odds e predicoes
das ligas monitoradas. Substitui gradualmente a API-Football.
"""
from datetime import datetime, timedelta

from loguru import logger

from .base_collector import BaseCollector
from config.leagues import LEAGUES_BY_BSD_ID
from config.settings import BSD_TOKEN, BSD_BASE_URL


class BSDCollector(BaseCollector):
    """Coleta dados da BSD API v2: eventos, stats, odds, predicoes."""

    def __init__(self):
        super().__init__("BSD", rate_per_min=60)
        self.base_url = BSD_BASE_URL
        self.headers = {
            "Authorization": f"Token {BSD_TOKEN}",
            "Accept": "application/json",
        }
        if not BSD_TOKEN:
            logger.error("BSD_TOKEN nao configurado no .env")

    def _call(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Chamada a BSD API com rate limit e retry."""
        url = f"{self.base_url}{endpoint}"
        resp = self._request(url, headers=self.headers, params=params, retries=3, timeout=30)
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception as e:
            logger.error(f"Erro parseando JSON de {url}: {e}")
            return None

    def _fetch_all_paginated(self, endpoint: str, params: dict | None = None) -> list[dict]:
        """Percorre todas as paginas de um endpoint paginado."""
        all_results = []
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        while True:
            resp = self._request(url, headers=self.headers, params=params, retries=3, timeout=30)
            if resp is None:
                break
            try:
                data = resp.json()
            except Exception as e:
                logger.error(f"Erro parseando JSON de {url}: {e}")
                break

            results = data.get("results", [])
            all_results.extend(results)

            next_url = data.get("next")
            if not next_url:
                break
            # next_url ja vem com offset incrementado, usar direto
            url = next_url
            params = {}  # params ja estao na URL

        return all_results

    # ------------------------------------------------------------------
    # Metodos da API
    # ------------------------------------------------------------------

    def get_events(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        league_ids: list[int] | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Busca eventos com suporte a paginacao."""
        params = {}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if league_ids:
            params["league_id"] = ",".join(str(i) for i in league_ids)
        if status:
            params["status"] = status

        return self._fetch_all_paginated("/events/", params)

    def get_event_detail(self, event_id: int) -> dict | None:
        """Busca detalhes de um evento."""
        result = self._call(f"/events/{event_id}/")
        return result

    def get_event_stats(self, event_id: int) -> dict | None:
        """Busca estatisticas de um evento."""
        result = self._call(f"/events/{event_id}/stats/")
        if result:
            return result.get("stats")
        return None

    def get_event_odds(self, event_id: int) -> dict | None:
        """Busca odds consensuais de um evento."""
        result = self._call(f"/events/{event_id}/odds/")
        if result:
            return result.get("odds")
        return None

    def get_incidents(self, event_id: int) -> list[dict]:
        """Busca incidentes de um evento (gols, cartoes, substituicoes)."""
        result = self._call(f"/events/{event_id}/incidents/")
        if result:
            return result.get("incidents", [])
        return []

    def get_odds_best(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        league_ids: list[int] | None = None,
    ) -> list[dict]:
        """Busca melhores odds disponiveis (paginado)."""
        params = {}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if league_ids:
            params["league_id"] = ",".join(str(i) for i in league_ids)

        return self._fetch_all_paginated("/odds/best/", params)

    def get_predictions(
        self,
        league_id: int | None = None,
        event_id: int | None = None,
    ) -> list[dict]:
        """Busca predicoes ML do BSD (paginado)."""
        params = {}
        if league_id:
            params["league_id"] = league_id
        if event_id:
            params["event_id"] = event_id

        return self._fetch_all_paginated("/predictions/", params)

    # ------------------------------------------------------------------
    # Conversao de dados BSD -> schema do banco
    # ------------------------------------------------------------------

    BSD_STATUS_MAP = {
        "notstarted": "scheduled",
        "inprogress": "in_progress",
        "finished": "finished",
        "postponed": "postponed",
        "canceled": "cancelled",
    }

    STAT_KEY_MAP = {
        "total_shots": "shots",
        "shots_on_target": "shots_on_target",
        "ball_possession": "possession",
        "corner_kicks": "corners",
        "fouls": "fouls",
        "yellow_cards": "yellow",
        "red_cards": "red",
    }

    def _event_to_match(self, event: dict) -> dict | None:
        """Converte um evento BSD no formato matches do banco."""
        league_id = event.get("league_id")
        league = LEAGUES_BY_BSD_ID.get(league_id)
        if not league:
            return None

        status_raw = event.get("status", "")
        status = self.BSD_STATUS_MAP.get(status_raw, status_raw)

        match = {
            "match_id": f"bsd_{event['id']}",
            "league": league.name,
            "season": self._resolve_season(event),
            "round": event.get("round_number"),
            "match_date": (event.get("event_date") or "")[:10],
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "home_goals": event.get("home_score"),
            "away_goals": event.get("away_score"),
            "venue": None,
            "referee": None,
            "status": status,
            "source": "bsd",
        }
        return match

    def _resolve_season(self, event: dict) -> str:
        """Deriva temporada a partir do season_id."""
        return "2025/2026"

    def _enrich_with_stats(self, match: dict, stats: dict) -> dict:
        """Adiciona estatisticas detalhadas ao dict da partida."""
        if not stats:
            return match

        for side, prefix in [("home", "home"), ("away", "away")]:
            side_stats = stats.get(side, {})
            if not side_stats:
                continue

            for bsd_key, db_key in self.STAT_KEY_MAP.items():
                val = _get_stat_value(side_stats, bsd_key)
                if val is not None:
                    match[f"{prefix}_{db_key}"] = val

            # xg: pode vir como float direto ou {"actual": float}
            xg_val = _get_stat_value(side_stats, "expected_goals")
            if xg_val is not None:
                match[f"{prefix}_xg"] = xg_val

        return match

    def _odds_to_rows(self, match_id: str, odds: dict) -> list[dict]:
        """Converte odds consensuais BSD em linhas para tabela odds."""
        if not odds:
            return []

        now = datetime.now().isoformat()
        rows = []

        # Mapeamento chave BSD -> (mercado, selecao)
        mkt_map = {
            "home_win": ("1x2", "home"),
            "draw": ("1x2", "draw"),
            "away_win": ("1x2", "away"),
            "over_25_goals": ("over_under", "over_25"),
            "under_25_goals": ("over_under", "under_25"),
            "over_15_goals": ("over_under", "over_15"),
            "under_15_goals": ("over_under", "under_15"),
            "over_35_goals": ("over_under", "over_35"),
            "under_35_goals": ("over_under", "under_35"),
            "btts_yes": ("btts", "yes"),
            "btts_no": ("btts", "no"),
            "double_chance_1x": ("double_chance", "1x"),
            "double_chance_12": ("double_chance", "12"),
            "double_chance_x2": ("double_chance", "x2"),
        }

        for bsd_key, (market, selection) in mkt_map.items():
            value = odds.get(bsd_key)
            if value is not None:
                rows.append({
                    "match_id": match_id,
                    "bookmaker": "BSD_Consensus",
                    "market": market,
                    "selection": selection,
                    "odd_value": float(value),
                    "timestamp": now,
                })

        return rows

    # ------------------------------------------------------------------
    # Metodos de coleta principais
    # ------------------------------------------------------------------

    @property
    def _league_ids(self) -> list[int]:
        """Retorna IDs BSD das ligas monitoradas."""
        return [l.bsd_id for l in LEAGUES_BY_BSD_ID.values() if l.bsd_id]

    def collect_today(self, save: bool = True) -> list[dict]:
        """Coleta eventos de hoje + detalhes das partidas encerradas."""
        self.log_start("coleta do dia (BSD)")
        today = datetime.now().strftime("%Y-%m-%d")

        league_ids = self._league_ids
        if not league_ids:
            logger.warning("Nenhuma liga com bsd_id configurado")
            return []

        events = self.get_events(date_from=today, date_to=today, league_ids=league_ids)
        logger.info(f"  {len(events)} eventos encontrados (BSD)")

        if not events:
            return []

        matches = []
        all_odds_rows = []

        for event in events:
            match = self._event_to_match(event)
            if not match:
                continue

            event_id = event["id"]

            # Buscar odds (consenso BSD)
            odds = self.get_event_odds(event_id)
            if odds:
                all_odds_rows.extend(self._odds_to_rows(match["match_id"], odds))

            # Se encerrado, buscar stats detalhadas
            if event.get("status") == "finished":
                stats = self.get_event_stats(event_id)
                if stats:
                    match = self._enrich_with_stats(match, stats)
                else:
                    logger.debug(f"  Sem stats para evento {event_id}")

            matches.append(match)

        # Salvar no banco
        if save and matches:
            from database.queries import save_matches, save_odds

            saved = save_matches(matches)
            logger.info(f"  {saved} partidas salvas (BSD)")

            if all_odds_rows:
                saved_odds = save_odds(all_odds_rows)
                logger.info(f"  {saved_odds} odds salvas (BSD)")

        return matches

    def collect_upcoming(self, days: int = 7, save: bool = True) -> list[dict]:
        """Coleta eventos dos proximos dias."""
        self.log_start("coleta de proximos dias (BSD)")
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

        league_ids = self._league_ids
        if not league_ids:
            return []

        events = self.get_events(date_from=today, date_to=future, league_ids=league_ids)
        logger.info(f"  {len(events)} eventos futuros encontrados (BSD)")

        matches = []
        all_odds_rows = []

        for event in events:
            match = self._event_to_match(event)
            if not match:
                continue
            odds = self.get_event_odds(event["id"])
            if odds:
                all_odds_rows.extend(self._odds_to_rows(match["match_id"], odds))
            matches.append(match)

        if save and matches:
            from database.queries import save_matches, save_odds
            saved = save_matches(matches)
            logger.info(f"  {saved} partidas futuras salvas (BSD)")
            if all_odds_rows:
                save_odds(all_odds_rows)

        return matches

    def collect_results(self, days_back: int = 3, save: bool = True) -> list[dict]:
        """Coleta resultados de dias anteriores para atualizar stats."""
        self.log_start("coleta de resultados (BSD)")
        today = datetime.now().strftime("%Y-%m-%d")
        past = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        league_ids = self._league_ids
        if not league_ids:
            return []

        events = self.get_events(
            date_from=past, date_to=today,
            league_ids=league_ids, status="finished",
        )
        logger.info(f"  {len(events)} resultados encontrados (BSD)")

        results = []
        for event in events:
            match = self._event_to_match(event)
            if not match:
                continue
            stats = self.get_event_stats(event["id"])
            if stats:
                match = self._enrich_with_stats(match, stats)
            results.append(match)

        if save and results:
            from database.queries import save_matches
            saved = save_matches(results)
            logger.info(f"  {saved} resultados salvos (BSD)")

        return results

    def collect_predictions(self, league_id: int | None = None, save: bool = True) -> list[dict]:
        """Coleta predicoes ML do BSD."""
        self.log_start("coleta de predicoes (BSD)")

        league_ids = [league_id] if league_id else self._league_ids
        all_predictions = []

        for lid in league_ids:
            predictions = self.get_predictions(league_id=lid)
            all_predictions.extend(predictions)
            logger.info(f"  League {lid}: {len(predictions)} predicoes")

        if save and all_predictions:
            from database.queries import save_predictions
            # Converter formato BSD para schema do banco
            converted = []
            for p in all_predictions:
                ev = p.get("event", {})
                league_name = None
                lid = ev.get("league_id")
                if lid:
                    league_obj = LEAGUES_BY_BSD_ID.get(lid)
                    if league_obj:
                        league_name = league_obj.name
                if not league_name:
                    continue  # so salvar das ligas monitoradas

                m = p.get("markets", {})
                base = {
                    "fixture_id": str(ev.get("id")),
                    "home_team": ev.get("home_team", ""),
                    "away_team": ev.get("away_team", ""),
                    "league": league_name,
                    "match_date": ev.get("event_date", ""),
                    "source": "bsd",
                }

                # 1X2 → normalizar para "result" (mesmo formato dos nossos modelos)
                mr = m.get("match_result", {})
                converted.append({**base, "market": "result", "line": 0,
                                  "direction": "home", "probability": mr.get("prob_home", 0) / 100})
                converted.append({**base, "market": "result", "line": 0,
                                  "direction": "draw", "probability": mr.get("prob_draw", 0) / 100})
                converted.append({**base, "market": "result", "line": 0,
                                  "direction": "away", "probability": mr.get("prob_away", 0) / 100})

                # Expected Goals (mantido como mercado proprio)
                eg = m.get("expected_goals", {})
                eg_home = eg.get("home", 0)
                eg_away = eg.get("away", 0)
                converted.append({**base, "market": "expected_goals", "line": 0,
                                  "direction": "home", "probability": eg_home})
                converted.append({**base, "market": "expected_goals", "line": 0,
                                  "direction": "away", "probability": eg_away})

                # Over/Under → normalizar para "total_goals" (adicionando under = 1 - over)
                ou = m.get("over_under", {})
                for line, key in [(1.5, "prob_over_15"), (2.5, "prob_over_25"), (3.5, "prob_over_35")]:
                    prob_over = ou.get(key)
                    if prob_over is not None:
                        prob_over = prob_over / 100
                        converted.append({**base, "market": "total_goals", "line": line,
                                          "direction": "over", "probability": round(prob_over, 4)})
                        converted.append({**base, "market": "total_goals", "line": line,
                                          "direction": "under", "probability": round(1 - prob_over, 4)})

                # BTTS → normalizar para "btts" (sim = yes, nao = 1 - yes)
                btts = m.get("btts", {})
                prob_yes = btts.get("prob_yes")
                if prob_yes is not None:
                    prob_yes = prob_yes / 100
                    converted.append({**base, "market": "btts", "line": 0,
                                      "direction": "sim", "probability": round(prob_yes, 4)})
                    converted.append({**base, "market": "btts", "line": 0,
                                      "direction": "nao", "probability": round(1 - prob_yes, 4)})

            if converted:
                saved = save_predictions(converted)
                logger.info(f"  {saved} predicoes salvas (BSD)")
            else:
                logger.info("  Nenhuma predicao das ligas monitoradas")

        return all_predictions


def _get_stat_value(side_stats: dict, key: str):
    """Extrai valor de estatistica, lidando com formato aninhado."""
    val = side_stats.get(key)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("actual") or val.get("value")
    if isinstance(val, (int, float)):
        return val
    return None
