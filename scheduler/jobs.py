"""Jobs agendados para coleta, predicao e geracao de relatorios.

Usa APScheduler para rodar tarefas diarias:
- 07:00 — Pre-match predictions para jogos do dia
- 12:00 — Re-check predictions (mercados podem mudar)
- 19:00 — Coleta resultados do dia + merge
- 23:00 — Relatorio diario + indexacao RAG
"""
from datetime import datetime, date
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config.settings import LOGS_DIR
from config.leagues import LEAGUES


class FootballScheduler:
    """Gerenciador de jobs agendados do Football AI."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self._setup_jobs()

    def _setup_jobs(self):
        """Registra todos os jobs recorrentes."""
        # 07:00 — Pre-match predictions
        self.scheduler.add_job(
            self.job_pre_match,
            CronTrigger(hour=7, minute=0),
            id="pre_match",
            name="Pre-match predictions",
            misfire_grace_time=600,
        )

        # 12:00 — Re-check
        self.scheduler.add_job(
            self.job_recheck,
            CronTrigger(hour=12, minute=0),
            id="recheck",
            name="Re-check predictions",
            misfire_grace_time=600,
        )

        # 19:30 — Coleta resultados do dia
        self.scheduler.add_job(
            self.job_collect_results,
            CronTrigger(hour=19, minute=30),
            id="collect_results",
            name="Collect today results",
            misfire_grace_time=1800,
        )

        # 22:55 — Backup para Backblaze B2
        self.scheduler.add_job(
            self.job_backup,
            CronTrigger(hour=22, minute=55),
            id="daily_backup",
            name="Daily backup to B2",
            misfire_grace_time=3600,
        )

        # 23:00 — Relatorio diario
        self.scheduler.add_job(
            self.job_daily_report,
            CronTrigger(hour=23, minute=0),
            id="daily_report",
            name="Daily report + RAG index",
            misfire_grace_time=3600,
        )

        logger.info("[Scheduler] Jobs registrados: pre_match(07:00), recheck(12:00), collect(19:30), backup(22:55), report(23:00)")

    # --- Job implementations ---

    def _save_all_predictions(self, predictions: list[dict]):
        """Salva todas as linhas de previsão no banco para comparação futura."""
        from database.queries import save_predictions

        rows = []
        for pred in predictions:
            base = {
                "fixture_id": pred.get("fixture_id", ""),
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred["league"],
                "match_date": str(pred.get("match_date", ""))[:10],
            }

            for market_key, market_name, predicted_key in [
                ("corners", "total_corners", "predicted_total_corners"),
                ("shots", "total_shots", "predicted_total_shots"),
                ("shots_on_target", "total_shots_on_target", "predicted_total_shots_on_target"),
                ("fouls", "total_fouls", "predicted_total_fouls"),
                ("cards", "total_yellow", "predicted_total_yellow"),
            ]:
                data = pred.get(market_key)
                if data and data.get("probabilities"):
                    for key, prob in data["probabilities"].items():
                        parts = key.split("_")
                        rows.append({**base,
                            "market": market_name,
                            "direction": parts[0],
                            "line": float(parts[1]),
                            "probability": round(prob, 4),
                            "predicted_value": round(data[predicted_key], 1),
                        })

            # Escanteios por time (home + away separados)
            tc = pred.get("team_corners")
            if tc:
                for side, market_name in [("home", "home_corners"), ("away", "away_corners")]:
                    probs = tc.get(f"{side}_probabilities")
                    pred_key = f"predicted_{side}_corners"
                    if probs:
                        for key, prob in probs.items():
                            parts = key.split("_")
                            rows.append({**base,
                                "market": market_name,
                                "direction": parts[0],
                                "line": float(parts[1]),
                                "probability": round(prob, 4),
                                "predicted_value": round(tc.get(pred_key, 0), 1),
                            })

            # Resultado (Dixon-Coles)
            result = pred.get("result")
            if result:
                for outcome, label in [("prob_home", "home"), ("prob_draw", "draw"), ("prob_away", "away")]:
                    rows.append({**base,
                        "market": "result",
                        "direction": label,
                        "line": 0,
                        "probability": round(result.get(outcome, 0), 4),
                        "predicted_value": result.get("most_likely_score", ""),
                    })

            # O/U Gols via Dixon-Coles
            goals = pred.get("goals")
            if goals and goals.get("probabilities"):
                for key, prob in goals["probabilities"].items():
                    parts = key.split("_")
                    rows.append({**base,
                        "market": "total_goals",
                        "direction": parts[0],
                        "line": float(parts[1]),
                        "probability": round(prob, 4),
                        "predicted_value": round(goals["predicted_total_goals"], 1),
                    })

            # BTTS via Dixon-Coles
            btts = pred.get("btts")
            if btts and btts.get("probabilities"):
                for key, prob in btts["probabilities"].items():
                    parts = key.split("_")
                    rows.append({**base,
                        "market": "btts",
                        "direction": parts[0],
                        "line": float(parts[1]),
                        "probability": round(prob, 4),
                        "predicted_value": round(btts.get("btts_prob", 0), 4),
                    })

            # Double Chance via Dixon-Coles
            dc = pred.get("double_chance")
            if dc and dc.get("probabilities"):
                for key, prob in dc["probabilities"].items():
                    parts = key.split("_")
                    rows.append({**base,
                        "market": "double_chance",
                        "direction": parts[0],
                        "line": float(parts[1]),
                        "probability": round(prob, 4),
                        "predicted_value": 0,
                    })

        saved = save_predictions(rows)
        logger.info(f"[Predictions] {saved} linhas salvas no banco")
        return saved

    def job_pre_match(self):
        """Gera pre-match predictions para jogos do dia."""
        logger.info("[Job] Iniciando pre-match predictions...")
        today = date.today().strftime("%Y-%m-%d")

        try:
            from collectors.api_football_collector import APIFootballCollector

            api = APIFootballCollector()
            fixtures = api.collect_today()
            if not fixtures:
                logger.info("[Job] Nenhum jogo hoje")
                return

            logger.info(f"[Job] {len(fixtures)} jogos hoje")

            # Predicoes para cada jogo
            predictions = self._predict_fixtures(fixtures)

            # Salva previsões no banco para comparação futura
            self._save_all_predictions(predictions)

            # Envia alerta pre-match consolidado por liga
            self._send_pre_match_consolidated(predictions)

            # Palpites de alta confianca (>75%)
            tips = self._extract_high_confidence_tips(predictions)
            if tips:
                self._send_confidence_tips(tips)

            # Value bets
            value_bets = self._find_value_bets(predictions, fixtures)
            if value_bets:
                self._send_value_bets_alert(value_bets)

            logger.info(f"[Job] Pre-match concluido: {len(predictions)} previsoes, {len(value_bets)} value bets, {len(tips)} palpites")

        except Exception as e:
            logger.error(f"[Job] Erro pre-match: {e}")

    def job_recheck(self):
        """Re-check predictions ao meio-dia com alertas individuais."""
        logger.info("[Job] Re-check meio-dia...")
        try:
            from collectors.api_football_collector import APIFootballCollector

            api = APIFootballCollector()
            today = date.today().strftime("%Y-%m-%d")
            fixtures = api.get_fixtures(today)
            if not fixtures:
                return

            # Filtrar so jogos que ainda nao comecaram
            upcoming = [f for f in fixtures if f.get("status") in ("NS", "TBD")]

            if upcoming:
                predictions = self._predict_fixtures(upcoming)

                # Salva previsões no banco
                self._save_all_predictions(predictions)

                # Re-envia alertas consolidados
                self._send_pre_match_consolidated(predictions, prefix="[Re-check] ")

                # Palpites de alta confiança atualizados
                tips = self._extract_high_confidence_tips(predictions)
                if tips:
                    self._send_confidence_tips(tips, prefix="[Re-check] ")

                value_bets = self._find_value_bets(predictions, upcoming)
                if value_bets:
                    self._send_value_bets_alert(value_bets, prefix="[Re-check] ")

            logger.info(f"[Job] Re-check concluido: {len(upcoming)} jogos pendentes")

        except Exception as e:
            logger.error(f"[Job] Erro re-check: {e}")

    def job_collect_results(self):
        """Coleta resultados do dia e executa merge."""
        logger.info("[Job] Coletando resultados do dia...")
        try:
            from collectors.api_football_collector import APIFootballCollector
            from database.merge import merge_all

            api = APIFootballCollector()
            api.collect_today()

            # Merge com Understat se disponivel
            from config.settings import DB_PATH
            logger.info("[Job] Executando merge de dados...")
            merge_all()

            # Avaliar previsões com os resultados reais
            from database.queries import evaluate_predictions
            result = evaluate_predictions()
            if result["evaluated"] > 0:
                logger.info(f"[Job] {result['evaluated']} previsões avaliadas")
                for s in result["stats"]:
                    logger.info(f"  {s['market']} {s['direction']} {s['line']}: "
                                f"{s['hits']}/{s['total']} ({s['accuracy']:.1%})")

            # Enviar resultado dos palpites (1 por mercado) no WhatsApp
            self._send_tip_results()

            logger.info("[Job] Coleta de resultados concluida")

        except Exception as e:
            logger.error(f"[Job] Erro coleta resultados: {e}")

    def job_backup(self):
        """Executa backup para Backblaze B2."""
        import subprocess

        logger.info("[Job] Iniciando backup...")
        try:
            result = subprocess.run(["/root/backup.sh"], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info(f"[Job] Backup concluido: {result.stdout.strip()}")
            else:
                logger.error(f"[Job] Erro backup: {result.stderr}")
        except Exception as e:
            logger.error(f"[Job] Erro backup: {e}")

    def job_daily_report(self):
        """Gera relatorio diario e indexa RAG."""
        logger.info("[Job] Gerando relatorio diario...")
        try:
            # Indexar partidas do dia no RAG
            from rag.indexer import index_matches

            # Indexar partidas recentes (ultima temporada)
            indexed = index_matches(limit=1000)
            logger.info(f"[Job] RAG indexado: {indexed} documentos")

            # Log de status
            from database.schema import get_conn

            conn = get_conn()
            cur = conn.cursor()
            cur.execute("SELECT league, COUNT(*) FROM matches GROUP BY league")
            rows = cur.fetchall()
            conn.close()

            logger.info("[Job] Status do banco:")
            for league, count in rows:
                logger.info(f"  {league}: {count} partidas")

            logger.info("[Job] Relatorio diario concluido")

        except Exception as e:
            logger.error(f"[Job] Erro relatorio diario: {e}")

    def _predict_fixtures(self, fixtures: list[dict]) -> list[dict]:
        """Gera previsoes para uma lista de fixtures."""
        predictions = []

        from models.corners.corners_predictor import CornersPredictor
        from models.shots.shots_predictor import ShotsPredictor
        from models.results.dixon_coles import DixonColes
        from models.results.dixon_coles_markets import (
            compute_btts, compute_over_under, compute_double_chance,
        )
        from models.shots_on_target.shots_on_target_predictor import ShotsOnTargetPredictor
        from models.fouls.fouls_predictor import FoulsPredictor
        from models.cards.cards_predictor import CardsPredictor
        from models.team_corners.team_corners_predictor import TeamCornersPredictor

        for fx in fixtures:
            home = fx.get("home_team", "")
            away = fx.get("away_team", "")
            league = fx.get("league", "")
            date = str(fx.get("match_date", ""))[:10]

            try:
                cp = CornersPredictor(league=league)
                corners = cp.predict(home, away, league, date)

                sp = ShotsPredictor(league=league)
                shots = sp.predict(home, away, league, date)

                sot = ShotsOnTargetPredictor(league=league)
                shots_on_target = sot.predict(home, away, league, date)

                fp = FoulsPredictor(league=league)
                fouls = fp.predict(home, away, league, date)

                cap = CardsPredictor(league=league)
                cards = cap.predict(home, away, league, date)

                tcp = TeamCornersPredictor(league=league)
                team_corners = tcp.predict(home, away, league, date)

                dc = DixonColes(league=league)
                if dc.load():
                    result = dc.predict_score(home, away)
                    # Only compute derived markets if model found both teams
                    if result.get("prob_home", 0) > 0 or result.get("prob_draw", 0) > 0 or result.get("prob_away", 0) > 0:
                        score_probs = result.get("score_probabilities", {})
                        goals = compute_over_under(score_probs)
                        btts = compute_btts(score_probs)
                        double_chance = compute_double_chance(score_probs)
                    else:
                        goals = btts = double_chance = None
                else:
                    result = None
                    goals = btts = double_chance = None

                predictions.append({
                    "fixture_id": fx.get("fixture_id"),
                    "home_team": home,
                    "away_team": away,
                    "league": league,
                    "match_date": date,
                    "corners": corners,
                    "shots": shots,
                    "shots_on_target": shots_on_target,
                    "fouls": fouls,
                    "cards": cards,
                    "team_corners": team_corners,
                    "result": result,
                    "goals": goals,
                    "btts": btts,
                    "double_chance": double_chance,
                })

            except Exception as e:
                logger.warning(f"[Job] Erro prevendo {home}x{away}: {e}")

        return predictions

    def _extract_high_confidence_tips(self, predictions: list[dict],
                                       min_conf: float = 0.75) -> list[dict]:
        """Extrai palpites de alta confiança (>75%) das previsões.

        Para cada mercado coberto, seleciona a linha mais justa (>75%),
        priorizando linhas mais próximas do valor previsto.
        """
        market_configs = [
            ("escanteios", "corners", "predicted_total_corners", None),
            ("finalizações", "shots", "predicted_total_shots", None),
            ("chutes no gol", "shots_on_target", "predicted_total_shots_on_target", None),
            ("faltas", "fouls", "predicted_total_fouls", None),
            ("cartões", "cards", "predicted_total_yellow", None),
            ("esc. casa", "team_corners", "predicted_home_corners", "home"),
            ("esc. fora", "team_corners", "predicted_away_corners", "away"),
            ("gols", "goals", "predicted_total_goals", None),
            ("BTTS", "btts", "btts_prob", None),
            ("Double Chance", "double_chance", "predicted_double_chance", None),
        ]

        tips = []
        for pred in predictions:
            home = pred["home_team"]
            away = pred["away_team"]
            league = pred["league"]
            match = f"{home} x {away}"

            for label, key, predicted_key, side in market_configs:
                data = pred.get(key)
                if not data:
                    continue

                probs_key = "probabilities"
                if side:
                    probs_key = f"{side}_probabilities"

                probs = data.get(probs_key)
                if not probs:
                    continue

                predicted = data.get(predicted_key, 0)

                best_tip = None
                for k, prob in probs.items():
                    if prob < min_conf:
                        continue
                    parts = k.split("_")
                    direction, line = parts[0], float(parts[1])

                    # tightness = how close the line is to the predicted value
                    # for non-numeric predictions (e.g., double chance), use 0
                    predicted_num = predicted if isinstance(predicted, (int, float)) else 0
                    tightness = abs(line - predicted_num)
                    if best_tip is None or tightness < best_tip["tightness"]:
                        best_tip = {
                            "match": match,
                            "league": league,
                            "market": label,
                            "direction": direction,
                            "line": line,
                            "confidence": round(prob * 100, 1),
                            "predicted": round(predicted_num, 1) if isinstance(predicted_num, float) else predicted,
                            "tightness": tightness,
                        }

                if best_tip:
                    del best_tip["tightness"]
                    tips.append(best_tip)

        tips.sort(key=lambda t: -t["confidence"])
        return tips

    def _send_pre_match_consolidated(self, predictions: list[dict], prefix: str = ""):
        """Envia alertas pre-match consolidados por liga."""
        if not predictions:
            return

        from scheduler.alerts import _send_both

        from collections import OrderedDict

        by_league: dict[str, list[dict]] = OrderedDict()
        for pred in predictions:
            league = pred["league"]
            if league not in by_league:
                by_league[league] = []
            by_league[league].append(pred)

        for league, preds in by_league.items():
            league_preds = [
                p for p in preds
                if p.get("result") and p["result"].get("prob_home", 0) > 0
            ]
            if not league_preds:
                continue

            lines = [f"{prefix}🏆 *PRÉVIA — {league}*\n"]
            for p in league_preds:
                home = p["home_team"]
                away = p["away_team"]
                result = p["result"]
                corners = p.get("corners")

                score = result.get("most_likely_score", "?")
                ph = f"{result['prob_home']:.0%}" if result.get("prob_home") else "?"
                pd = f"{result['prob_draw']:.0%}" if result.get("prob_draw") else "?"
                pa = f"{result['prob_away']:.0%}" if result.get("prob_away") else "?"

                line = f"\n📋 *{home} x {away}*"
                line += f"\n📊 {score} (Casa {ph} | Emp {pd} | Fora {pa})"

                if corners and corners.get("predicted_total_corners"):
                    tc = corners["predicted_total_corners"]
                    line += f"\n🥅 Escanteios: {tc} totais"

                lines.append(line)

            msg = "\n".join(lines)
            try:
                _send_both(msg)
                logger.info(f"[PreMatch] Enviado: {league} ({len(league_preds)} jogos)")
            except Exception as e:
                logger.warning(f"[PreMatch] Erro ao enviar {league}: {e}")

    def _send_confidence_tips(self, tips: list[dict], prefix: str = ""):
        """Envia palpites de alta confiança agrupados por jogo."""
        if not tips:
            return

        from scheduler.whatsapp import send_text

        market_emoji = {
            "escanteios": "🥅",
            "finalizações": "💥",
            "chutes no gol": "🎯",
            "faltas": "🟨",
            "cartões": "🟨",
            "esc. casa": "🏠",
            "esc. fora": "✈️",
            "gols": "⚽",
            "BTTS": "🤝",
            "Double Chance": "🛡️",
        }

        # Agrupar por jogo (mantendo ordenação por confiança)
        from collections import OrderedDict

        grouped: dict[str, dict] = OrderedDict()
        for tip in tips:
            key = tip["match"]
            if key not in grouped:
                grouped[key] = {"league": tip["league"], "tips": []}
            grouped[key]["tips"].append(tip)

        for match, data in grouped.items():
            league = data["league"]
            tip_list = data["tips"]
            header = f"{prefix}📋 *{match}* ({league})"
            lines = []

            for tip in tip_list:
                emoji = market_emoji.get(tip["market"], "📊")
                market = tip["market"]

                if market in ("BTTS", "Double Chance"):
                    direction_label = {"sim": "Sim", "nao": "Não",
                                       "casa-empate": "Casa ou Empate",
                                       "fora-empate": "Fora ou Empate"}.get(
                        tip["direction"], tip["direction"]
                    )
                    line_str = ""
                else:
                    direction_label = "Under" if tip["direction"] == "under" else "Over"
                    line_str = f" {tip['line']}"

                lines.append(
                    f"{emoji} {market}: {direction_label}{line_str} "
                    f"({tip['confidence']:.0f}%)"
                )

            msg = header + "\n" + "\n".join(lines)
            try:
                ok = send_text(msg)
                logger.debug(
                    f"[Tips] {'Enviado' if ok else 'Falha'}: {match} "
                    f"({len(tip_list)} palpites)"
                )
            except Exception as e:
                logger.warning(f"[Tips] Erro ao enviar {match}: {e}")

    def _find_value_bets(self, predictions: list[dict],
                          fixtures: list[dict]) -> list:
        """Encontra value bets comparando previsoes com odds reais da The-Odds-API.

        Para cada liga, busca odds via TheOddsAPICollector, casa por nome do time,
        e executa o ValueBetDetector para mercados disponiveis (h2h, totals).
        """
        from models.value_bet import ValueBetDetector
        from collectors.the_odds_api_collector import TheOddsAPICollector
        from config.leagues import LEAGUES_BY_NAME
        from database.merge import match_teams

        detector = ValueBetDetector()
        collector = TheOddsAPICollector()
        all_bets = []

        # Agrupar predictions por liga
        by_league: dict[str, list[dict]] = {}
        for pred in predictions:
            league_name = pred["league"]
            by_league.setdefault(league_name, []).append(pred)

        for league_name, league_preds in by_league.items():
            league_config = LEAGUES_BY_NAME.get(league_name)
            if not league_config:
                logger.warning(f"[ValueBet] Liga sem config: {league_name}")
                continue

            # Buscar odds reais para a liga
            odds_data = collector.get_odds_for_sport(league_config.sport_key)
            if not odds_data:
                logger.info(f"[ValueBet] Sem odds disponiveis para {league_name}")
                continue

            logger.info(f"[ValueBet] {league_name}: {len(odds_data)} fixtures com odds")

            for pred in league_preds:
                home = pred["home_team"]
                away = pred["away_team"]

                # Encontrar fixture correspondente nas odds por nome do time
                matched_fx = None
                for fx in odds_data:
                    api_home = fx.get("home_team", "")
                    api_away = fx.get("away_team", "")
                    if match_teams(api_home, home) and match_teams(api_away, away):
                        matched_fx = fx
                        break

                if not matched_fx:
                    continue

                # Extrair odds da fixture
                market_odds = collector._extract_market_odds(matched_fx)

                # Match Result (h2h) — usar nomes da API como chaves
                h2h = market_odds.get("h2h", {})
                if h2h:
                    api_home_team = matched_fx.get("home_team", "")
                    api_away_team = matched_fx.get("away_team", "")
                    odds_home = h2h.get(api_home_team, 0)
                    odds_draw = h2h.get("Draw", 0)
                    odds_away = h2h.get(api_away_team, 0)

                    if odds_home > 0 and odds_away > 0:
                        bets = detector.check_match_result(
                            pred, odds_home, odds_draw, odds_away
                        )
                        all_bets.extend(bets)

                # Over/Under Goals (totals)
                totals = market_odds.get("totals", {})
                if totals:
                    bets = detector.check_over_under_goals(pred, totals)
                    all_bets.extend(bets)

        all_bets.sort(key=lambda b: -b.edge)
        logger.info(f"[ValueBet] Total encontradas: {len(all_bets)}")
        return all_bets

    def _send_value_bets_alert(self, value_bets: list, prefix: str = ""):
        """Envia alerta de value bets (placeholder)."""
        if not value_bets:
            return
        try:
            from scheduler.alerts import send_telegram
            msg = f"{prefix}Value Bets encontradas: {len(value_bets)}"
            send_telegram(msg)
        except Exception as e:
            logger.warning(f"[Job] Erro ao enviar alerta: {e}")

    def _send_tip_results(self):
        """Envia resultado dos palpites (1 por mercado por jogo) no WhatsApp."""
        try:
            from database.schema import get_conn
            from scheduler.whatsapp import send_text
            from collections import OrderedDict

            conn = get_conn()
            cur = conn.cursor()
            cur.execute("""
                SELECT p.fixture_id, p.home_team, p.away_team, p.market,
                       p.direction, p.line, p.probability, p.predicted_value,
                       p.actual_value, p.was_correct,
                       m.home_goals, m.away_goals, m.league
                FROM predictions p
                JOIN matches m ON 'api_' || p.fixture_id = m.match_id
                WHERE p.probability >= 0.75 AND p.was_correct IS NOT NULL
                ORDER BY p.fixture_id, p.market, p.line
            """)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()

            if not rows:
                return

            # 1 per (fixture, market) — closest line to predicted_value
            market_map = {}
            for r in rows:
                key = (r["fixture_id"], r["market"])
                tightness = abs(r["line"] - (r["predicted_value"] or 0))
                if key not in market_map or tightness < market_map[key]["tightness"]:
                    market_map[key] = {**r, "tightness": tightness}

            tips = list(market_map.values())

            market_emoji = {
                "total_corners": "🥅", "total_shots": "💥",
                "total_shots_on_target": "🎯", "total_fouls": "🟨",
                "total_yellow": "🟨", "home_corners": "🏠",
                "away_corners": "✈️", "total_goals": "⚽",
                "btts": "🤝", "double_chance": "🛡️",
            }
            market_label = {
                "total_corners": "Escanteios", "total_shots": "Finalizações",
                "total_shots_on_target": "Ch. gol", "total_fouls": "Faltas",
                "total_yellow": "Amarelos", "home_corners": "Esc. casa",
                "away_corners": "Esc. fora", "total_goals": "Gols",
                "btts": "BTTS", "double_chance": "DC",
            }

            # Group by match
            grouped = OrderedDict()
            for t in tips:
                key = f"{t['home_team']} x {t['away_team']}"
                if key not in grouped:
                    grouped[key] = {"tips": []}
                grouped[key]["tips"].append(t)

            for match, data in grouped.items():
                hits = sum(1 for t in data["tips"] if t["was_correct"])
                total = len(data["tips"])
                pct = hits / total * 100
                lines = [f"📋 *{match}* — *{hits}/{total} ({pct:.0f}%)*\n"]

                for t in data["tips"]:
                    status = "✅" if t["was_correct"] else "❌"
                    em = market_emoji.get(t["market"], "📊")
                    ml = market_label.get(t["market"], t["market"])
                    d = t["direction"]
                    if d == "over": dl = "Over"
                    elif d == "under": dl = "Under"
                    elif d in ("sim", "nao"): dl = "Sim" if d == "sim" else "Não"
                    elif d == "casa-empate": dl = "Casa/Emp"
                    elif d == "fora-empate": dl = "Fora/Emp"
                    else: dl = d
                    conf = f"{t['probability']:.0%}"
                    lines.append(f"{status} {em} {ml}: {dl} {t['line']} (real: {t['actual_value']}) {conf}")

                msg = "\n".join(lines)
                send_text(msg)
                logger.info(f"[TipResult] Enviado: {match} ({hits}/{total})")

        except Exception as e:
            logger.warning(f"[TipResult] Erro ao enviar resultados: {e}")

    def start(self):
        """Inicia o scheduler."""
        self.scheduler.start()
        logger.info("[Scheduler] Iniciado. Jobs agendados rodando em background.")
        jobs = self.scheduler.get_jobs()
        for job in jobs:
            logger.info(f"  {job.name} | proxima exec: {job.next_run_time}")

    def stop(self):
        """Para o scheduler."""
        self.scheduler.shutdown(wait=False)
        logger.info("[Scheduler] Parado.")
