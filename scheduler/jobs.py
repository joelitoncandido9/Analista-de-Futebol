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

        # 23:00 — Relatorio diario
        self.scheduler.add_job(
            self.job_daily_report,
            CronTrigger(hour=23, minute=0),
            id="daily_report",
            name="Daily report + RAG index",
            misfire_grace_time=3600,
        )

        logger.info("[Scheduler] Jobs registrados: pre_match(07:00), recheck(12:00), collect(19:30), report(23:00)")

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

            # Escanteios
            corners = pred.get("corners")
            if corners and corners.get("probabilities"):
                for key, prob in corners["probabilities"].items():
                    parts = key.split("_")
                    rows.append({**base,
                        "market": "total_corners",
                        "direction": parts[0],
                        "line": float(parts[1]),
                        "probability": round(prob, 4),
                        "predicted_value": round(corners["predicted_total_corners"], 1),
                    })

            # Finalizações
            shots = pred.get("shots")
            if shots and shots.get("probabilities"):
                for key, prob in shots["probabilities"].items():
                    parts = key.split("_")
                    rows.append({**base,
                        "market": "total_shots",
                        "direction": parts[0],
                        "line": float(parts[1]),
                        "probability": round(prob, 4),
                        "predicted_value": round(shots["predicted_total_shots"], 1),
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

            # Envia alerta pre-match individual para cada jogo
            from scheduler.alerts import alert_pre_match

            for pred in predictions:
                try:
                    alert_pre_match(
                        home=pred["home_team"],
                        away=pred["away_team"],
                        league=pred["league"],
                        corners_pred=pred.get("corners"),
                        result_pred=pred.get("result"),
                        value_bets=None,
                    )
                except Exception as e:
                    logger.warning(f"[Job] Erro alerta {pred['home_team']}x{pred['away_team']}: {e}")

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

                # Re-envia alertas individuais com dados atualizados
                from scheduler.alerts import alert_pre_match

                for pred in predictions:
                    try:
                        alert_pre_match(
                            home=pred["home_team"],
                            away=pred["away_team"],
                            league=pred["league"],
                            corners_pred=pred.get("corners"),
                            result_pred=pred.get("result"),
                            value_bets=None,
                        )
                    except Exception as e:
                        logger.warning(f"[Job] Erro re-check {pred['home_team']}x{pred['away_team']}: {e}")

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

            logger.info("[Job] Coleta de resultados concluida")

        except Exception as e:
            logger.error(f"[Job] Erro coleta resultados: {e}")

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

                dc = DixonColes(league=league)
                if dc.load():
                    result = dc.predict_score(home, away)
                else:
                    result = None

                predictions.append({
                    "fixture_id": fx.get("fixture_id"),
                    "home_team": home,
                    "away_team": away,
                    "league": league,
                    "match_date": date,
                    "corners": corners,
                    "shots": shots,
                    "result": result,
                })

            except Exception as e:
                logger.warning(f"[Job] Erro prevendo {home}x{away}: {e}")

        return predictions

    def _extract_high_confidence_tips(self, predictions: list[dict],
                                       min_conf: float = 0.75) -> list[dict]:
        """Extrai palpites de alta confiança (>75%) das previsões.

        Por partida, retorna no máximo 2 palpites (escanteios + finalizações),
        priorizando as linhas mais justas (mais próximas do valor previsto).
        """
        tips = []
        for pred in predictions:
            home = pred["home_team"]
            away = pred["away_team"]
            league = pred["league"]
            match = f"{home} x {away}"

            for market, data in [("escanteios", pred.get("corners")),
                                  ("finalizações", pred.get("shots"))]:
                if not data or not data.get("probabilities"):
                    continue

                predicted = data.get(f"predicted_total_{'corners' if market == 'escanteios' else 'shots'}", 0)

                # Encontrar a linha mais justa (>75%) em qualquer direção
                best_tip = None
                for key, prob in data["probabilities"].items():
                    if prob < min_conf:
                        continue
                    parts = key.split("_")
                    direction, line = parts[0], float(parts[1])

                    # Linha mais justa = menor diferença entre a linha e o previsto
                    tightness = abs(line - predicted)
                    if best_tip is None or tightness < best_tip["tightness"]:
                        best_tip = {
                            "match": match,
                            "league": league,
                            "market": market,
                            "direction": direction,
                            "line": line,
                            "confidence": round(prob * 100, 1),
                            "predicted": round(predicted, 1),
                            "tightness": tightness,
                        }

                if best_tip:
                    del best_tip["tightness"]
                    tips.append(best_tip)

        tips.sort(key=lambda t: -t["confidence"])
        return tips

    def _send_confidence_tips(self, tips: list[dict], prefix: str = ""):
        """Envia palpites de alta confiança via WhatsApp um por um."""
        if not tips:
            return

        from scheduler.whatsapp import send_text

        market_emoji = {"escanteios": "🥅", "finalizações": "💥"}

        for tip in tips:
            emoji = market_emoji.get(tip["market"], "📊")
            direction_label = "Under" if tip["direction"] == "under" else "Over"
            msg = (
                f"{emoji} *PALPITE {tip['market'].upper()}*\n"
                f"📋 {tip['match']} ({tip['league']})\n"
                f"🎯 {direction_label} {tip['line']} {tip['market']}\n"
                f"📈 Confiança: {tip['confidence']:.0f}%\n"
                f"📊 Previsto: {tip['predicted']} totais\n\n"
                f"#{tip['market']} #{direction_label}{tip['line']}"
            )
            try:
                ok = send_text(msg)
                logger.debug(f"[Tips] {'Enviado' if ok else 'Falha'}: {tip['match']} - {direction_label} {tip['line']}")
            except Exception as e:
                logger.warning(f"[Tips] Erro: {e}")

    def _find_value_bets(self, predictions: list[dict],
                          fixtures: list[dict]) -> list:
        """Encontra value bets nas previsoes."""
        from models.value_bet import ValueBetDetector

        detector = ValueBetDetector()
        all_bets = []

        for pred in predictions:
            corners = pred.get("corners")
            if corners and corners.get("probabilities"):
                # Sem odds de mercado por enquanto — usa odds hipoteticas
                # TODO: integrar com the-odds-api
                pass

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
