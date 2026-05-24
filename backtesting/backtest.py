"""Motor de backtesting: avalia predicoes vs resultados reais, calcula ROI e calibra thresholds."""
from datetime import datetime, timedelta
from collections import defaultdict

from loguru import logger

from database.schema import get_conn
from database.queries import get_match_odds


class BacktestEngine:
    """Analisa predicoes historicas, calcula acuracia, ROI e sugere calibracao."""

    CONFIDENCE_BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    BUCKET_LABELS = {
        (0.5, 0.6): "50-60%",
        (0.6, 0.7): "60-70%",
        (0.7, 0.8): "70-80%",
        (0.8, 0.9): "80-90%",
        (0.9, 1.0): "90-100%",
    }

    def __init__(self):
        self.conn = get_conn()
        self.cur = self.conn.cursor()

    def evaluate_pending(self) -> int:
        """Avalia predicoes pendentes contra resultados reais."""
        updated = 0

        # Statistical markets (corners, shots, etc.)
        self.cur.execute("""
            UPDATE predictions
            SET actual_value = (
                CASE
                    WHEN market = 'total_corners' THEN home_corners + away_corners
                    WHEN market = 'total_shots' THEN home_shots + away_shots
                    WHEN market = 'total_shots_on_target' THEN home_shots_on_target + away_shots_on_target
                    WHEN market = 'total_fouls' THEN home_fouls + away_fouls
                    WHEN market = 'total_yellow' THEN home_yellow + away_yellow
                    WHEN market = 'home_corners' THEN home_corners
                    WHEN market = 'away_corners' THEN away_corners
                END
            ),
            was_correct = (
                CASE
                    WHEN direction = 'over' AND (
                        CASE
                            WHEN market = 'total_corners' THEN home_corners + away_corners
                            WHEN market = 'total_shots' THEN home_shots + away_shots
                            WHEN market = 'total_shots_on_target' THEN home_shots_on_target + away_shots_on_target
                            WHEN market = 'total_fouls' THEN home_fouls + away_fouls
                            WHEN market = 'total_yellow' THEN home_yellow + away_yellow
                            WHEN market = 'home_corners' THEN home_corners
                            WHEN market = 'away_corners' THEN away_corners
                        END
                    ) > line THEN 1
                    WHEN direction = 'under' AND (
                        CASE
                            WHEN market = 'total_corners' THEN home_corners + away_corners
                            WHEN market = 'total_shots' THEN home_shots + away_shots
                            WHEN market = 'total_shots_on_target' THEN home_shots_on_target + away_shots_on_target
                            WHEN market = 'total_fouls' THEN home_fouls + away_fouls
                            WHEN market = 'total_yellow' THEN home_yellow + away_yellow
                            WHEN market = 'home_corners' THEN home_corners
                            WHEN market = 'away_corners' THEN away_corners
                        END
                    ) < line THEN 1
                    WHEN actual_value IS NOT NULL THEN 0
                END
            )
            FROM matches
            WHERE (matches.match_id = 'api_' || predictions.fixture_id
                   OR matches.match_id = 'bsd_' || predictions.fixture_id)
            AND predictions.actual_value IS NULL
            AND matches.home_corners IS NOT NULL
        """)
        updated += self.cur.rowcount

        # Total Goals (O/U via Dixon-Coles)
        self.cur.execute("""
            UPDATE predictions
            SET actual_value = (home_goals + away_goals),
                was_correct = CASE
                    WHEN direction = 'over' AND (home_goals + away_goals) > line THEN 1
                    WHEN direction = 'under' AND (home_goals + away_goals) < line THEN 1
                    ELSE 0
                END
            FROM matches
            WHERE (matches.match_id = 'api_' || predictions.fixture_id
                   OR matches.match_id = 'bsd_' || predictions.fixture_id)
            AND predictions.actual_value IS NULL
            AND predictions.market = 'total_goals'
            AND matches.home_goals IS NOT NULL
        """)
        updated += self.cur.rowcount

        # BTTS
        self.cur.execute("""
            UPDATE predictions
            SET actual_value = CASE WHEN home_goals > 0 AND away_goals > 0 THEN 1 ELSE 0 END,
                was_correct = CASE
                    WHEN direction = 'sim' AND home_goals > 0 AND away_goals > 0 THEN 1
                    WHEN direction = 'nao' AND (home_goals = 0 OR away_goals = 0) THEN 1
                    ELSE 0
                END
            FROM matches
            WHERE (matches.match_id = 'api_' || predictions.fixture_id
                   OR matches.match_id = 'bsd_' || predictions.fixture_id)
            AND predictions.actual_value IS NULL
            AND predictions.market = 'btts'
            AND matches.home_goals IS NOT NULL
        """)
        updated += self.cur.rowcount

        # Double Chance
        self.cur.execute("""
            UPDATE predictions
            SET actual_value = CASE
                    WHEN home_goals > away_goals THEN 1
                    WHEN home_goals = away_goals THEN 0
                    ELSE -1
                END,
                was_correct = CASE
                    WHEN direction = 'casa-empate' AND home_goals >= away_goals THEN 1
                    WHEN direction = 'fora-empate' AND away_goals >= home_goals THEN 1
                    ELSE 0
                END
            FROM matches
            WHERE (matches.match_id = 'api_' || predictions.fixture_id
                   OR matches.match_id = 'bsd_' || predictions.fixture_id)
            AND predictions.actual_value IS NULL
            AND predictions.market = 'double_chance'
            AND matches.home_goals IS NOT NULL
        """)
        updated += self.cur.rowcount

        self.conn.commit()
        if updated:
            logger.info(f"[Backtest] {updated} predicoes avaliadas")
        return updated

    def _bucket(self, prob: float) -> tuple:
        """Retorna o bucket de confianca para uma probabilidade."""
        for lo, hi in self.CONFIDENCE_BUCKETS:
            if lo <= prob < hi:
                return (lo, hi)
        return (0.9, 1.0)

    def summary_by_market(self) -> list[dict]:
        """Acuracia agregada por mercado e direcao."""
        self.cur.execute("""
            SELECT market, direction,
                   COUNT(*) as total,
                   SUM(was_correct) as hits,
                   ROUND(AVG(probability), 4) as avg_prob,
                   ROUND(AVG(was_correct), 4) as acc
            FROM predictions
            WHERE was_correct IS NOT NULL
              AND market NOT IN ('result', 'expected_goals')
            GROUP BY market, direction
            ORDER BY total DESC
        """)
        return [dict(r) for r in self.cur.fetchall()]

    def summary_by_confidence(self) -> list[dict]:
        """Acuracia agregada por bucket de confianca."""
        self.cur.execute("""
            SELECT market, probability, was_correct
            FROM predictions
            WHERE was_correct IS NOT NULL
              AND market NOT IN ('result', 'expected_goals')
        """)
        raw = [dict(r) for r in self.cur.fetchall()]

        buckets = defaultdict(lambda: {"total": 0, "hits": 0, "markets": set()})
        for r in raw:
            b = self._bucket(r["probability"])
            buckets[b]["total"] += 1
            buckets[b]["hits"] += r["was_correct"]
            buckets[b]["markets"].add(r["market"])

        results = []
        for b in sorted(buckets.keys()):
            d = buckets[b]
            results.append({
                "bucket": self.BUCKET_LABELS[b],
                "total": d["total"],
                "hits": d["hits"],
                "acc": round(d["hits"] / d["total"], 4) if d["total"] else 0,
                "markets": sorted(d["markets"]),
            })
        return results

    def summary_by_market_confidence(self) -> list[dict]:
        """Acuracia por mercado + bucket de confianca."""
        self.cur.execute("""
            SELECT market, probability, direction, was_correct
            FROM predictions
            WHERE was_correct IS NOT NULL
              AND market NOT IN ('result', 'expected_goals')
        """)
        raw = [dict(r) for r in self.cur.fetchall()]

        buckets = defaultdict(lambda: {"total": 0, "hits": 0})
        for r in raw:
            b = self._bucket(r["probability"])
            key = (r["market"], self.BUCKET_LABELS[b])
            buckets[key]["total"] += 1
            buckets[key]["hits"] += r["was_correct"]

        results = []
        for (market, label), d in sorted(buckets.items()):
            results.append({
                "market": market,
                "bucket": label,
                "total": d["total"],
                "hits": d["hits"],
                "acc": round(d["hits"] / d["total"], 4) if d["total"] else 0,
            })
        return results

    def roi_simulation(self) -> list[dict]:
        """ROI simulado para mercados com odds disponiveis.

        Para cada predicao avaliada, busca a odd correspondente no banco
        e calcula: ROI = (odds * acertos - total_apostas) / total_apostas
        """
        # Buscar predicoes + odds
        self.cur.execute("""
            SELECT p.market, p.direction, p.line, p.probability,
                   p.was_correct, p.fixture_id
            FROM predictions p
            WHERE p.was_correct IS NOT NULL
              AND p.market IN ('total_goals', 'btts', 'double_chance')
        """)
        preds = [dict(r) for r in self.cur.fetchall()]

        # Mapear mercado/direction/line -> odds selection
        mkt_map = {
            "total_goals": {"odds_market": "over_under", "line_scale": 10},
            "btts": {"odds_market": "btts", "dir_map": {"sim": "yes", "nao": "no"}},
            "double_chance": {"odds_market": "double_chance",
                              "dir_map": {"casa-empate": "1x", "fora-empate": "x2", "12": "12"}},
        }

        results = []
        for p in preds:
            config = mkt_map.get(p["market"])
            if not config:
                continue

            # Construir selection key
            if "dir_map" in config:
                sel = config["dir_map"].get(p["direction"])
            elif "line_scale" in config:
                line_int = int(p["line"] * config["line_scale"])
                sel = f"{p['direction']}_{line_int}"
            else:
                sel = p["direction"]

            # Buscar odd
            match_id = f"bsd_{p['fixture_id']}"
            odds_rows = get_match_odds(match_id, config["odds_market"])
            odd = None
            for o in odds_rows:
                if o["selection"] == sel:
                    odd = o["odd_value"]
                    break

            if odd is not None:
                results.append({
                    "market": p["market"],
                    "direction": p["direction"],
                    "line": p["line"],
                    "prob": p["probability"],
                    "odd": odd,
                    "was_correct": p["was_correct"],
                    "ev": round(p["probability"] * odd - 1, 4),
                })

        # Agregar ROI por mercado
        by_market = defaultdict(lambda: {"bets": 0, "hits": 0, "total_staked": 0.0, "total_returned": 0.0})
        for r in results:
            m = r["market"]
            by_market[m]["bets"] += 1
            by_market[m]["hits"] += r["was_correct"]
            by_market[m]["total_staked"] += 1.0  # 1u flat
            by_market[m]["total_returned"] += r["odd"] if r["was_correct"] else 0

        roi_results = []
        for market, d in sorted(by_market.items()):
            roi = (d["total_returned"] - d["total_staked"]) / d["total_staked"] if d["total_staked"] else 0
            roi_results.append({
                "market": market,
                "bets": d["bets"],
                "hits": d["hits"],
                "acc": round(d["hits"] / d["bets"], 4) if d["bets"] else 0,
                "roi": round(roi, 4),
                "profit": round(d["total_returned"] - d["total_staked"], 2),
            })

        return roi_results

    def calibrate_thresholds(self) -> list[dict]:
        """Sugere thresholds de confianca minimos por mercado.

        Para cada mercado, encontra o menor nivel de confianca onde
        a acuracia real >= confianca (modelo calibrado).
        """
        self.cur.execute("""
            SELECT market, probability, was_correct
            FROM predictions
            WHERE was_correct IS NOT NULL
              AND market NOT IN ('result', 'expected_goals')
            ORDER BY market, probability DESC
        """)
        raw = [dict(r) for r in self.cur.fetchall()]

        by_market = defaultdict(list)
        for r in raw:
            by_market[r["market"]].append(r)

        suggestions = []
        for market, preds in sorted(by_market.items()):
            if len(preds) < 5:
                suggestions.append({"market": market, "n": len(preds), "suggested_threshold": None,
                                    "note": "poucos dados (min 5)"})
                continue

            total = len(preds)
            hits = sum(p["was_correct"] for p in preds)
            overall_acc = hits / total

            # Sugerir threshold baseado na acuracia geral
            suggested = max(0.5, overall_acc - 0.05)  # margem de 5%
            suggestions.append({
                "market": market,
                "n": total,
                "acc": round(overall_acc, 3),
                "suggested_threshold": round(suggested, 2),
                "note": "calibrado" if overall_acc >= 0.7 else "pouca amostra",
            })

        return suggestions

    def generate_report(self, include_roi: bool = True) -> str:
        """Gera relatorio completo de backtesting."""
        lines = []
        lines.append("📊 *RELATORIO DE BACKTESTING*")
        lines.append(f"_{datetime.now().strftime('%d/%m/%Y %H:%M')}_\n")

        # 1. Overview
        self.cur.execute("""
            SELECT COUNT(*) as total,
                   SUM(was_correct) as hits,
                   SUM(1 - was_correct) as misses
            FROM predictions WHERE was_correct IS NOT NULL
        """)
        r = dict(self.cur.fetchone())
        total = r["total"] or 0
        hits = r["hits"] or 0
        acc = f"{hits / total:.1%}" if total else "N/A"
        lines.append(f"*Geral:* {hits}/{total} acertos ({acc})\n")

        # 2. By confidence bucket
        lines.append("*Acuracia por Confianca:*")
        buckets = self.summary_by_confidence()
        for b in buckets:
            if b["total"] == 0:
                continue
            bar = "▓" * int(b["acc"] * 20) + "░" * (20 - int(b["acc"] * 20))
            lines.append(f"  {b['bucket']:8s}: {b['hits']:3d}/{b['total']:<3d} {bar} {b['acc']:.0%}")
        lines.append("")

        # 3. By market
        lines.append("*Acuracia por Mercado:*")
        markets = self.summary_by_market()
        for m in markets:
            bar = "▓" * int(m["acc"] * 20) + "░" * (20 - int(m["acc"] * 20))
            lines.append(f"  {m['market']:25s} {m['direction']:10s}: {m['hits']:3d}/{m['total']:<3d} {bar} {m['acc']:.0%} (prob media {m['avg_prob']:.0%})")
        lines.append("")

        # 4. ROI (if available)
        if include_roi:
            lines.append("*ROI Simulado (1u flat):*")
            roi_data = self.roi_simulation()
            if roi_data:
                for r in roi_data:
                    signal = "+" if r["profit"] >= 0 else ""
                    lines.append(f"  {r['market']:20s}: {r['bets']} bets, {r['hits']} hits, "
                                 f"ROI {signal}{r['roi']:.1%} (P&L {signal}{r['profit']:.2f}u)")
            else:
                lines.append("  Sem dados suficientes para ROI")
            lines.append("")

        # 5. Calibration suggestions
        lines.append("*Calibracao Sugerida:*")
        cal = self.calibrate_thresholds()
        for c in cal:
            th = f"{c['suggested_threshold']:.0%}" if c["suggested_threshold"] else "---"
            lines.append(f"  {c['market']:25s}: threshold {th} ({c['n']} amostras, {c['note']})")

        return "\n".join(lines)

    def close(self):
        self.conn.close()


def run_backtest(verbose: bool = True) -> str:
    """Executa backtesting completo e retorna relatorio."""
    engine = BacktestEngine()

    # 1. Evaluate pending
    updated = engine.evaluate_pending()
    if verbose:
        logger.info(f"[Backtest] {updated} novas predicoes avaliadas")

    # 2. Generate report
    report = engine.generate_report(include_roi=True)

    if verbose:
        logger.info(f"\n{report}")

    engine.close()
    return report
