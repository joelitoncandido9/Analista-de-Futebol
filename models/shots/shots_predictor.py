"""Preditor de finalizacoes usando modelo XGBoost treinado."""
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from models.features import load_matches, build_match_features
from models.shots.shots_trainer import ShotsTrainer


class ShotsPredictor:
    """Preditor de finalizacoes para partidas."""

    def __init__(self, league: str | None = None):
        self.league = league
        self.trainer = ShotsTrainer(league=league)
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._loaded = self.trainer.load()

    def predict(self, home_team: str, away_team: str,
                league: str, match_date: str | None = None) -> dict | None:
        """Preve total de finalizacoes para uma partida."""
        self._ensure_loaded()
        if not self._loaded or self.trainer.model is None:
            logger.error("[Shots] Modelo nao carregado")
            return None

        if not match_date:
            match_date = datetime.now().strftime("%Y-%m-%d")

        df = load_matches(league=league)
        if df.empty:
            return None

        row = pd.Series({
            "match_date": pd.Timestamp(match_date),
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
        })

        feats = build_match_features(row, df)
        if feats["home_total_games"] < 2 or feats["away_total_games"] < 2:
            return None

        X_dict = {name: feats.get(name, np.nan)
                  for name in self.trainer.feature_names}
        X_df = pd.DataFrame([X_dict])[self.trainer.feature_names]
        X_df = X_df.fillna(X_df.mean())

        pred = float(self.trainer.model.predict(X_df.values)[0])
        calib = self.trainer.calib
        residual_std = calib.get("residual_std", 3.0)

        lines = [20.5, 22.5, 24.5, 26.5, 28.5, 30.5]
        probs = {}
        for line in lines:
            p_over = 1.0 - stats.norm.cdf(line, loc=pred, scale=residual_std)
            probs[f"over_{line}"] = round(p_over, 4)
            probs[f"under_{line}"] = round(1.0 - p_over, 4)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "match_date": match_date,
            "predicted_total_shots": round(pred, 2),
            "predicted_home_shots": round(pred * 0.53, 2),
            "predicted_away_shots": round(pred * 0.47, 2),
            "probabilities": probs,
            "calib_std": round(residual_std, 2),
        }

    def predict_many(self, fixtures: list[dict]) -> list[dict]:
        """Preve finalizacoes para multiplas partidas."""
        results = []
        for fx in fixtures:
            try:
                pred = self.predict(
                    home_team=fx.get("home_team", ""),
                    away_team=fx.get("away_team", ""),
                    league=fx.get("league", ""),
                    match_date=fx.get("match_date"),
                )
                if pred:
                    pred["fixture_id"] = fx.get("fixture_id")
                    results.append(pred)
            except Exception as e:
                logger.warning(f"[Shots] Erro: {e}")
        return results
