"""Preditor de escanteios por time usando modelos XGBoost treinados.

Usa dois modelos separados: um para escanteios do time da casa,
outro para escanteios do time visitante.
Gera probabilidades para over/under de cada time.
"""
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from models.features import load_matches, build_match_features
from models.team_corners.team_corners_trainer import (
    TeamCornersHomeTrainer,
    TeamCornersAwayTrainer,
)


class TeamCornersPredictor:
    """Preditor de escanteios por time para partidas."""

    def __init__(self, league: str | None = None):
        self.league = league
        self.home_trainer = TeamCornersHomeTrainer(league=league)
        self.away_trainer = TeamCornersAwayTrainer(league=league)
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._loaded = (
                self.home_trainer.load() and self.away_trainer.load()
            )

    def predict(self, home_team: str, away_team: str,
                league: str, match_date: str | None = None) -> dict | None:
        """Preve escanteios do time da casa e visitante."""
        self._ensure_loaded()
        if not self._loaded:
            logger.error("[TeamCorners] Modelos nao carregados")
            return None

        if not match_date:
            from datetime import datetime
            match_date = datetime.now().strftime("%Y-%m-%d")

        df = load_matches(league=league)
        if df.empty:
            logger.warning(f"[TeamCorners] Sem dados historicos para {league}")
            return None

        row = pd.Series({
            "match_date": pd.Timestamp(match_date),
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
        })

        feats = build_match_features(row, df)
        if feats["home_total_games"] < 2 or feats["away_total_games"] < 2:
            logger.warning(f"[TeamCorners] Times com pouco historico: "
                           f"{home_team}({feats['home_total_games']}j), "
                           f"{away_team}({feats['away_total_games']}j)")
            return None

        # --- Home corners ---
        X_home = pd.DataFrame([{name: feats.get(name, np.nan)
                                for name in self.home_trainer.feature_names}])
        X_home = X_home[self.home_trainer.feature_names]
        nan_cols = X_home.columns[X_home.isna().any()].tolist()
        if nan_cols:
            X_home = X_home.fillna(X_home.mean())

        pred_home = float(self.home_trainer.model.predict(X_home.values)[0])
        std_home = self.home_trainer.calib.get("residual_std", 1.5)

        home_lines = [4.5, 5.5, 6.5, 7.5]
        home_probs = {}
        for line in home_lines:
            p_over = 1.0 - stats.norm.cdf(line, loc=pred_home, scale=std_home)
            home_probs[f"over_{line}"] = round(p_over, 4)
            home_probs[f"under_{line}"] = round(1.0 - p_over, 4)

        # --- Away corners ---
        X_away = pd.DataFrame([{name: feats.get(name, np.nan)
                                for name in self.away_trainer.feature_names}])
        X_away = X_away[self.away_trainer.feature_names]
        nan_cols = X_away.columns[X_away.isna().any()].tolist()
        if nan_cols:
            X_away = X_away.fillna(X_away.mean())

        pred_away = float(self.away_trainer.model.predict(X_away.values)[0])
        std_away = self.away_trainer.calib.get("residual_std", 1.5)

        away_lines = [2.5, 3.5, 4.5, 5.5]
        away_probs = {}
        for line in away_lines:
            p_over = 1.0 - stats.norm.cdf(line, loc=pred_away, scale=std_away)
            away_probs[f"over_{line}"] = round(p_over, 4)
            away_probs[f"under_{line}"] = round(1.0 - p_over, 4)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "match_date": match_date,
            "predicted_home_corners": round(pred_home, 2),
            "predicted_away_corners": round(pred_away, 2),
            "home_probabilities": home_probs,
            "away_probabilities": away_probs,
            "home_calib_std": round(std_home, 2),
            "away_calib_std": round(std_away, 2),
        }

    def predict_many(self, fixtures: list[dict]) -> list[dict]:
        """Preve escanteios por time para multiplas partidas."""
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
                logger.warning(f"[TeamCorners] Erro prevendo {fx.get('home_team')}x{fx.get('away_team')}: {e}")
        return results
