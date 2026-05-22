"""Backtesting dos modelos: validacao temporal com metricas reais.

Simula apostas em jogos historicos para avaliar performance:
- ROI por linha de over/under
- Acuracia das probabilidades (calibracao)
- Value se tivessemos odds reais do mercado
"""
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from config.settings import MODELS_DIR, DB_PATH
from models.features import load_matches, build_match_features, temporal_split

warnings.filterwarnings("ignore")


class Backtester:
    """Backtesting temporal dos modelos de escanteios/finalizacoes."""

    def __init__(self, model_type: str = "corners", league: str | None = None):
        self.model_type = model_type
        self.league = league
        self.target_col = "total_corners" if model_type == "corners" else "total_shots"
        self.results: list[dict] = []
        self.metrics: dict = {}

    def run(self, test_games: int = 760, window: int = 3000, step: int = 20) -> dict:
        """Executa backtesting com janela deslizante otimizada.

        Pre-computa todas as features primeiro (uma unica passada),
        depois faz janela deslizante treinando modelo a cada `step` jogos.

        Args:
            test_games: Quantos jogos testar no final.
            window: Janela maxima de treino.
            step: Treina modelo a cada N jogos (batch), nao por jogo.

        Returns:
            Dict com metricas de performance.
        """
        df = load_matches(league=self.league)
        if df.empty:
            return {}

        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)
        if len(df) < 500:
            logger.warning(f"[Backtest] Dados insuficientes: {len(df)} partidas")
            return {}

        logger.info(f"[Backtest] Pre-computando features para {len(df)} partidas...")
        t0 = time.time()

        # Pre-computar features para todas as partidas
        all_feats = []
        all_targets = []
        for idx, row in df.iterrows():
            feats = build_match_features(row, df)
            all_feats.append(feats)
            all_targets.append(row[self.target_col])

        all_feats_df = pd.DataFrame(all_feats).fillna(0)
        all_targets = np.array(all_targets, dtype=float)
        feature_cols = list(all_feats_df.columns)
        logger.info(f"[Backtest] Features pre-computadas em {time.time()-t0:.1f}s "
                     f"({len(all_feats_df)} linhas, {len(feature_cols)} colunas)")

        start_idx = len(df) - test_games
        predictions = np.full(test_games, np.nan)
        actuals = np.full(test_games, np.nan)
        n_valid = 0

        from xgboost import XGBRegressor

        t0 = time.time()
        for batch_start in range(start_idx, len(df), step):
            batch_end = min(batch_start + step, len(df))

            # Dados de treino: ate o primeiro jogo deste batch
            train_end = batch_start
            train_start = max(0, train_end - window)

            X_train = all_feats_df.iloc[train_start:train_end]
            y_train = all_targets[train_start:train_end]

            if len(X_train) < 50:
                continue

            model = XGBRegressor(n_estimators=200, max_depth=4,
                                 learning_rate=0.1, verbosity=0)
            model.fit(X_train.values, y_train)

            # Prever todos os jogos deste batch
            for i in range(batch_start, batch_end):
                if i >= len(df):
                    break
                test_idx = i - start_idx
                X_test = all_feats_df.iloc[i:i+1]
                pred = float(model.predict(X_test.values)[0])
                predictions[test_idx] = pred
                actuals[test_idx] = float(all_targets[i])
                n_valid += 1

            elapsed = time.time() - t0
            games_done = min(batch_end, len(df)) - start_idx
            pct = games_done / test_games * 100
            rate = games_done / elapsed if elapsed > 0 else 0
            remaining = (test_games - games_done) / rate if rate > 0 else 0
            logger.info(f"[Backtest] {games_done}/{test_games} ({pct:.0f}%) | "
                         f"{rate:.1f} jogos/s | ETA: {remaining:.0f}s")

        # Remover NaNs (partidas que pularam)
        valid_mask = ~np.isnan(predictions)
        if valid_mask.sum() < 50:
            logger.warning("[Backtest] Muitas previsoes invalidas")
            return {}

        preds = predictions[valid_mask]
        acts = actuals[valid_mask]

        mae = float(np.mean(np.abs(preds - acts)))
        rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))
        bias = float(np.mean(preds - acts))

        # Acuracia em diferentes linhas
        lines = [8.5, 9.5, 10.5, 11.5, 12.5] if self.model_type == "corners" else [20.5, 22.5, 24.5, 26.5, 28.5]
        line_results = {}
        for line in lines:
            pred_over = (preds > line).mean()
            actual_over = (acts > line).mean()
            line_results[f"over_{line}"] = {
                "predicted_rate": round(float(pred_over), 4),
                "actual_rate": round(float(actual_over), 4),
                "error": round(float(pred_over - actual_over), 4),
            }

        self.metrics = {
            "model_type": self.model_type,
            "league": self.league or "all",
            "n_tests": int(valid_mask.sum()),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "bias": round(bias, 3),
            "mean_actual": round(float(acts.mean()), 2),
            "mean_predicted": round(float(preds.mean()), 2),
            "line_accuracy": line_results,
        }

        logger.info(f"[Backtest] {self.model_type.upper()} | "
                     f"MAE: {mae:.2f} | Bias: {bias:.2f} | "
                     f"N: {int(valid_mask.sum())}")

        return self.metrics

    def save_results(self):
        path = MODELS_DIR / self.model_type / "backtest_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"[Backtest] Resultados salvos em {path}")
