"""Backtesting dos modelos: validacao temporal com metricas reais.

Simula apostas em jogos historicos para avaliar performance:
- ROI por linha de over/under
- Acuracia das probabilidades (calibracao)
- Value se tivessemos odds reais do mercado
"""
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
        """
        Args:
            model_type: 'corners' ou 'shots'
            league: Liga especifica ou None para todas
        """
        self.model_type = model_type
        self.league = league
        self.target_col = "total_corners" if model_type == "corners" else "total_shots"
        self.results: list[dict] = []
        self.metrics: dict = {}

    def run(self, test_games: int = 760, window: int = 500) -> dict:
        """Executa backtesting com janela deslizante.

        Simula: treina com ultimos N jogos, preve proximo jogo, avanca.

        Args:
            test_games: Quantos jogos testar.
            window: Janela de treino (ultimos N jogos).

        Returns:
            Dict com metricas de performance.
        """
        df = load_matches(league=self.league)
        if df.empty:
            return {}

        # Filtrar partidas com dados completos
        df = df.dropna(subset=[self.target_col]).reset_index(drop=True)

        if len(df) < window + test_games:
            logger.warning(f"[Backtest] Dados insuficientes: {len(df)} partidas")
            return {}

        logger.info(f"[Backtest] Iniciando com {len(df)} partidas, "
                     f"testando ultimas {test_games}")

        from xgboost import XGBRegressor

        predictions = []
        actuals = []

        for i in range(len(df) - test_games, len(df)):
            test_row = df.iloc[i]

            # Dados de treino: ate o jogo anterior
            train_df = df.iloc[max(0, i - window):i]

            if len(train_df) < 30:
                continue

            # Construir features manualmente para esta particao
            feats = build_match_features(test_row, train_df)
            if feats["home_total_games"] < 3 or feats["away_total_games"] < 3:
                continue

            # Montar dataset de treino
            train_features = []
            train_targets = []
            for _, tr in train_df.iterrows():
                f = build_match_features(tr, train_df)
                if f["home_total_games"] >= 3 and f["away_total_games"] >= 3:
                    train_features.append(f)
                    train_targets.append(tr.get(self.target_col))

            if len(train_features) < 50:
                continue

            X_train = pd.DataFrame(train_features).fillna(0)
            y_train = np.array(train_targets)

            feature_names = list(X_train.columns)

            # Feature do jogo de teste
            X_test = pd.DataFrame([feats]).fillna(0)
            # Garantir colunas iguais
            for col in feature_names:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[feature_names]

            # Treinar e prever
            model = XGBRegressor(n_estimators=200, max_depth=4,
                                  learning_rate=0.1, verbosity=0)
            model.fit(X_train.values, y_train)
            pred = float(model.predict(X_test.values)[0])
            actual = float(test_row[self.target_col])

            predictions.append(pred)
            actuals.append(actual)

            if (i - (len(df) - test_games)) % 100 == 0:
                logger.info(f"[Backtest] Progresso: {i - (len(df) - test_games) + 1}/{test_games}")

        if not predictions:
            logger.warning("[Backtest] Nenhuma previsao gerada")
            return {}

        # Metricas
        preds = np.array(predictions)
        acts = np.array(actuals)

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
            "n_tests": len(predictions),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "bias": round(bias, 3),
            "mean_actual": round(float(acts.mean()), 2),
            "mean_predicted": round(float(preds.mean()), 2),
            "line_accuracy": line_results,
        }

        logger.info(f"[Backtest] {self.model_type.upper()} | "
                     f"MAE: {mae:.2f} | Bias: {bias:.2f} | "
                     f"N: {len(predictions)}")

        return self.metrics

    def save_results(self):
        """Salva resultados em JSON."""
        path = MODELS_DIR / self.model_type / "backtest_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"[Backtest] Resultados salvos em {path}")
