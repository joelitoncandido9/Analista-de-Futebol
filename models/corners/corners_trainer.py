"""Treinamento do modelo de escanteios (XGBoost).

Treina um XGBRegressor para prever total de escanteios por partida,
com validacao temporal e calibracao de probabilidades.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.settings import MODELS_DIR
from models.features import build_dataset, temporal_split, load_matches

warnings.filterwarnings("ignore")


def _league_dir(league: str | None) -> str:
    return league.lower().replace(" ", "_") if league else "all"


def _model_paths(league: str | None) -> dict[str, Path]:
    ld = _league_dir(league)
    base = MODELS_DIR / "corners"
    return {
        "model": base / ld / "model.json",
        "params": base / ld / "params.json",
        "metrics": base / ld / "metrics.json",
        "calib": base / ld / "calib.json",
    }


class CornersTrainer:
    """Treinador do modelo de escanteios com XGBoost."""

    def __init__(self, league: str | None = None):
        self.league = league
        self.paths = _model_paths(league)
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []
        self.metrics: dict = {}
        self.calib: dict = {}  # Para calibracao de probabilidades

    def _get_default_params(self) -> dict:
        return {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": 42,
            "early_stopping_rounds": 30,
            "eval_metric": "mae",
            "objective": "reg:absoluteerror",
            "verbosity": 0,
        }

    def train(self, test_games: int = 380,
              retrain_full: bool = True) -> dict:
        """Treina o modelo com validacao temporal.

        Args:
            test_games: Quantos jogos usar como teste (ultimos N).
            retrain_full: Se True, retreina com todos os dados ao final.

        Returns:
            Dict com metricas.
        """
        logger.info(f"[Corners] Gerando dataset para {self.league or 'todas ligas'}...")

        X, y = build_dataset(
            league=self.league,
            target_col="total_corners",
            min_data_games=5,
        )

        if X.empty:
            logger.error("[Corners] Dataset vazio — verifique se ha dados no banco")
            return {}

        self.feature_names = list(X.columns)
        logger.info(f"[Corners] Dataset: {len(X)} amostras, {len(self.feature_names)} features")

        # Split temporal
        X_train, X_test, y_train, y_test = temporal_split(X, y, test_games)

        params = self._get_default_params()

        if len(X_train) < 100:
            logger.warning("[Corners] Poucos dados de treino, usando params mais simples")
            params["max_depth"] = 3
            params["n_estimators"] = 100

        logger.info(f"[Corners] Treino: {len(X_train)} | Teste: {len(X_test)}")

        self.model = xgb.XGBRegressor(**{k: v for k, v in params.items()
                                          if k != "early_stopping_rounds"})

        eval_set = [(X_train.values, y_train.values), (X_test.values, y_test.values)]
        self.model.fit(
            X_train.values, y_train.values,
            eval_set=eval_set,
            verbose=False,
        )

        # Metricas no teste
        y_pred = self.model.predict(X_test.values)
        y_train_pred = self.model.predict(X_train.values)

        self.metrics = {
            "mae_test": float(mean_absolute_error(y_test, y_pred)),
            "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae_train": float(mean_absolute_error(y_train, y_train_pred)),
            "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "mean_actual": float(y_test.mean()),
            "mean_pred": float(y_pred.mean()),
            "league": self.league or "all",
        }

        # Calibracao: residuos para gerar probabilidades
        residuals = y_test.values - y_pred
        self.calib = {
            "residual_std": float(np.std(residuals)),
            "residual_mean": float(np.mean(residuals)),
        }

        logger.info(f"[Corners] MAE: {self.metrics['mae_test']:.2f} | "
                     f"RMSE: {self.metrics['rmse_test']:.2f} | "
                     f"Media real: {self.metrics['mean_actual']:.1f}")

        # Feature importance
        importance = self.model.feature_importances_
        feat_imp = sorted(zip(self.feature_names, importance),
                          key=lambda x: -x[1])
        logger.info("[Corners] Top 5 features:")
        for name, imp in feat_imp[:5]:
            logger.info(f"  {name}: {imp:.4f}")

        self._save()

        # Retreinar com todos os dados se solicitado
        if retrain_full and len(X_test) > 0:
            logger.info("[Corners] Retreinando com todos os dados...")
            self.model = xgb.XGBRegressor(**{k: v for k, v in params.items()
                                              if k != "early_stopping_rounds"})
            self.model.fit(X.values, y.values, verbose=False)
            self._save()

        return self.metrics

    def _save(self):
        """Salva modelo, params e metricas."""
        self.paths["model"].parent.mkdir(parents=True, exist_ok=True)

        if self.model:
            self.model.save_model(str(self.paths["model"]))
            logger.info(f"[Corners] Modelo salvo em {self.paths['model']}")

        with open(self.paths["params"], "w") as f:
            json.dump(self.feature_names, f)
        with open(self.paths["metrics"], "w") as f:
            json.dump(self.metrics, f)
        with open(self.paths["calib"], "w") as f:
            json.dump(self.calib, f)

    def load(self) -> bool:
        """Carrega modelo salvo."""
        if not self.paths["model"].exists():
            logger.warning(f"[Corners] Modelo nao encontrado em {self.paths['model']}")
            return False

        self.model = xgb.XGBRegressor()
        self.model.load_model(str(self.paths["model"]))

        if self.paths["params"].exists():
            with open(self.paths["params"]) as f:
                self.feature_names = json.load(f)
        if self.paths["metrics"].exists():
            with open(self.paths["metrics"]) as f:
                self.metrics = json.load(f)
        if self.paths["calib"].exists():
            with open(self.paths["calib"]) as f:
                self.calib = json.load(f)

        logger.info(f"[Corners] Modelo carregado ({self.metrics.get('mae_test', '?')} MAE)")
        return True
