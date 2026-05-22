"""Modelo Dixon-Coles para previsao de resultados de partidas.

Implementa o modelo classico de Dixon & Coles (1997) que modela
gols marcados por cada time como uma distribuicao Poisson
com parametros ajustados por forca de ataque, defesa e vantagem de casa.

Inclui o parametro rho (τ) para ajustar correlacao em jogos de baixo placar.

Referencia: Dixon, M. J. and Coles, S. G. (1997),
"Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
"""
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from loguru import logger

from config.settings import MODELS_DIR, DB_PATH
from models.features import load_matches, league_averages

warnings.filterwarnings("ignore")

def _model_path(league: str) -> Path:
    return MODELS_DIR / "results" / f"dixon_coles_{league.lower().replace(' ', '_')}.npz"


def _params_path(league: str) -> Path:
    return MODELS_DIR / "results" / f"dc_params_{league.lower().replace(' ', '_')}.json"


# Mapeamento de nomes de times: API-Football → football-data.co.uk
# Necessário porque a API retorna nomes completos e o modelo foi treinado com
# abreviações do football-data.co.uk
TEAM_NAME_MAP = {
    # La Liga
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Espanyol": "Espanol",
    "Real Sociedad": "Sociedad",
    "Athletic Club": "Ath Bilbao",
    "Athletic Club Bilbao": "Ath Bilbao",
    "Rayo Vallecano": "Vallecano",
    "UD Almeria": "Almeria",
    "Almeria": "Almeria",
    "Real Valladolid": "Valladolid",
    "Deportivo Alaves": "Alaves",
    "RCD Mallorca": "Mallorca",
    "Cadiz CF": "Cadiz",
    "CD Leganes": "Leganes",
    # Serie A
    "AC Milan": "Milan",
    "Inter Milan": "Inter",
    "Inter": "Inter",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Lazio": "Lazio",
    "ACF Fiorentina": "Fiorentina",
    "Hellas Verona": "Verona",
    "US Lecce": "Lecce",
    "Bologna FC": "Bologna",
    "Bologna 1909": "Bologna",
    "Parma Calcio 1913": "Parma",
    "Parma": "Parma",
    "Venezia FC": "Venezia",
    "Como": "Como 1907",
    "Como 1907": "Como 1907",
    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton & Hove Albion": "Brighton",
    "Brighton": "Brighton",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "West Ham United": "West Ham",
    # Bundesliga
    "Borussia Dortmund": "Dortmund",
    "Borussia M'gladbach": "M'gladbach",
    "Borussia Mönchengladbach": "M'gladbach",
    "Bayer Leverkusen": "Leverkusen",
    "RB Leipzig": "Leipzig",
    "Leipzig": "Leipzig",
    "Eintracht Frankfurt": "Frankfurt",
    "FSV Mainz 05": "Mainz",
    "Mainz": "Mainz",
    "VfB Stuttgart": "Stuttgart",
    "FC Augsburg": "Augsburg",
    "Werder Bremen": "Bremen",
    "VfL Wolfsburg": "Wolfsburg",
    "TSG Hoffenheim": "Hoffenheim",
    "1. FC Heidenheim 1846": "Heidenheim",
    "Heidenheim": "Heidenheim",
    "1. FC Union Berlin": "Union Berlin",
    "Union Berlin": "Union Berlin",
    "SC Freiburg": "Freiburg",
    "VfL Bochum": "Bochum",
    "Holstein Kiel": "Holstein Kiel",
    "St. Pauli": "St Pauli",
    "FC St. Pauli": "St Pauli",
    "FC St Pauli": "St Pauli",
    # Ligue 1
    "Paris Saint Germain": "PSG",
    "Paris Saint-Germain": "PSG",
    "AS Monaco": "Monaco",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "Lyon": "Lyon",
    "LOSC Lille": "Lille",
    "Lille": "Lille",
    "OGC Nice": "Nice",
    "Montpellier HSC": "Montpellier",
    "Stade Rennais": "Rennes",
    "Rennes": "Rennes",
    "Stade de Reims": "Reims",
    "Stade Brestois 29": "Brest",
    "Brest": "Brest",
    "RC Lens": "Lens",
    "FC Nantes": "Nantes",
    "Toulouse FC": "Toulouse",
    "AS Saint-Etienne": "St Etienne",
    "Saint-Etienne": "St Etienne",
    "Angers SCO": "Angers",
    "Angers": "Angers",
    "Le Havre AC": "Le Havre",
    "Le Havre": "Le Havre",
    "AJ Auxerre": "Auxerre",
    "Strasbourg": "Strasbourg",
    "RC Strasbourg Alsace": "Strasbourg",
    # Brasileirão (nomes API-Football → football_data.db)
    "Botafogo": "Botafogo RJ",
    "Botafogo RJ": "Botafogo RJ",
    "Botafogo-RJ": "Botafogo RJ",
    "Flamengo": "Flamengo RJ",
    "Flamengo RJ": "Flamengo RJ",
    "Flamengo-RJ": "Flamengo RJ",
    "Athletico Paranaense": "Athletico-PR",
    "Athletico-PR": "Athletico-PR",
    "Athletico PR": "Athletico-PR",
    "Atletico Mineiro": "Atletico-MG",
    "Atlético Mineiro": "Atletico-MG",
    "Atlético-MG": "Atletico-MG",
    "Atletico-MG": "Atletico-MG",
    "Atletico Goianiense": "Atletico GO",
    "Atlético Goianiense": "Atletico GO",
    "Atletico-GO": "Atletico GO",
    "Cuiaba": "Cuiaba",
    "Cuiabá": "Cuiaba",
    "Fortaleza": "Fortaleza",
    "Fortaleza EC": "Fortaleza",
    "Vasco": "Vasco",
    "Vasco da Gama": "Vasco",
    "Vasco DG": "Vasco",
    "RB Bragantino": "Bragantino",
    "Red Bull Bragantino": "Bragantino",
    "Bragantino": "Bragantino",
}


class DixonColes:
    """Modelo Dixon-Coles para resultados de partidas.

    Estima parametros de ataque e defesa para cada time,
    vantagem de casa e o parametro rho de Dixon-Coles.

    Uso:
        dc = DixonColes(league='Premier League')
        dc.train()
        prob_home, prob_draw, prob_away = dc.predict('Arsenal', 'Chelsea')
    """

    def __init__(self, league: str | None = None):
        self.league = league
        self.model_path = _model_path(league) if league else MODELS_DIR / "results" / "dixon_coles.npz"
        self.params_path = _params_path(league) if league else MODELS_DIR / "results" / "dc_params.json"
        self.teams: list[str] = []
        self.team_idx: dict[str, int] = {}
        self.n_teams = 0

        # Parametros
        self.attack: np.ndarray | None = None
        self.defense: np.ndarray | None = None
        self.home_adv: float = 0.0
        self.rho: float = 0.0  # Dixon-Coles adjustment
        self.intercept: float = 0.0

        # Medias historicas da liga
        self.avg_home_goals: float = 1.5
        self.avg_away_goals: float = 1.2

        self._trained = False

    def train(self, decay_weights: bool = True,
              half_life_games: int = 500) -> dict:
        """Estima parametros do modelo por maximum likelihood.

        Args:
            decay_weights: Se True, partidas mais antigas tem menos peso.
            half_life_games: Meia-vida do decaimento (em jogos).

        Returns:
            Dict com metricas do treinamento.
        """
        logger.info(f"[Dixon-Coles] Carregando dados para {self.league or 'todas ligas'}...")

        df = load_matches(league=self.league, require_corners=False)
        if df.empty:
            logger.error("[Dixon-Coles] Sem dados")
            return {}

        # So usar partidas com gols
        df = df.dropna(subset=["home_goals", "away_goals"]).copy()

        # Identificar times
        all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
        self.teams = sorted(all_teams)
        self.n_teams = len(self.teams)
        self.team_idx = {t: i for i, t in enumerate(self.teams)}

        # Medias da liga
        avgs = league_averages(df)
        league_key = self.league or list(avgs.keys())[0]
        if league_key in avgs:
            self.avg_home_goals = avgs[league_key]["avg_home_goals"]
            self.avg_away_goals = avgs[league_key]["avg_away_goals"]

        logger.info(f"[Dixon-Coles] {len(df)} partidas, {self.n_teams} times, "
                     f"media casa: {self.avg_home_goals:.2f}, fora: {self.avg_away_goals:.2f}")

        # Pesos de decaimento temporal
        if decay_weights and len(df) > half_life_games:
            df = df.sort_values("match_date").reset_index(drop=True)
            games_idx = np.arange(len(df))
            weights = 2 ** (-games_idx / half_life_games)
        else:
            weights = np.ones(len(df))

        # Otimizacao
        n_params = 2 * self.n_teams + 2  # attack + defense pra cada time + home_adv + rho
        x0 = np.zeros(n_params)

        # Chutes iniciais
        for i, team in enumerate(self.teams):
            team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)]
            team_goals_for = np.concatenate([
                team_matches[team_matches["home_team"] == team]["home_goals"].values,
                team_matches[team_matches["away_team"] == team]["away_goals"].values,
            ])
            team_goals_against = np.concatenate([
                team_matches[team_matches["home_team"] == team]["away_goals"].values,
                team_matches[team_matches["away_team"] == team]["home_goals"].values,
            ])
            if len(team_goals_for) > 0:
                x0[i] = math.log(max(np.mean(team_goals_for) / self.avg_home_goals, 0.01))
                x0[self.n_teams + i] = math.log(max(
                    np.mean(team_goals_against) / self.avg_away_goals, 0.01))

        x0[-2] = math.log(self.avg_home_goals / self.avg_away_goals)  # home_adv

        # Pre-computar arrays para otimizacao vetorizada
        self._home_idx = np.array([self.team_idx.get(t, -1) for t in df["home_team"]])
        self._away_idx = np.array([self.team_idx.get(t, -1) for t in df["away_team"]])
        self._home_goals = df["home_goals"].values.astype(int)
        self._away_goals = df["away_goals"].values.astype(int)
        self._weights = weights if isinstance(weights, np.ndarray) else np.array(weights)

        logger.info(f"[Dixon-Coles] Otimizando {n_params} parametros...")

        result = minimize(
            self._neg_log_likelihood_vec,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-6},
        )

        if not result.success:
            logger.warning(f"[Dixon-Coles] Otimizacao nao convergiu: {result.message}")

        x_opt = result.x
        self.attack = np.exp(x_opt[:self.n_teams])
        self.defense = np.exp(x_opt[self.n_teams:2 * self.n_teams])
        self.home_adv = math.exp(x_opt[-2])
        self.rho = np.tanh(x_opt[-1]) * 0.25  # rho entre -0.25 e 0.25
        self._trained = True

        logger.info(f"[Dixon-Coles] Home advantage: {self.home_adv:.3f}")
        logger.info(f"[Dixon-Coles] Rho: {self.rho:.4f}")

        # Top 5 ataques
        att_sorted = sorted(zip(self.teams, self.attack), key=lambda x: -x[1])
        logger.info("[Dixon-Coles] Top 5 ataques:")
        for team, att in att_sorted[:5]:
            logger.info(f"  {team}: {att:.3f}")

        self._save()
        return {"nll": float(result.fun), "n_teams": self.n_teams,
                "n_matches": len(df), "home_adv": float(self.home_adv)}

    def _neg_log_likelihood_vec(self, params: np.ndarray) -> float:
        """Negative log-likelihood vetorizada (numpy)."""
        n = self.n_teams
        attack = np.exp(params[:n])
        defense = np.exp(params[n:2 * n])
        home_adv = math.exp(params[-2])
        rho = np.tanh(params[-1]) * 0.25

        # Indices
        hi = self._home_idx
        ai = self._away_idx
        valid = (hi >= 0) & (ai >= 0)

        # Expected goals vetorizado
        mu_h = attack[hi] * defense[ai] * home_adv * self.avg_home_goals
        mu_a = attack[ai] * defense[hi] * self.avg_away_goals

        # Gols
        gh = self._home_goals
        ga = self._away_goals

        # Tau adjustment vetorizado
        tau = np.ones_like(mu_h)
        mask_00 = (gh == 0) & (ga == 0)
        mask_01 = (gh == 0) & (ga == 1)
        mask_10 = (gh == 1) & (ga == 0)
        mask_11 = (gh == 1) & (ga == 1)
        tau[mask_00] = 1.0 - mu_h[mask_00] * mu_a[mask_00] * rho
        tau[mask_01] = 1.0 + mu_h[mask_01] * rho
        tau[mask_10] = 1.0 + mu_a[mask_10] * rho
        tau[mask_11] = 1.0 - rho

        # Log-likelihood vetorizado
        ll = np.log(tau + 1e-10) + poisson.logpmf(gh, mu_h) + poisson.logpmf(ga, mu_a)
        nll = -np.sum(self._weights[valid] * ll[valid])

        return nll

    @staticmethod
    def _dc_tau(x: int, y: int, mu_h: float, mu_a: float, rho: float) -> float:
        """Dixon-Coles tau adjustment para correlacao em jogos de baixo placar."""
        if x == 0 and y == 0:
            return 1.0 - mu_h * mu_a * rho
        if x == 0 and y == 1:
            return 1.0 + mu_h * rho
        if x == 1 and y == 0:
            return 1.0 + mu_a * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def _resolve_team_name(self, name: str) -> str:
        """Tenta resolver nome do time via lookup direto ou mapeamento."""
        if name in self.team_idx:
            return name
        mapped = TEAM_NAME_MAP.get(name)
        if mapped and mapped in self.team_idx:
            return mapped
        return name

    def predict(self, home_team: str, away_team: str,
                max_goals: int = 10) -> tuple[float, float, float]:
        """Preve probabilidades de resultado.

        Returns:
            (prob_home, prob_draw, prob_away) em percentual (0-1).
        """
        if not self._trained:
            logger.error("[Dixon-Coles] Modelo nao treinado")
            return (0.0, 0.0, 0.0)

        home_resolved = self._resolve_team_name(home_team)
        away_resolved = self._resolve_team_name(away_team)
        team_h = self.team_idx.get(home_resolved)
        team_a = self.team_idx.get(away_resolved)
        if team_h is None or team_a is None:
            logger.warning(f"[Dixon-Coles] Time nao encontrado: {home_team if team_h is None else away_team}")
            return (0.0, 0.0, 0.0)

        mu_h = self.attack[team_h] * self.defense[team_a] * self.home_adv * self.avg_home_goals
        mu_a = self.attack[team_a] * self.defense[team_h] * self.avg_away_goals

        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = (self._dc_tau(i, j, mu_h, mu_a, self.rho)
                     * poisson.pmf(i, mu_h)
                     * poisson.pmf(j, mu_a))
                if i > j:
                    prob_home += p
                elif i == j:
                    prob_draw += p
                else:
                    prob_away += p

        return (prob_home, prob_draw, prob_away)

    def predict_score(self, home_team: str, away_team: str) -> dict:
        """Preve resultado mais provavel + distribuicao de gols."""
        probs = self.predict(home_team, away_team)

        home_resolved = self._resolve_team_name(home_team)
        away_resolved = self._resolve_team_name(away_team)
        team_h = self.team_idx.get(home_resolved)
        team_a = self.team_idx.get(away_resolved)
        if team_h is None or team_a is None:
            return {
                "home_team": home_team, "away_team": away_team,
                "prob_home": 0.0, "prob_draw": 0.0, "prob_away": 0.0,
                "expected_home_goals": 0.0, "expected_away_goals": 0.0,
                "most_likely_score": "0x0", "most_likely_prob": 0.0,
                "score_probabilities": {},
            }

        mu_h = self.attack[team_h] * self.defense[team_a] * self.home_adv * self.avg_home_goals
        mu_a = self.attack[team_a] * self.defense[team_h] * self.avg_away_goals

        # Placar mais provavel
        best_score = (0, 0)
        best_prob = 0.0
        score_probs = {}
        for i in range(7):
            for j in range(7):
                p = (self._dc_tau(i, j, mu_h, mu_a, self.rho)
                     * poisson.pmf(i, mu_h)
                     * poisson.pmf(j, mu_a))
                score_probs[f"{i}x{j}"] = round(float(p), 4)
                if p > best_prob:
                    best_prob = p
                    best_score = (i, j)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "prob_home": round(float(probs[0]), 4),
            "prob_draw": round(float(probs[1]), 4),
            "prob_away": round(float(probs[2]), 4),
            "expected_home_goals": round(float(mu_h), 2),
            "expected_away_goals": round(float(mu_a), 2),
            "most_likely_score": f"{best_score[0]}x{best_score[1]}",
            "most_likely_prob": round(float(best_prob), 4),
            "score_probabilities": score_probs,
        }

    def _save(self):
        (MODELS_DIR / "results").mkdir(parents=True, exist_ok=True)
        if self._trained:
            np.savez(str(self.model_path),
                     attack=self.attack, defense=self.defense,
                     home_adv=np.array([self.home_adv]),
                     rho=np.array([self.rho]),
                     teams=np.array(self.teams, dtype=object),
                     avg_home_goals=np.array([self.avg_home_goals]),
                     avg_away_goals=np.array([self.avg_away_goals]))
            logger.info(f"[Dixon-Coles] Modelo salvo em {self.model_path}")

    def load(self) -> bool:
        if not self.model_path.exists():
            logger.warning(f"[Dixon-Coles] Modelo nao encontrado em {self.model_path}")
            return False
        data = np.load(str(self.model_path), allow_pickle=True)
        self.attack = data["attack"]
        self.defense = data["defense"]
        self.home_adv = float(data["home_adv"][0])
        self.rho = float(data["rho"][0])
        self.teams = list(data["teams"])
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        self.n_teams = len(self.teams)
        self.avg_home_goals = float(data["avg_home_goals"][0])
        self.avg_away_goals = float(data["avg_away_goals"][0])
        self._trained = True
        logger.info(f"[Dixon-Coles] Modelo carregado ({self.n_teams} times)")
        return True
