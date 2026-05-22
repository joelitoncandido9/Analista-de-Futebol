"""Dashboard Football AI — Streamlit.

Uso: streamlit run dashboard/app.py
"""
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Football AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = st.sidebar.text_input("API URL", value="http://localhost:8000")

st.sidebar.markdown("---")
st.sidebar.markdown("### Navegacao")
page = st.sidebar.radio(
    "Pagina",
    ["Visao Geral", "Partidas", "Previsoes", "Times", "Modelos", "Analista"],
    label_visibility="collapsed",
)


def api_get(path: str) -> dict | list | None:
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Erro ao conectar: {e}")
        return None


def api_post(path: str, params: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"Erro: {e}")
        return None


# ─── Visao Geral ───────────────────────────────────────────────────────

if page == "Visao Geral":
    st.title("⚽ Football AI")
    st.markdown("### Painel de Controle")

    health = api_get("/health")
    db_stats = api_get("/stats/database")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Partidas", health.get("total_matches", 0) if health else 0)
    with col2:
        st.metric("Ligas", health.get("leagues", 0) if health else 0)
    with col3:
        st.metric("Status", health.get("status", "error") if health else "error")
    with col4:
        st.metric("Fontes", db_stats.get("sources", 0) if db_stats else 0)

    if db_stats:
        st.subheader("Partidas por Liga")
        df = pd.DataFrame(db_stats.get("per_league", []))
        if not df.empty:
            df.columns = ["Liga", "Total", "Primeira", "Ultima", "Com xG"]
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Links")
    st.markdown("[Documentacao da API](/docs)")

# ─── Partidas ──────────────────────────────────────────────────────────

elif page == "Partidas":
    st.title("📋 Partidas")

    col1, col2 = st.columns(2)
    with col1:
        league_filter = st.text_input("Liga (deixe vazio para todas)")
    with col2:
        team_filter = st.text_input("Time")
    limit = st.number_input("Limite", min_value=10, max_value=1000, value=50)

    params = {"limit": limit}
    if league_filter:
        params["league"] = league_filter
    if team_filter:
        params["team"] = team_filter

    matches = api_get(f"/matches?{'&'.join(f'{k}={v}' for k, v in params.items())}")

    if matches:
        df = pd.DataFrame(matches)
        cols = ["match_date", "league", "season", "home_team", "away_team",
                "home_goals", "away_goals", "home_corners", "away_corners",
                "home_shots", "away_shots", "home_xg", "away_xg"]
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols], hide_index=True, use_container_width=True)
        st.caption(f"Total: {len(df)} partidas")
    else:
        st.info("Nenhuma partida encontrada")

# ─── Previsoes ─────────────────────────────────────────────────────────

elif page == "Previsoes":
    st.title("🔮 Previsoes")

    tab1, tab2, tab3 = st.tabs(["Escanteios", "Finalizacoes", "Resultado"])

    with tab1:
        st.subheader("Previsao de Escanteios")
        c1, c2, c3 = st.columns(3)
        with c1:
            home = st.text_input("Casa", key="c_home")
        with c2:
            away = st.text_input("Fora", key="c_away")
        with c3:
            league = st.text_input("Liga", key="c_league")

        if st.button("Prever Escanteios") and home and away and league:
            result = api_get(
                f"/predict/corners?home={home}&away={away}&league={league}"
            )
            if result:
                st.metric("Total Previsto", result.get("predicted_total_corners", "?"))
                probs = result.get("probabilities", {})
                if probs:
                    st.subheader("Probabilidades")
                    for line, prob in list(probs.items())[:6]:
                        st.metric(line, f"{prob:.1%}")

    with tab2:
        st.subheader("Previsao de Finalizacoes")
        c1, c2, c3 = st.columns(3)
        with c1:
            home = st.text_input("Casa", key="s_home")
        with c2:
            away = st.text_input("Fora", key="s_away")
        with c3:
            league = st.text_input("Liga", key="s_league")

        if st.button("Prever Finalizacoes") and home and away and league:
            result = api_get(
                f"/predict/shots?home={home}&away={away}&league={league}"
            )
            if result:
                st.metric("Total Previsto", result.get("predicted_total_shots", "?"))
                probs = result.get("probabilities", {})
                if probs:
                    st.subheader("Probabilidades")
                    for line, prob in list(probs.items())[:6]:
                        st.metric(line, f"{prob:.1%}")

    with tab3:
        st.subheader("Previsao de Resultado (Dixon-Coles)")
        c1, c2, c3 = st.columns(3)
        with c1:
            home = st.text_input("Casa", key="r_home")
        with c2:
            away = st.text_input("Fora", key="r_away")
        with c3:
            league = st.text_input("Liga", key="r_league")

        if st.button("Prever Resultado") and home and away and league:
            result = api_get(
                f"/predict/result?home={home}&away={away}&league={league}"
            )
            if result:
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Casa", f"{result.get('prob_home', 0):.1%}")
                with cols[1]:
                    st.metric("Empate", f"{result.get('prob_draw', 0):.1%}")
                with cols[2]:
                    st.metric("Fora", f"{result.get('prob_away', 0):.1%}")
                st.metric("Placar Mais Provavel", result.get("most_likely_score", "?"))

# ─── Times ─────────────────────────────────────────────────────────────

elif page == "Times":
    st.title("🏆 Times")

    team_name = st.text_input("Nome do Time")
    league_filter = st.text_input("Liga (opcional)", key="team_league")

    if team_name:
        params = f"/teams/{team_name}"
        if league_filter:
            params += f"?league={league_filter}"
        data = api_get(params)
        if data:
            st.subheader(f"{data['name']}")
            st.markdown(f"**Ligas:** {', '.join(data['leagues'])}")

            if data.get("averages"):
                st.subheader("Medias (ultimos 10 jogos)")
                for lg, avg in data["averages"].items():
                    with st.expander(lg):
                        for k, v in avg.items():
                            st.metric(k.replace("_", " ").title(), f"{v:.2f}")

            if data.get("recent_matches"):
                st.subheader("Ultimas Partidas")
                df = pd.DataFrame(data["recent_matches"])
                cols = ["match_date", "league", "home_team", "away_team",
                        "home_goals", "away_goals"]
                cols = [c for c in cols if c in df.columns]
                st.dataframe(df[cols], hide_index=True, use_container_width=True)

# ─── Modelos ───────────────────────────────────────────────────────────

elif page == "Modelos":
    st.title("🧠 Modelos")

    model_data = api_get("/stats/models")
    if model_data and model_data.get("status") == "trained":
        results = model_data["results"]
        for league_name, metrics in results.items():
            with st.expander(f"**{league_name}**", expanded=True):
                col1, col2, col3 = st.columns(3)

                corners = metrics.get("corners", {})
                with col1:
                    st.markdown("**Escanteios**")
                    if corners.get("mae_test"):
                        st.metric("MAE", f"{corners['mae_test']:.2f}")
                        st.metric("RMSE", f"{corners['rmse_test']:.2f}")
                        st.metric("Treino", f"{corners['n_train']} jogos")
                        st.metric("Teste", f"{corners['n_test']} jogos")
                    else:
                        st.info(corners.get("error", "N/A"))

                shots = metrics.get("shots", {})
                with col2:
                    st.markdown("**Finalizacoes**")
                    if shots.get("mae_test"):
                        st.metric("MAE", f"{shots['mae_test']:.2f}")
                        st.metric("RMSE", f"{shots['rmse_test']:.2f}")
                        st.metric("Treino", f"{shots['n_train']} jogos")
                        st.metric("Teste", f"{shots['n_test']} jogos")
                    else:
                        st.info(shots.get("error", "N/A"))

                dc = metrics.get("dixon_coles", {})
                with col3:
                    st.markdown("**Dixon-Coles**")
                    if dc.get("n_teams"):
                        st.metric("Times", dc["n_teams"])
                        st.metric("Partidas", dc["n_matches"])
                        st.metric("Home Adv.", f"{dc.get('home_adv', 0):.2f}")
                    else:
                        st.info(dc.get("error", "N/A"))
    else:
        st.warning("Nenhum modelo treinado ainda.")
        st.info("Execute: `python models/train_all.py`")

# ─── Analista ──────────────────────────────────────────────────────────

elif page == "Analista":
    st.title("🤖 Analista Tatico")

    question = st.text_area(
        "Faça uma pergunta sobre futebol, times, jogadores ou partidas:",
        placeholder="Ex: Qual a vantagem de jogar em casa na Premier League esse ano?",
        height=100,
    )
    league = st.text_input("Liga (opcional)", placeholder="Deixe vazio para todas")

    if st.button("Perguntar") and question:
        with st.spinner("Analisando..."):
            result = api_post("/agents/analyst", {"question": question, "league": league})
            if result:
                st.markdown("### Resposta")
                st.markdown(result["response"])

    st.markdown("---")
    st.subheader("Scouting")

    player = st.text_input("Nome do Jogador", placeholder="Ex: Vinicius Junior")
    scout_league = st.text_input("Liga (opcional)", key="scout_league",
                                  placeholder="Ex: La Liga")

    if st.button("Scout") and player:
        with st.spinner("Buscando dados do jogador..."):
            params = {"player_name": player}
            if scout_league:
                params["league"] = scout_league
            result = api_post("/agents/scout", params)
            if result:
                st.markdown("### Relatorio de Scouting")
                st.markdown(result["report"])
