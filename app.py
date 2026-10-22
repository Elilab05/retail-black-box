"""Interactive investigation dashboard; no claims beyond observed data/experiments."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allows `streamlit run app.py` even when package was not installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from retail_black_box.data import load_data, store_day_coverage
from retail_black_box.quality import quality_report, quality_flags
from retail_black_box.incidents import identify_incidents, simulate_masked_day
from retail_black_box.experiments import masked_outage_experiment

st.set_page_config(page_title="Retail Black Box", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1350px}
h1,h2,h3 {letter-spacing: -.025em}
[data-testid="stMetricValue"] {font-weight: 700}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Leyendo las tres fuentes...")
def get_data():
    return load_data("data")


@st.cache_data(show_spinner="Analizando integridad...")
def get_analysis(t, s, c):
    return quality_report(t, s, c), identify_incidents(t, s, c)


def metric_mxn(value):
    return f"${value / 1_000_000:,.2f} M" if pd.notna(value) else "N/D"


st.title("Retail Black Box")
st.caption("Operational intelligence · Explain first, predict second · Technical proof of concept")
try:
    transactions, stores, calendar = get_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.info("Coloca transactions.csv, stores.csv y calendar.csv de la prueba en la carpeta data/.")
    st.stop()

report, (incidents, details) = get_analysis(transactions, stores, calendar)
all_regions = ["Todas"] + sorted(stores["region"].unique().tolist())
region = st.sidebar.selectbox("Región", all_regions)
filter_stores = stores if region == "Todas" else stores[stores["region"] == region]
ids = filter_stores["store_id"].tolist()
selected_store = st.sidebar.selectbox("Tienda", ["Todas"] + sorted(ids))
category = st.sidebar.selectbox("Categoría", ["Todas"] + sorted(transactions["category"].unique().tolist()))
filtered = transactions[transactions["store_id"].isin(ids)].copy()
if selected_store != "Todas":
    filtered = filtered[filtered["store_id"] == selected_store]
if category != "Todas":
    filtered = filtered[filtered["category"] == category]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros observados", f"{len(filtered):,}")
k2.metric("Días sin datos (cadena)", report["missing_whole_store_days"])
k3.metric("Conteos inconsistentes (cadena)", f"{report['mismatched_transaction_count']:,}")
k4.metric("Volumen esperado no observado*", metric_mxn(incidents["estimate_mxn"].sum()))
st.caption("*Estimación sobre registros ausentes; NO representa ventas perdidas ni un monto recuperable. Indicadores de calidad e incidentes abarcan la cadena completa salvo el volumen de registros filtrados.")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Operación", "🚨 Incidentes", "🧪 Replay & Stress Test", "🎯 Forecast (backtest)"])
with tab1:
    st.subheader("Comportamiento comercial observado")
    daily = filtered.groupby("date", as_index=False)["amount_total"].sum()
    fig = px.line(daily, x="date", y="amount_total", labels={"date": "Fecha", "amount_total": "Ventas registradas (MXN)"})
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        categories = filtered.groupby("category", as_index=False)["amount_total"].sum().sort_values("amount_total", ascending=False)
        st.plotly_chart(px.bar(categories, x="category", y="amount_total", title="Volumen registrado por categoría"), use_container_width=True)
    with right:
        q = quality_flags(filtered)
        counts = pd.DataFrame({"validación": ["Conteo (cash+card)", "Ticket promedio", "Importes (cash+card)"], "filas": [int(q["transaction_count_mismatch"].sum()), int(q["ticket_mismatch"].sum()), int(q["amount_mismatch"].sum())]})
        st.plotly_chart(px.bar(counts, x="filas", y="validación", orientation="h", title="Alertas de calidad en el filtro"), use_container_width=True)
    with st.expander("Resumen completo de calidad y supuestos"):
        st.json(report)

with tab2:
    st.subheader("Incidentes de cobertura")
    st.markdown("Días en que una tienda no reportó ninguna de las seis categorías. Requieren investigación de origen; no se etiquetan automáticamente como ventas cero.")
    if len(incidents):
        shown = incidents[incidents["store_id"].isin(ids)]
        if selected_store != "Todas":
            shown = shown[shown["store_id"] == selected_store]
        if len(shown):
            st.dataframe(shown[["date", "store_id", "region", "estimate_mxn", "p10_mxn", "p90_mxn", "categories_estimated", "status"]].sort_values("estimate_mxn", ascending=False), use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(shown, x="date", y="estimate_mxn", color="store_id", title="Estimación puntual de volumen no observado (MXN)"), use_container_width=True)
            st.caption("p10/p90 son cuantiles empíricos de una muestra histórica reducida; NO están calibrados como intervalos de predicción.")
        else:
            st.info("Sin días completamente ausentes para el filtro elegido.")
    else:
        st.success("No se detectaron días completos ausentes.")

with tab3:
    st.subheader("Experimento: ocultar un día que sí conocemos")
    st.markdown("Ocultamos conceptualmente las ventas de una tienda en una fecha observada. La reconstrucción solo consulta fechas anteriores. Así podemos comparar estimación contra el valor real.")
    sim_store = st.selectbox("Tienda para experimento", sorted(ids), index=0)
    earliest = transactions["date"].min() + pd.Timedelta(days=70)
    sim_dates = transactions.loc[(transactions["store_id"] == sim_store) & (transactions["date"] >= earliest), "date"].drop_duplicates().sort_values()
    if len(sim_dates):
        sim_date = st.select_slider("Día observado a ocultar", options=list(sim_dates.dt.date), value=sim_dates.dt.date.iloc[-10] if len(sim_dates) >= 10 else sim_dates.dt.date.iloc[-1])
        if st.button("Ejecutar replay", type="primary"):
            comparison = simulate_masked_day(transactions, sim_store, pd.Timestamp(sim_date))
            if len(comparison):
                comparison["error_abs"] = (comparison["estimate"] - comparison["actual"]).abs()
                a, b, d = st.columns(3)
                a.metric("Ventas reales", metric_mxn(comparison["actual"].sum()))
                b.metric("Ventas estimadas", metric_mxn(comparison["estimate"].sum()))
                d.metric("Error absoluto", metric_mxn(comparison["error_abs"].sum()))
                grouped = comparison.melt(id_vars=["category"], value_vars=["actual", "estimate"], var_name="serie", value_name="MXN")
                st.plotly_chart(px.bar(grouped, x="category", y="MXN", color="serie", barmode="group", title="Valor conocido vs estimación histórica"), use_container_width=True)
                st.dataframe(comparison[["category", "actual", "estimate", "support", "method"]], hide_index=True, use_container_width=True)
    with st.expander("Resultados del backtest masivo"):
        path = Path("outputs/backtest_report.json")
        if path.exists():
            info = json.loads(path.read_text(encoding="utf-8"))
            st.metric("WAPE de reconstrucción", f"{info['wape']:.2%}")
            st.metric("Días simulados", info["sampled_store_days"])
            st.json(info)
        else:
            st.info("Ejecuta: retail-black-box backtest")

with tab4:
    st.subheader("Demanda a 7 días: evaluación estrictamente histórica")
    path = Path("outputs/forecast_report.json")
    predictions_file = Path("outputs/forecast_predictions.csv")
    if path.exists() and predictions_file.exists():
        scores = json.loads(path.read_text(encoding="utf-8"))
        preds = pd.read_csv(predictions_file, parse_dates=["date_target", "date_asof"])
        if selected_store != "Todas":
            preds = preds[preds["store_id"] == selected_store]
        else:
            preds = preds[preds["store_id"].isin(ids)]
        if category != "Todas":
            preds = preds[preds["category"] == category]
        v1, v2, v3 = st.columns(3)
        v1.metric("WAPE LightGBM (cadena)", f"{scores['wape_ml']:.2%}")
        v2.metric("WAPE seasonal naive (cadena)", f"{scores['wape_seasonal_naive']:.2%}")
        v3.metric("Modelo supera baseline", "Sí" if scores["model_wins"] else "No")
        trend = preds.groupby("date_target", as_index=False)[["actual", "predicted_ml", "predicted_seasonal_naive"]].sum()
        melted = trend.melt(id_vars=["date_target"], var_name="serie", value_name="MXN")
        st.plotly_chart(px.line(melted, x="date_target", y="MXN", color="serie", title="Ventas reales vs predicciones históricas"), use_container_width=True)
        st.caption("Se pronostica t+7 usando solo información disponible al cierre de t. Entrenamiento con etiquetas hasta 2024-01-24; evaluación de objetivos en febrero 2024.")
        with st.expander("Protocolo y resultados completos"):
            st.json(scores)
    else:
        st.info("Primero ejecuta: pip install -e '.[all]' y después retail-black-box forecast. Aquí no se muestran métricas inventadas.")

st.divider()
st.caption("Technical challenge · Proof of concept · Findings are descriptive, not causal · Original CSVs are never modified")
