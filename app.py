"""Retail Black Box 1.1: scoped KPIs, incident triage and transparent backtesting."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from retail_black_box.data import load_data
from retail_black_box.quality import quality_flags, quality_report
from retail_black_box.incidents import identify_incidents, simulate_masked_day
from retail_black_box.dashboard_metrics import compute_wape, scope_incidents, selection_quality_summary

st.set_page_config(page_title="Retail Black Box | Decision Intelligence", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.4rem; max-width: 1450px}
h1,h2,h3 {letter-spacing: -.025em}
[data-testid="stMetricValue"] {font-weight: 750}
.rbb-eyebrow {font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; opacity:.7; font-weight:700}
.rbb-subtle {font-size:.9rem; opacity:.72}
.rbb-divider {height:2px; margin:.8rem 0 1rem; background:linear-gradient(90deg, #00b4d8, #a78bfa, transparent)}
</style>""", unsafe_allow_html=True)


def money(value: float) -> str:
    if pd.isna(value):
        return "N/D"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f} mil M"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f} M"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) and np.isfinite(value) else "N/D"


@st.cache_data(show_spinner="Cargando datos comerciales...")
def get_data():
    return load_data(ROOT / "data")


@st.cache_data(show_spinner="Validando datos y cobertura...")
def get_analysis(t, s, c):
    return quality_report(t, s, c), identify_incidents(t, s, c)


@st.cache_data(show_spinner="Leyendo resultados de evaluación...")
def get_forecast():
    fp = ROOT / "outputs" / "forecast_predictions.csv"
    rp = ROOT / "outputs" / "forecast_report.json"
    if not (fp.exists() and rp.exists()):
        return None, None
    return pd.read_csv(fp, parse_dates=["date_target", "date_asof"]), json.loads(rp.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Cargando validación multimes...")
def get_rolling_predictions():
    fp = ROOT / "outputs" / "rolling_predictions.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp)


st.markdown('<div class="rbb-eyebrow">Retail Operations • Decision Intelligence</div>', unsafe_allow_html=True)
st.title("Retail Black Box")
st.caption("Detectar → investigar → priorizar. Diagnósticos verificables sobre datos reales de la prueba técnica.")
st.markdown('<div class="rbb-divider"></div>', unsafe_allow_html=True)

try:
    t, stores, calendar = get_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.info("Los tres CSV del ejercicio deben estar en la carpeta data/.")
    st.stop()

quality_global, (incidents, estimate_detail) = get_analysis(t, stores, calendar)

st.sidebar.header("Ámbito de análisis")
region = st.sidebar.selectbox("Región", ["Todas"] + sorted(stores["region"].dropna().unique().tolist()))
region_stores = stores if region == "Todas" else stores[stores["region"] == region]
selected_store = st.sidebar.selectbox("Tienda", ["Todas"] + sorted(region_stores["store_id"].tolist()))
category = st.sidebar.selectbox("Categoría", ["Todas"] + sorted(t["category"].dropna().unique().tolist()))
store_ids = [selected_store] if selected_store != "Todas" else region_stores["store_id"].tolist()
selected = t[t["store_id"].isin(store_ids)]
if category != "Todas":
    selected = selected[selected["category"] == category]
selected = selected.copy()
flags = quality_flags(selected)
scoped_quality = selection_quality_summary(selected, flags)
scoped_incidents = scope_incidents(incidents, estimate_detail, store_ids, None if category == "Todas" else category)

scope_label = "Toda la cadena" if region == selected_store == category == "Todas" else "Selección actual"
st.markdown(f"**Vista:** {scope_label} · Región: {region} · Tienda: {selected_store} · Categoría: {category}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ventas registradas", money(scoped_quality["observed_amount_mxn"]), help="Suma observada dentro de los filtros actuales.")
c2.metric("Registros", f"{scoped_quality['observed_rows']:,}", help="Registros observados dentro de la selección.")
c3.metric("Conteos inconsistentes", f"{scoped_quality['mismatched_count']:,}", help="Cash + card distinto de transacciones totales, dentro de la selección.")
c4.metric("Volumen esperado sin observar*", money(scoped_incidents["estimate_mxn"].sum(min_count=1)), help="Estimación no causal sobre fechas con ausencia completa de registros; respeta tienda y categoría.")
st.caption("Todos los indicadores superiores responden a los filtros. *Volumen esperado durante periodos sin datos; **no es una pérdida comprobada** ni ingreso recuperable.")

operation, cases, replay, forecast = st.tabs(["📈 Panorama", "🚨 Investigar incidencias", "🧪 Replay & Stress Test", "🎯 Evaluar el modelo"])

with operation:
    st.subheader("Comportamiento comercial observado")
    if selected.empty:
        st.info("No hay transacciones observadas para esta selección.")
    else:
        daily = selected.groupby("date", as_index=False)["amount_total"].sum().sort_values("date")
        daily["rolling_7"] = daily["amount_total"].rolling(7, min_periods=3).mean()
        daily["ventas_m"] = daily["amount_total"] / 1_000_000
        daily["promedio7_m"] = daily["rolling_7"] / 1_000_000
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["ventas_m"], name="Ventas observadas", mode="lines", line=dict(color="#43b3f3", width=1.5), hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} M MXN<extra></extra>"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["promedio7_m"], name="Media móvil 7 días", mode="lines", line=dict(color="#f59e0b", width=2), hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} M MXN<extra></extra>"))
        fig.update_layout(height=365, xaxis_title="Fecha", yaxis_title="Millones MXN", margin=dict(l=10, r=10, t=5, b=10), legend_orientation="h", legend_y=1.13)
        st.plotly_chart(fig, use_container_width=True)
        l, r = st.columns([1.1, 1])
        with l:
            grouped = selected.groupby("category", as_index=False)["amount_total"].sum().sort_values("amount_total")
            chart = px.bar(grouped, x="amount_total", y="category", orientation="h", title="Mix comercial por categoría", labels={"amount_total": "MXN", "category": "Categoría"})
            chart.update_layout(height=315, margin=dict(l=10,r=10,t=50,b=10), showlegend=False)
            st.plotly_chart(chart, use_container_width=True)
        with r:
            qcounts = pd.DataFrame({"Control": ["Conteo de transacciones", "Ticket promedio", "Importes por medio de pago"], "Alertas": [scoped_quality["mismatched_count"], scoped_quality["mismatched_ticket"], scoped_quality["mismatched_amount"]]})
            chart = px.bar(qcounts, x="Alertas", y="Control", orientation="h", title="Controles de integridad en la selección")
            chart.update_layout(height=315, margin=dict(l=10,r=10,t=50,b=10), showlegend=False)
            st.plotly_chart(chart, use_container_width=True)
        with st.expander("Auditoría de calidad de toda la cadena"):
            st.json(quality_global)
            st.caption("Estas estadísticas son globales y no cambian con los filtros.")

with cases:
    st.subheader("Cola de investigación: periodos sin registros")
    st.write("La ausencia de seis categorías en una tienda y fecha es una **brecha de cobertura**. No implica que la tienda cerró ni que se perdieron ventas.")
    if scoped_incidents.empty:
        st.info("No hay incidentes de cobertura total para esta selección. Cambia los filtros si quieres investigar otras tiendas.")
    else:
        priority = scoped_incidents[["date", "store_id", "region", "estimate_mxn", "categories_estimated"]].copy()
        priority["prioridad"] = range(1, len(priority) + 1)
        priority = priority[["prioridad", "store_id", "date", "region", "estimate_mxn", "categories_estimated"]]
        st.dataframe(priority, use_container_width=True, hide_index=True, column_config={"prioridad": "Orden de revisión*", "date": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"), "store_id": "Tienda", "region": "Región", "estimate_mxn": st.column_config.NumberColumn("Volumen esperado (MXN)", format="$%.0f"), "categories_estimated": "Categorías estimadas"})
        st.caption("*Orden por volumen esperado estimado, no por riesgo confirmado. Los importes no representan ingresos perdidos.")
        choices = [f"{r.store_id} | {r.date.date()}" for r in scoped_incidents.itertuples(index=False)]
        choice = st.selectbox("Inspeccionar caso", choices)
        idx = choices.index(choice)
        row = scoped_incidents.iloc[idx]
        sid = row["store_id"]
        event_date = pd.Timestamp(row["date"])
        d1, d2, d3 = st.columns(3)
        d1.metric("Tienda", sid)
        d2.metric("Fecha sin registros", str(event_date.date()))
        d3.metric("Volumen esperado", money(row["estimate_mxn"]))
        timeline = t[(t["store_id"] == sid) & (t["date"].between(event_date - pd.Timedelta(days=28), event_date + pd.Timedelta(days=14)))].copy()
        if category != "Todas":
            timeline = timeline[timeline["category"] == category]
        timeline = timeline.groupby("date")["amount_total"].sum().reindex(pd.date_range(event_date - pd.Timedelta(days=28), event_date + pd.Timedelta(days=14))).rename_axis("date").reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline["date"], y=timeline["amount_total"], mode="lines+markers", name="Ventas observadas", connectgaps=False))
        fig.add_trace(go.Scatter(x=[event_date], y=[row["estimate_mxn"]], mode="markers", name="Estimación (no observada)", marker=dict(size=14, symbol="diamond", color="#fb923c")))
        fig.add_shape(type="line", x0=event_date, x1=event_date, y0=0, y1=1, yref="paper", line=dict(color="#fb923c", dash="dash"))
        fig.update_layout(height=340, xaxis_title="Fecha", yaxis_title="MXN", title="Evidencia temporal: no convertir ausencias en ceros", margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)
        specific = estimate_detail[(estimate_detail["store_id"] == sid) & (estimate_detail["date"] == event_date)].copy()
        if category != "Todas":
            specific = specific[specific["category"] == category]
        if not specific.empty:
            st.dataframe(specific[["category", "estimate", "support", "method"]], hide_index=True, use_container_width=True, column_config={"estimate": st.column_config.NumberColumn("Volumen esperado (MXN)", format="$%.0f"), "support": "Observaciones históricas", "method": "Método"})
        with st.container(border=True):
            st.markdown("**Acción operativa sugerida**")
            st.write("1) Revisar si hubo fallas en ingesta, conectividad o el POS; 2) contrastar con cierre contable u otra fuente independiente; 3) solo entonces determinar si existió una caída real de ventas.")
            st.caption("El diagnóstico es una hipótesis de calidad de datos, no una atribución causal automática.")
        st.download_button("Descargar cola de incidencias (CSV)", priority.to_csv(index=False).encode("utf-8-sig"), file_name="retail_incidents_scoped.csv", mime="text/csv")

with replay:
    st.subheader("Experimento de ocultamiento retrospectivo")
    st.write("Selecciona una fecha con ventas conocidas. Se ocultan conceptualmente sus registros y se reconstruye la demanda usando **solo días anteriores**.")
    sim_store = st.selectbox("Tienda para probar", sorted(store_ids))
    earliest = t["date"].min() + pd.Timedelta(days=70)
    valid_dates = t.loc[(t["store_id"] == sim_store) & (t["date"] >= earliest), "date"].drop_duplicates().sort_values()
    if valid_dates.empty:
        st.info("No hay suficientes observaciones históricas para esta tienda.")
    else:
        values = valid_dates.dt.date.tolist()
        default = values[-min(10, len(values))]
        sim_date = st.select_slider("Día a ocultar", options=values, value=default)
        if st.button("Ejecutar replay", type="primary"):
            comparison = simulate_masked_day(t, sim_store, pd.Timestamp(sim_date))
            if category != "Todas":
                comparison = comparison[comparison["category"] == category]
            if comparison.empty:
                st.warning("No hay datos de la categoría seleccionada para la fecha.")
            else:
                measured = compute_wape(comparison["actual"], comparison["estimate"])
                a,b,c = st.columns(3)
                a.metric("Observado", money(comparison["actual"].sum()))
                b.metric("Estimado", money(comparison["estimate"].sum(min_count=1)))
                c.metric("WAPE de esta simulación", pct(measured))
                melted = comparison.melt(id_vars=["category"], value_vars=["actual", "estimate"], var_name="serie", value_name="MXN")
                st.plotly_chart(px.bar(melted, x="category", y="MXN", color="serie", barmode="group", title="Observado vs estimado en la fecha ocultada"), use_container_width=True)
                st.dataframe(comparison[["category", "actual", "estimate", "support", "method"]], hide_index=True, use_container_width=True)
    with st.expander("Evaluación masiva de reconstrucción (120 días)"):
        path = ROOT / "outputs" / "backtest_report.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            b1,b2,b3 = st.columns(3)
            b1.metric("WAPE", pct(result["wape"]))
            b2.metric("Días evaluados", int(result["sampled_store_days"]))
            b3.metric("Cobertura p10–p90 observada", pct(result["empirical_p10_p90_coverage"]))
            st.warning("Los cuantiles p10–p90 históricos NO son intervalos calibrados. La cobertura observada no alcanza por sí sola para sostener garantías probabilísticas.")
        else:
            st.info("Falta outputs/backtest_report.json; ejecuta el backtest para generar los resultados.")

with forecast:
    st.subheader("Forecast a 7 días · Evaluación fuera de muestra")
    predictions, report = get_forecast()
    if predictions is None:
        st.info("Faltan outputs/forecast_report.json o outputs/forecast_predictions.csv. Ejecuta el pipeline de evaluación.")
    else:
        preds = predictions[predictions["store_id"].isin(store_ids)].copy()
        if category != "Todas":
            preds = preds[preds["category"] == category]
        if preds.empty:
            st.info("No hay predicciones históricas para la selección.")
        else:
            w_ml = compute_wape(preds["actual"], preds["predicted_ml"])
            w_base = compute_wape(preds["actual"], preds["predicted_seasonal_naive"])
            rel = (w_base - w_ml) / w_base if w_base > 0 else float("nan")
            a,b,c,d = st.columns(4)
            a.metric("WAPE LightGBM", pct(w_ml))
            b.metric("WAPE baseline", pct(w_base))
            c.metric("Mejora relativa", pct(rel), help="(WAPE baseline - WAPE ML) / WAPE baseline; negativa si ML es peor.")
            d.metric("Predicciones evaluadas", f"{len(preds):,}")
            st.caption(f"Métricas recalculadas para esta selección. Backtest histórico: {report['evaluation_target_start']} a {report['evaluation_target_end']}; horizonte de 7 días.")
            day = preds.groupby("date_target", as_index=False)[["actual", "predicted_ml", "predicted_seasonal_naive"]].sum()
            day = day.melt(id_vars="date_target", var_name="Serie", value_name="MXN")
            lab = {"actual":"Observado", "predicted_ml":"LightGBM", "predicted_seasonal_naive":"Seasonal naive"}
            day["Serie"] = day["Serie"].map(lab)
            figure = px.line(day, x="date_target", y="MXN", color="Serie", labels={"date_target":"Día objetivo"}, title="Ventas históricas observadas vs pronósticos (mismo periodo)")
            figure.update_layout(height=410, margin=dict(l=10,r=10,t=55,b=75), legend=dict(orientation="h", y=-0.27, x=0), legend_title_text="", xaxis_title="Fecha objetivo")
            st.plotly_chart(figure, use_container_width=True)
            # Expanding-window validation: filtered independently of February holdout.
            monthly_fp = ROOT / "outputs" / "rolling_predictions.csv"
            if monthly_fp.exists():
                st.subheader("Robustez temporal · nueve cortes mensuales")
                st.caption("Cada mes se evalúa con un modelo reentrenado con etiquetas disponibles antes del primer día de pronóstico. Pronósticos de siete días; periodos de junio 2023 a febrero 2024.")
                rolling_data = get_rolling_predictions()
                rolling_data = rolling_data[rolling_data["store_id"].isin(store_ids)]
                if category != "Todas":
                    rolling_data = rolling_data[rolling_data["category"] == category]
                if not rolling_data.empty:
                    monthly = []
                    for month, g in rolling_data.groupby("period", sort=True):
                        wr_ml = compute_wape(g["actual"], g["predicted_ml"])
                        wr_baseline = compute_wape(g["actual"], g["predicted_seasonal_naive"])
                        bias = (g["predicted_ml"] - g["actual"]).sum() / g["actual"].sum() if g["actual"].sum() else float("nan")
                        monthly.append({"Mes": str(month), "LightGBM (%)": wr_ml * 100, "Baseline (%)": wr_baseline * 100, "Sesgo ML (%)": bias * 100, "Filas": len(g)})
                    monthly = pd.DataFrame(monthly)
                    w_all_ml = compute_wape(rolling_data["actual"], rolling_data["predicted_ml"])
                    w_all_base = compute_wape(rolling_data["actual"], rolling_data["predicted_seasonal_naive"])
                    worst = monthly.loc[monthly["LightGBM (%)"].idxmax()]
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("WAPE ML · nueve meses", pct(w_all_ml))
                    rc2.metric("WAPE baseline · nueve meses", pct(w_all_base))
                    rc3.metric("Peor mes ML", f"{worst['Mes']} · {worst['LightGBM (%)']:.1f}%")
                    monthly_long = monthly.melt(id_vars="Mes", value_vars=["LightGBM (%)", "Baseline (%)"], var_name="Modelo", value_name="WAPE (%)")
                    chart_monthly = px.bar(monthly_long, x="Mes", y="WAPE (%)", color="Modelo", barmode="group", title="Rendimiento en distintos escenarios estacionales")
                    chart_monthly.update_layout(height=350, margin=dict(l=10,r=10,t=50,b=10), legend_orientation="h")
                    st.plotly_chart(chart_monthly, use_container_width=True)
                    st.dataframe(monthly, hide_index=True, use_container_width=True, column_config={
                        "LightGBM (%)": st.column_config.NumberColumn("WAPE ML", format="%.2f %%"),
                        "Baseline (%)": st.column_config.NumberColumn("WAPE baseline", format="%.2f %%"),
                        "Sesgo ML (%)": st.column_config.NumberColumn("Sesgo firmado", format="%+.2f %%"),
                    })
                    risky = monthly[(monthly["LightGBM (%)"] > 30) | (monthly["Sesgo ML (%)"].abs() > 20)]
                    if not risky.empty:
                        st.warning("Riesgo de régimen estacional: algunos meses superan 30% de WAPE o 20% de sesgo. Revisar antes de utilizar el modelo para decisiones operativas.")
                    st.caption("Métricas calculadas desde pronósticos individuales; filtros de región, tienda y categoría aplicados. El WAPE de nueve meses se pondera por el volumen real, no por el promedio de porcentajes mensuales. No equivale a rendimiento futuro garantizado.")
                    st.download_button("Descargar resultados mensuales (CSV)", monthly.to_csv(index=False).encode("utf-8-sig"), file_name="rolling_validation_filtered.csv", mime="text/csv")
            else:
                st.info("La comparación multimes no está disponible. Ejecuta `retail-black-box rolling` para generar los resultados.")
            st.markdown("**Detalle por categoría — febrero de 2024 (backtest de un mes)**")
            breakdown = []
            for key, group in preds.groupby("category", observed=True):
                breakdown.append({"Categoría": key, "Filas": len(group), "WAPE ML": compute_wape(group["actual"], group["predicted_ml"]) * 100, "WAPE baseline": compute_wape(group["actual"], group["predicted_seasonal_naive"]) * 100})
            breakdown = pd.DataFrame(breakdown).sort_values("WAPE ML")
            st.dataframe(breakdown, hide_index=True, use_container_width=True, column_config={"WAPE ML": st.column_config.NumberColumn(format="%.2f %%"), "WAPE baseline": st.column_config.NumberColumn(format="%.2f %%")})
            st.caption("WAPE expresado como porcentaje. La validación mensual anterior permite estudiar el desempeño estacional; existen periodos de error elevado.")
            st.download_button("Exportar predicciones filtradas (CSV)", preds.to_csv(index=False).encode("utf-8-sig"), file_name="retail_forecast_scoped.csv", mime="text/csv")
            with st.expander("Protocolo de evaluación y limitaciones"):
                st.json(report)
                st.markdown("**Importante:** esto es un backtest de febrero de 2024; no constituye una predicción operativa actual. La mejora en WAPE no demuestra por sí misma ahorro financiero.")

st.divider()
st.caption("Retail Black Box · Evidencia, no inferencias causales · Datos de prueba históricos (2023–2024). Fuentes originales sin alteración.")
