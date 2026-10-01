import joblib, pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title="Segmentación de países", layout="wide")

@st.cache_resource
def cargar():
    return joblib.load("modelo_paises.joblib"), pd.read_csv("Country-data.csv")

art, df = cargar()
scaler, km, mapa, feats = art["scaler"], art["km"], art["mapa"], art["feats"]
nombres, colores, pca = art["nombres"], art["colores"], art["pca"]
orden = list(nombres.values())

X = df.set_index("country")[feats]
Z = scaler.transform(X)
res = X.copy()
res["Grupo"] = [nombres[mapa[l]] for l in km.predict(Z)]
res[["PC1", "PC2", "PC3"]] = pca.transform(Z)

st.title("Segmentación de países según su nivel de desarrollo")
st.caption("Modelo K-means (k=3) entrenado con 9 indicadores socioeconómicos y de salud.")
t1, t2, t3 = st.tabs(["Clasificar un país", "Países en 3D", "Países por grupo"])

with t1:
    st.write("Ingresa los indicadores (puedes partir de un país existente y modificarlos).")

    
    base = st.selectbox("País de partida", ["Mediana de todos los países"] + list(X.index))
    ref = X.median() if base.startswith("Mediana") else X.loc[base]
    cols = st.columns(3)
    vals = {f: cols[i % 3].number_input(f, value=float(ref[f]), key=f"{f}_{base}") for i, f in enumerate(feats)}
    if st.button("Clasificar", type="primary"):
        nuevo = pd.DataFrame([vals])[feats]
        g = nombres[mapa[int(km.predict(scaler.transform(nuevo))[0])]]
        st.subheader(f"Grupo asignado: {g}")
        mediana = res.groupby("Grupo")[feats].median().loc[g]
        st.write("Comparación de tus valores con la mediana del grupo asignado:")
        st.dataframe(pd.DataFrame({"Tus valores": pd.Series(vals), f"Mediana de '{g}'": mediana}))

with t2:
    fig = px.scatter_3d(res.reset_index(), x="PC1", y="PC2", z="PC3", color="Grupo", color_discrete_map=colores,
                        hover_name="country", category_orders={"Grupo": orden}, height=650)
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, width="stretch")
    st.caption("PC1 es un eje de desarrollo (mortalidad y fecundidad vs. esperanza de vida, ingreso y PIB).")

with t3:
    g = st.selectbox("Grupo", orden)
    sub = res[res["Grupo"] == g][feats]
    st.write(f"{len(sub)} países")
    st.dataframe(sub)
    st.download_button("Descargar CSV", sub.to_csv().encode("utf-8"), f"{g}.csv", "text/csv")
