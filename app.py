# ───────── Segmentación de países: K-means (k=3) ─────────
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px, plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             calinski_harabasz_score)

SEED, K = 42, 3
NOMBRES = {0: "Necesidad alta", 1: "Intermedio", 2: "Desarrollado"}
COLORES = {"Necesidad alta": "#d62728", "Intermedio": "#ff9f1c", "Desarrollado": "#2ca02c"}
ORDEN = list(NOMBRES.values())
DESCRIPCION = {
    "Necesidad alta": "Mortalidad infantil y fecundidad altas, ingreso y esperanza de vida bajos. Es el grupo prioritario para la ayuda.",
    "Intermedio": "Indicadores de salud e ingreso en un nivel medio. Es el grupo más numeroso.",
    "Desarrollado": "Mortalidad infantil y fecundidad bajas, ingreso y esperanza de vida altos.",
}

st.set_page_config(page_title="Segmentación de países", layout="wide")

# ───────── Funciones auxiliares y entrenamiento ─────────
def ordenar_etiquetas(labels, X):
    # Renumera los grupos según el PIB per cápita mediano (0 = más bajo, 2 = más alto)
    med = X["gdpp"].groupby(labels).median().sort_values()
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(med.index)}
    return np.array([mapa[l] for l in labels]), mapa

def wcss(Z, lab):
    return sum(((Z[lab == c] - Z[lab == c].mean(0))**2).sum() for c in np.unique(lab))

def metricas(Z, lab):
    return {"Silueta (↑)": silhouette_score(Z, lab), "Davies-Bouldin (↓)": davies_bouldin_score(Z, lab),
            "Calinski-Harabasz (↑)": calinski_harabasz_score(Z, lab), "Inercia (↓)": wcss(Z, lab)}

@st.cache_resource(show_spinner="Entrenando el modelo...")
def entrenar():
    # El modelo se entrena al iniciar la app con los mismos parámetros del laboratorio
    X = pd.read_csv("Country-data.csv").set_index("country")
    feats = list(X.columns)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)

    km = KMeans(n_clusters=K, init="k-means++", n_init=50, random_state=SEED).fit(Z)
    lab_km, mapa = ordenar_etiquetas(km.labels_, X)
    centros = np.array([km.cluster_centers_[v] for v, n in sorted(mapa.items(), key=lambda t: t[1])])

    pca = PCA(n_components=3, random_state=SEED).fit(Z)
    res = X.copy()
    res["Grupo"] = [NOMBRES[l] for l in lab_km]
    res[["PC1", "PC2", "PC3"]] = pca.transform(Z)
    res["silueta"] = silhouette_samples(Z, lab_km)

    # Comparación con el clustering jerárquico (solo para mostrar por qué se eligió K-means)
    lab_w, _ = ordenar_etiquetas(AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Z).labels_, X)
    comp = pd.DataFrame({"K-means (k=3)": metricas(Z, lab_km), "Jerárquico Ward (k=3)": metricas(Z, lab_w)}).T
    comp["Tamaños (necesidad / interm. / desarr.)"] = [str([int((l == i).sum()) for i in range(K)]) for l in (lab_km, lab_w)]

    # Métricas para distintos valores de k (selección de k)
    filas = []
    for k in range(2, 9):
        m = KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=SEED).fit(Z)
        filas.append(dict(k=k, inercia=m.inertia_, silueta=silhouette_score(Z, m.labels_),
                          davies_bouldin=davies_bouldin_score(Z, m.labels_),
                          calinski_harabasz=calinski_harabasz_score(Z, m.labels_)))
    sel = pd.DataFrame(filas).set_index("k")

    return dict(X=X, feats=feats, scaler=scaler, Z=Z, lab=lab_km, centros=centros, pca=pca,
                res=res, comp=comp, sel=sel)

M = entrenar()
X, feats, res = M["X"], M["feats"], M["res"]

def clasificar(vals):
    # Escala el país igual que en el entrenamiento y lo asigna al centroide más cercano
    z = M["scaler"].transform(pd.DataFrame([vals])[feats])[0]
    dist = {NOMBRES[n]: float(np.linalg.norm(z - M["centros"][n])) for n in range(K)}
    grupo = min(dist, key=dist.get)
    return grupo, dist, M["pca"].transform(z.reshape(1, -1))[0]

def grafico_2d(punto=None, etiqueta=""):
    fig = px.scatter(res.reset_index(), x="PC1", y="PC2", color="Grupo", color_discrete_map=COLORES,
                     hover_name="country", category_orders={"Grupo": ORDEN}, height=520)
    fig.update_traces(marker=dict(size=7, opacity=.75))
    if punto is not None:
        fig.add_trace(go.Scatter(x=[punto[0]], y=[punto[1]], mode="markers+text", text=[etiqueta],
                                 textposition="top center", name=etiqueta,
                                 marker=dict(symbol="star", size=18, color="black", line=dict(color="white", width=1))))
    return fig

# ───────── Encabezado, barra lateral y pestaña 1: clasificar ─────────
st.title("Segmentación de países según su nivel de desarrollo")
st.caption("Modelo K-means (k=3) entrenado con 9 indicadores socioeconómicos y de salud de 167 países.")

with st.sidebar:
    st.header("Sobre el modelo")
    st.write("Los países se agrupan en tres niveles de desarrollo. El grupo **Necesidad alta** es el candidato "
             "para priorizar la ayuda de la ONG.")
    st.write("Los grupos se solapan en la frontera, por lo que los casos dudosos deben contrastarse con datos oficiales.")
    st.caption("El dataset no indica el año de los datos.")

t1, t2, t3, t4, t5 = st.tabs(["Clasificar un país", "Mapa de países", "Perfil de los grupos",
                              "Evaluación del modelo", "Países por grupo"])

with t1:
    modo = st.radio("¿Qué quieres hacer?", ["Seleccionar un país existente", "Ingresar datos nuevos"], horizontal=True)
    paises = list(X.index)
    if modo.startswith("Seleccionar"):
        pais = st.selectbox("País", paises, index=paises.index("Peru") if "Peru" in paises else 0)
        vals, etiqueta = X.loc[pais].to_dict(), pais
    else:
        base = st.selectbox("Punto de partida (puedes modificar los valores)", ["Mediana de todos los países"] + paises)
        ref = X.median() if base.startswith("Mediana") else X.loc[base]
        cols = st.columns(3)
        vals = {f: cols[i % 3].number_input(f, value=float(ref[f]), key=f"{f}_{base}") for i, f in enumerate(feats)}
        etiqueta, pais = "País nuevo", None

    grupo, dist, punto = clasificar(vals)
    izq, der = st.columns([1, 1.4])
    with izq:
        st.subheader(f"Grupo asignado: {grupo}")
        st.write(DESCRIPCION[grupo])
        st.write("**Distancia a cada centroide** (en valores estandarizados):")
        st.dataframe(pd.DataFrame({"Distancia": dist}).round(2))
        st.caption("Cuanto menor es la distancia, más cerca está el país de ese grupo. Si dos distancias son parecidas, "
                   "el país está en la frontera entre esos grupos.")
        if pais is not None:
            s = res.loc[pais, "silueta"]
            st.write(f"Silueta de {pais}: **{s:.2f}**" + (" (valor negativo: caso dudoso)" if s < 0 else ""))
        mediana = res.groupby("Grupo")[feats].median().loc[grupo]
        st.write(f"**Comparación con la mediana del grupo '{grupo}':**")
        st.dataframe(pd.DataFrame({etiqueta: pd.Series(vals), "Mediana del grupo": mediana}).round(2))
    with der:
        st.plotly_chart(grafico_2d(punto, etiqueta), width="stretch")
        st.caption("PC1 es un eje de desarrollo (mortalidad y fecundidad frente a esperanza de vida, ingreso y PIB). "
                   "La estrella marca la posición del país consultado.")

# ───────── Pestañas 2 y 3: mapa y perfil de los grupos ─────────
with t2:
    dim = st.radio("Vista", ["2D", "3D"], horizontal=True)
    if dim == "2D":
        st.plotly_chart(grafico_2d(), width="stretch")
    else:
        fig = px.scatter_3d(res.reset_index(), x="PC1", y="PC2", z="PC3", color="Grupo", color_discrete_map=COLORES,
                            hover_name="country", category_orders={"Grupo": ORDEN}, height=650)
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, width="stretch")
    st.caption("Cada punto es un país. Pasa el cursor para ver su nombre. Los tres componentes explican cerca del 76 % "
               "de la variación, por lo que el dibujo es una aproximación.")

with t3:
    perfil = res.groupby("Grupo")[feats].median().loc[ORDEN]
    perfil.insert(0, "n_países", res["Grupo"].value_counts())
    st.write("**Mediana de cada variable por grupo** (valores originales):")
    st.dataframe(perfil.round(2))
    cent = pd.DataFrame(M["centros"], columns=feats, index=pd.Index(ORDEN, name="Grupo")).reset_index()
    cent = cent.melt(id_vars="Grupo", var_name="Variable", value_name="Valor estandarizado")
    fig = px.bar(cent, x="Variable", y="Valor estandarizado", color="Grupo", barmode="group",
                 color_discrete_map=COLORES, category_orders={"Grupo": ORDEN}, height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption("Los centroides muestran cuánto se aleja cada grupo del promedio de todos los países. "
               "Los grupos se diferencian sobre todo en mortalidad infantil, fecundidad, esperanza de vida, ingreso y PIB.")

# ───────── Pestañas 4 y 5: evaluación y listado por grupo ─────────
with t4:
    st.write("**Comparación de modelos** (métricas internas; el modelo desplegado es K-means):")
    st.dataframe(M["comp"].style.format({"Silueta (↑)": "{:.3f}", "Davies-Bouldin (↓)": "{:.2f}",
                                         "Calinski-Harabasz (↑)": "{:.1f}", "Inercia (↓)": "{:.1f}"}))

    st.write("**Selección de k para K-means** (línea naranja: k = 3):")
    sel = M["sel"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, (col, t) in zip(axes, [("inercia", "Inercia (codo) ↓"), ("silueta", "Silueta ↑"),
                                    ("davies_bouldin", "Davies-Bouldin ↓"), ("calinski_harabasz", "Calinski-Harabasz ↑")]):
        ax.plot(sel.index, sel[col], "o-", color="#3b4ea3"); ax.axvline(K, color="#f39c12", ls="--")
        ax.set_title(t); ax.set_xlabel("k")
    plt.tight_layout(); st.pyplot(fig)

    st.write("**Diagrama de silueta de K-means (k=3):**")
    s, lab = res["silueta"].values, M["lab"]
    fig, ax = plt.subplots(figsize=(8, 4.2)); y0 = 5
    for c in range(K):
        v = np.sort(s[lab == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=COLORES[NOMBRES[c]], alpha=.85)
        ax.text(-0.12, y0 + len(v)/2, NOMBRES[c], va="center"); y0 += len(v) + 5
    ax.axvline(s.mean(), color="k", ls="--", label=f"promedio = {s.mean():.3f}"); ax.axvline(0, color="grey", lw=.8)
    ax.set_xlabel("Silueta de cada país"); ax.set_yticks([]); ax.legend(); plt.tight_layout(); st.pyplot(fig)
    st.caption("Cada barra es un país: cuanto más larga hacia la derecha, mejor ubicado está en su grupo. "
               "Las barras con valor negativo son casos fronterizos.")

with t5:
    g = st.selectbox("Grupo", ORDEN)
    sub = res[res["Grupo"] == g][feats]
    st.write(f"{len(sub)} países")
    st.dataframe(sub)
    st.download_button("Descargar CSV", sub.to_csv().encode("utf-8"), f"{g}.csv", "text/csv")
