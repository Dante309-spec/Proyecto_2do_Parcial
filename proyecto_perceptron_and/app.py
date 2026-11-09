import streamlit as st
import pandas as pd

from perceptron import entrenar, predecir, AND_DATA
from pdf_generator import generar_pdf


# ---------------------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# ---------------------------------------

st.set_page_config(
    page_title="Perceptrón AND",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Simulador del Perceptrón Monocapa")
st.caption(
    "Entrenamiento e inferencia de la compuerta lógica AND"
)


# ---------------------------------------
# 1. ESQUEMA DE LA RED NEURONAL
# ---------------------------------------

st.subheader("1. Esquema de la red neuronal")

st.markdown("""
<div style="
    background:#eef4f8;
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:#173b57;
    font-size:18px;
">

<b>Entradas</b><br><br>

x₁ ── (w₁) ──┐<br>

x₂ ── (w₂) ──┼──► <b>Σ + b</b>
──► <b>Función escalón</b>
──► <b>Salida y</b><br>

Bias b ──────┘

</div>
""", unsafe_allow_html=True)


# ---------------------------------------
# 2. HIPERPARÁMETROS INICIALES
# ---------------------------------------

st.subheader("2. Hiperparámetros iniciales")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    w1 = st.number_input(
        "Peso w1",
        value=0.1,
        step=0.1,
        format="%.3f"
    )

with c2:
    w2 = st.number_input(
        "Peso w2",
        value=0.3,
        step=0.1,
        format="%.3f"
    )

with c3:
    bias = st.number_input(
        "Bias b",
        value=0.2,
        step=0.1,
        format="%.3f"
    )

with c4:
    eta = st.number_input(
        "Aprendizaje η",
        min_value=0.001,
        value=0.1,
        step=0.01,
        format="%.3f"
    )

with c5:
    max_epocas = st.number_input(
        "Máximo de épocas",
        min_value=1,
        max_value=100,
        value=4,
        step=1
    )


# ---------------------------------------
# 3. CONJUNTO DE ENTRENAMIENTO
# ---------------------------------------

st.markdown("### Conjunto de entrenamiento AND")

df_datos = pd.DataFrame(
    AND_DATA,
    columns=["x1", "x2", "Objetivo"]
)

st.dataframe(
    df_datos,
    hide_index=True,
    use_container_width=True
)


# ---------------------------------------
# 4. ENTRENAMIENTO DEL PERCEPTRÓN
# ---------------------------------------

if st.button(
    "▶ Entrenar perceptrón",
    type="primary"
):

    resultado = entrenar(
        w1,
        w2,
        bias,
        eta,
        int(max_epocas)
    )

    st.session_state["resultado"] = resultado

    st.session_state["parametros"] = {
        "w1": w1,
        "w2": w2,
        "b": bias,
        "eta": eta,
        "max_epocas": int(max_epocas)
    }


# ---------------------------------------
# 5. MOSTRAR RESULTADOS
# ---------------------------------------

if "resultado" in st.session_state:

    resultado = st.session_state["resultado"]

    parametros = st.session_state["parametros"]

    st.success(
        f"Entrenamiento terminado: "
        f"{len(resultado['epocas'])} época(s) ejecutada(s)."
    )


    # -----------------------------------
    # TABLA RESUMEN POR ÉPOCAS
    # -----------------------------------

    st.subheader("3. Resumen por épocas")

    df_epocas = pd.DataFrame(
        resultado["epocas"]
    )

    df_epocas = df_epocas.rename(
        columns={
            "epoca": "Época",
            "w1": "w1 final",
            "w2": "w2 final",
            "b": "Bias final",
            "errores": "Errores"
        }
    )

    st.dataframe(
        df_epocas,
        hide_index=True,
        use_container_width=True
    )


    # -----------------------------------
    # PROCEDIMIENTO MATEMÁTICO
    # -----------------------------------

    st.subheader(
        "4. Procedimiento matemático paso a paso"
    )

    for p in resultado["pasos"]:

        with st.expander(
            f"Época {p['epoca']} — "
            f"Entrada ({p['x1']}, {p['x2']})"
        ):

            # Suma ponderada
            st.markdown("**1. Suma ponderada**")

            st.latex(
                rf"""
                z = ({p['x1']} \cdot {p['w1_antes']:.4f})
                + ({p['x2']} \cdot {p['w2_antes']:.4f})
                + {p['b_antes']:.4f}
                = {p['z']:.4f}
                """
            )

            # Función escalón y error
            st.markdown(
                "**2. Función de activación y error**"
            )

            st.latex(
                rf"""
                y=f(z)={p['salida']}
                """
            )

            st.latex(
                rf"""
                e=t-y={p['objetivo']}-{p['salida']}
                ={p['error']}
                """
            )

            # Actualización del peso 1
            st.markdown("**3. Actualización de w1**")

            st.latex(
                rf"""
                w_1'={p['w1_antes']:.4f}
                +{parametros['eta']}\cdot({p['error']})
                \cdot{p['x1']}
                ={p['w1_despues']:.4f}
                """
            )

            # Actualización del peso 2
            st.markdown("**4. Actualización de w2**")

            st.latex(
                rf"""
                w_2'={p['w2_antes']:.4f}
                +{parametros['eta']}\cdot({p['error']})
                \cdot{p['x2']}
                ={p['w2_despues']:.4f}
                """
            )

            # Actualización del bias
            st.markdown("**5. Actualización del bias**")

            st.latex(
                rf"""
                b'={p['b_antes']:.4f}
                +{parametros['eta']}\cdot({p['error']})
                ={p['b_despues']:.4f}
                """
            )


    # -----------------------------------
    # 6. MÓDULO DE INFERENCIA
    # -----------------------------------

    st.subheader("5. Módulo de inferencia / prueba")

    st.write(
        "Selecciona una combinación de entradas "
        "para obtener una predicción."
    )

    a, bcol = st.columns(2)

    with a:
        x1_test = st.selectbox(
            "Entrada x1",
            [0, 1],
            key="x1_test"
        )

    with bcol:
        x2_test = st.selectbox(
            "Entrada x2",
            [0, 1],
            key="x2_test"
        )

    pred = predecir(
        x1_test,
        x2_test,
        resultado["w1"],
        resultado["w2"],
        resultado["b"]
    )

    st.metric(
        "Salida predicha",
        pred["salida"]
    )

    st.write(
        f"Suma ponderada: z = {pred['z']:.4f}"
    )

    st.caption(
        "La salida se actualiza al cambiar cualquiera "
        "de las entradas."
    )


    # -----------------------------------
    # 7. EXPORTACIÓN A PDF
    # -----------------------------------

    st.subheader("6. Exportar resultados")

    pdf_bytes = generar_pdf(
        resultado,
        parametros
    )

    st.download_button(
        label="⬇ Descargar informe PDF",
        data=pdf_bytes,
        file_name="informe_perceptron_and.pdf",
        mime="application/pdf"
    )


else:

    st.info(
        "Configura los parámetros y presiona "
        "«Entrenar perceptrón» para comenzar."
    )