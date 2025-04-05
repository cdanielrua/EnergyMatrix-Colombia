import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Dashboard de Predicción", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("databproduction_clean.csv")
        df_predictions = pd.read_csv("predicciones_energia.csv")
        return df, df_predictions
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame(), pd.DataFrame()

df, df_predictions = load_data()

# Cargar modelo entrenado
@st.cache_resource
def load_model():
    try:
        with open('/home/gerardo/proyecto_energia/notebooks/best_gb_model.pkl', "rb") as file:
            model = pickle.load(file)

        if isinstance(model, np.ndarray):
            raise ValueError("El archivo cargado no es un modelo válido, sino un array de NumPy.")

        return model    
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

# Lista de países originales
paises_originales = ["Colombia", "Brazil", "United States"]

# Sidebar para filtros
st.sidebar.image('/home/gerardo/proyecto_energia/source/logo.png.jpg', width=150)  # Asegurar que el nombre de la imagen es correcto
st.sidebar.header("Filtros")

if not df.empty:
    selected_country = st.sidebar.selectbox("Selecciona un país", paises_originales)
    selected_product = st.sidebar.selectbox("Selecciona un producto", df["PRODUCT"].unique())
else:
    st.sidebar.error("Los datos no se cargaron correctamente.")
    selected_country, selected_product = None, None

# Filtrar datos
if selected_country and selected_product:
    df = df[df["YEAR"] >= 2017]
    df_predictions = df_predictions[df_predictions["YEAR"] >= 2017]
    filtered_df = df[(df["COUNTRY"] == selected_country) & (df["PRODUCT"] == selected_product)]
    filtered_predictions = df_predictions[(df_predictions["COUNTRY"] == selected_country) & (df_predictions["PRODUCT"] == selected_product)]

    # Escalar valores
    scaler = MinMaxScaler()
    if not filtered_df.empty:
        filtered_df["SCALED_VALUE"] = scaler.fit_transform(filtered_df[["VALUE"]])
    if not filtered_predictions.empty:
        filtered_predictions["SCALED_PREDICTED_VALUE"] = scaler.fit_transform(filtered_predictions[["PREDICTED_VALUE"]])

    # Visualización de datos históricos
    st.subheader(f"Datos históricos de {selected_product} en {selected_country}")
    fig = px.line(filtered_df, x="YEAR", y="SCALED_VALUE", title=f"Tendencia de {selected_product} en {selected_country} (Escalado)")
    st.plotly_chart(fig)
    st.markdown("Este gráfico muestra la tendencia histórica de la producción de energía para el producto seleccionado en el país elegido.")

    # Visualización de predicciones
    st.subheader(f"Predicciones de {selected_product} en {selected_country}")
    fig = px.line(filtered_predictions, x="MONTH", y="SCALED_PREDICTED_VALUE", color="YEAR", markers=True, title=f"Predicción de {selected_product} en {selected_country} (Escalado)")
    st.plotly_chart(fig)
    st.markdown("Este gráfico muestra la predicción de producción de energía en los próximos meses basada en los datos históricos.")

    # Comparación entre países
    st.subheader("Comparación entre países")
    comparison_df = df[(df["PRODUCT"] == selected_product) & (df["COUNTRY"].isin(paises_originales))]
    comparison_df["SCALED_VALUE"] = scaler.fit_transform(comparison_df[["VALUE"]])
    fig = px.line(comparison_df, x="YEAR", y="SCALED_VALUE", color="COUNTRY", title=f"Comparación de {selected_product} entre {', '.join(paises_originales)} (Escalado)")
    st.plotly_chart(fig)
    st.markdown("Este gráfico compara la producción de energía del producto seleccionado en los tres países disponibles.")

    # Predicción personalizada
    st.sidebar.subheader("Predicción personalizada")
    year_input = st.sidebar.number_input("Año", min_value=2017, max_value=2026, step=1)
    month_input = st.sidebar.number_input("Mes", min_value=1, max_value=12, step=1)

    if st.sidebar.button("Predecir"):
        if model and not filtered_predictions.empty:
            try:
                # Filtrar para obtener datos codificados
                filtered_predictions = filtered_predictions[
                    (filtered_predictions["YEAR"] == year_input) & 
                    (filtered_predictions["MONTH"] == month_input)
                ]

                if "COUNTRY_encoded" in filtered_predictions.columns and "PRODUCT_encoded" in filtered_predictions.columns:
                    country_encoded = filtered_predictions["COUNTRY_encoded"].iloc[0]
                    product_encoded = filtered_predictions["PRODUCT_encoded"].iloc[0]
                else:
                    st.sidebar.error("Faltan datos codificados para predecir.")
                    country_encoded, product_encoded = None, None

                if country_encoded is not None and product_encoded is not None:
                    input_data = pd.DataFrame([[country_encoded, product_encoded, year_input, month_input]],
                                              columns=["COUNTRY_encoded", "PRODUCT_encoded", "YEAR", "MONTH"])
                    input_data = input_data.astype(float)
                    prediction = model.predict(input_data)[0]

                    # Agregar la predicción al dataframe
                    new_row = pd.DataFrame([[year_input, month_input, prediction, selected_country, selected_product]],
                                           columns=["YEAR", "MONTH", "SCALED_PREDICTED_VALUE", "COUNTRY", "PRODUCT"])
                    filtered_predictions = pd.concat([filtered_predictions, new_row], ignore_index=True)

                    st.sidebar.success(f"Predicción: {prediction:.2f} unidades")

                    # Actualizar la gráfica de predicción con la nueva predicción
                    fig = px.line(filtered_predictions, x="MONTH", y="SCALED_PREDICTED_VALUE", color="YEAR", markers=True,
                                  title=f"Predicción de {selected_product} en {selected_country} (Escalado)")
                    st.plotly_chart(fig)

            except Exception as e:
                st.sidebar.error(f"Error en la predicción: {e}")
        else:
            st.sidebar.error("No hay datos o modelo disponibles para la predicción.")
