import streamlit as st
import pandas as pd

st.title("Dashboard con Streamlit en Colab 🚀")

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    st.write("Resumen estadístico:")
    st.write(df.describe())

    columns = df.select_dtypes(include=['number']).columns
    if len(columns) > 0:
        x_axis = st.selectbox("Selecciona la variable del eje X", columns)
        y_axis = st.selectbox("Selecciona la variable del eje Y", columns)
        st.write("Gráfico de dispersión:")
        st.scatter_chart(df[[x_axis, y_axis]])
    else:
        st.warning("No hay columnas numéricas para graficar.")
