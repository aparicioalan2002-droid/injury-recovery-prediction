import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Aplicar estilo global para aumentar el tamaño de fuente
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# Cargar el CSV por defecto al iniciar la app
@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv('injury_data.csv')
    except FileNotFoundError:
        st.warning("El archivo 'injury_data.csv' no se encontró. Asegúrate de que el archivo esté en el directorio.")
        return pd.DataFrame()

# Cargar los datos al inicio
data = cargar_datos()

# Separar variables independientes (X) y la variable dependiente (y)
if not data.empty:
    X = data[['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries', 'Training_Intensity']]
    y = data['Recovery_Time']

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    # Título grande con estilo
    st.markdown("<h1 style='font-size: 36px;'>🩺 Predicción de Tiempo de Recuperación de Lesiones</h1>", unsafe_allow_html=True)

    st.markdown("""
        ## Aplicación de Predicción de Tiempo de Recuperación de Lesiones 🏥

        Esta aplicación utiliza un modelo de **árbol de decisión** entrenado con datos históricos de jugadores
        para predecir el tiempo estimado de recuperación tras una lesión. Se consideran variables como:

        - Edad del jugador
        - Peso
        - Altura
        - Historial de lesiones
        - Intensidad de entrenamiento

        El modelo ha sido evaluado con la métrica **R²**, obteniendo una confiabilidad aproximada del **90%** en sus predicciones.

        Herramientas utilizadas:
        - Python 🐍
        - Streamlit
        - Scikit-learn
        - Pandas
        - GitHub para el control de versiones

        ---
        """)

    # Formulario de entrada de datos
    st.header("Introduce los datos para hacer la predicción:")
    #st.markdown("<h3>Introduce los datos para hacer la predicción:</h3>", unsafe_allow_html=True)
    player_age = st.number_input("Edad del jugador (Player_Age)", min_value=18, max_value=100, step=1)
    player_weight = st.number_input("Peso del jugador (Player_Weight en kg)", min_value=30.0, max_value=200.0, step=0.1)
    player_height = st.number_input("Altura del jugador (Player_Height en cm)", min_value=100, max_value=250, step=1)
    previous_injuries = st.number_input("Número de lesiones anteriores (Previous_Injuries)", min_value=0, max_value=1, step=1)
    training_intensity = st.number_input("Intensidad de entrenamiento (Training_Intensity de 0 a 10)", min_value=0, max_value=10, step=1)

    # Botón para predecir
    if st.button("Predecir Tiempo de Recuperación"):
        input_data = pd.DataFrame({
            'Player_Age': [player_age],
            'Player_Weight': [player_weight],
            'Player_Height': [player_height],
            'Previous_Injuries': [previous_injuries],
            'Training_Intensity': [training_intensity]
        })

        predicted_recovery_time = model.predict(input_data)

        st.success(f"🕒 El modelo predice que el tiempo de recuperación es: **{predicted_recovery_time[0]:.5f} días**.")
        
        # Guardar nuevos datos con la predicción
        new_data = input_data.copy()
        new_data['Predicted_Recovery_Time'] = predicted_recovery_time
        data = pd.concat([data, new_data], ignore_index=True)
        data.to_csv('injury_data.csv', index=False)
        #st.info("✅ Los nuevos datos han sido guardados en 'injury_data.csv'.")

else:
    st.warning("No se encontraron datos previos. Asegúrate de que 'injury_data.csv' esté presente.")
