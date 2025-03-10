# 💡 Clasificación Binaria con NLP 🧩

![Estado](https://img.shields.io/badge/Estado-Completado-brightgreen)

:octocat:

**Estudiante**: Carlos Maximiliano García Medina

**Tarea 3**: Red Neuronal con Numpy

**Materia**: Sistemas de Visión Artificial  

## 📝 Descripción
Este repositorio contiene la implementación de un modelo de clasificación binaria utilizando Procesamiento de Lenguaje Natural (NLP) con el dataset IMDB. El proyecto incluye:

 * 🧩 Carga y preprocesamiento de datos: Carga del dataset IMDB y vectorización de las reseñas.

 * 🛠️ Construcción del modelo: Implementación de una red neuronal densa para clasificación binaria.

 * 🚀 Entrenamiento y evaluación: Entrenamiento del modelo y visualización de los resultados.

El código está comentado paso a paso para una mayor comprensión.
---
##  📋 Requisitos
Para ejecutar este proyecto, necesitas tener instaladas lo siguiente:
- **Python**: 
- [Python 3.9 hasta Python 3.12](https://www.python.org/downloads/)

Se puede comprobar la versión de python empleando el comando en terminal:

**IMPORTANTE:** Se requiere instalar ese rango de versiones debido a que tensorflow solo se puede emplear en esas versiones de python (https://www.tensorflow.org/install/pip?hl=es).

**En PowerShell:**
  ```
  python --version
  ```

**En Unix**
  ```
  python3 --version
  ```

Librerías (dependencias):
* NumPy: Para cálculos numéricos y manejo de arreglos.
* Matplotlib: Para la generación de gráficas.
* TensorFlow/Keras: Para construir y entrenar la red neuronal.

Puedes instalar en conjunto estas dependencias utilizando `pip`:

```bash
pip install numpy matplotlib tensorflow
```
**Nota:** Si en Unix (Linux) no funciona, emplea ```pip3```
---
## 🗂️ Estructura del Proyecto
El proyecto está organizado de la siguiente manera:
    ```
    BINARY_CLASSIFICATION_NLP_2230002/
    │
    ├── src/
    │   └── Binary_C_NLP.py  # Script principal del proyecto
    │
    ├── venv/                # Entorno virtual
    ├── .gitignore           # Archivo para ignorar archivos no deseados
    ├── main.py              # Script principal para ejecutar el proyecto
    ├── README.md            # Este archivo
    └── requirements.txt     # Lista de dependencias del proyecto
    ```
--- 
## 🚀 ¿Cómo usar este repositorio?
Sigue estos pasos para ejecutar el proyecto en tu lab:

**Clona el repositorio 🖥️:**
Abre una terminal y ejecuta el siguiente comando para clonar el repositorio en tu computadora:

    git clone https://github.com/MaxGm07/Binary_classification_NLP_2230002

**Cree un nuevo entorno virtual**
Se recomienda tener el entorno virtual generado en la carpeta principal para un fácil acceso, su activación y desactivación se realiza de la siguiente forma:

En PowerShell:
    .\venv\Scripts\Activate
    deactivate

En Unix: 
    source venv/bin/activate
    deactivate

**Instala las dependencias 📦:**
Asegúrate de tener instaladas las bibliotecas necesarias. Ejecuta el siguiente comando para instalarlas:

    pip install -r requirements.txt 

**Ejecuta el script principal🚀:**
Para entrenar y evaluar el modelo, ejecuta:

    python main.py

**Visualiza los resultados 📊:**
* El script mostrará la pérdida y precisión durante el entrenamiento en la consola.

* También se mostrará un gráfico con la pérdida y precisión en entrenamiento y validación.
---
## 🛠️ Tecnologías Utilizadas**
**Python:** Lenguaje de programación principal en este caso se utilizó la versión 3.11 para el desarrollo del proyecto.

**NumPy:** Para cálculos numéricos y manejo de arreglos.

**Matplotlib:** Para visualización de datos y gráficos.

**Keras:** Para construir y entrenar el modelo de red neuronal.

**TensorFlow:** Como backend para Keras.
---
## Explicación del código
El código realiza lo siguiente:

1. Carga los datos del dataset IMDB, que contiene reseñas de películas etiquetadas como positivas (1) o negativas (0).

2. Decodifica una reseña de ejemplo para mostrar cómo se ve una reseña en texto plano.

3. Vectoriza las secuencias de palabras para convertirlas en vectores binarios que pueden ser procesados por la red neuronal.

4. Construye el modelo de red neuronal con las siguientes capas:

   * Una capa densa de 64 neuronas con activación ReLU.

   * Una capa densa de 16 neuronas con activación ReLU.

   * Una capa de salida de 1 neurona con activación sigmoide para la clasificación binaria.

5. Compila el modelo utilizando el optimizador RMSprop, la función de pérdida de entropía cruzada binaria y la métrica de precisión.

6. Entrena el modelo con los datos de entrenamiento y valida con un conjunto de validación.

7. Visualiza los resultados del entrenamiento, mostrando la pérdida y precisión en entrenamiento y validación.

8. Evalúa el modelo en el conjunto de prueba y muestra la pérdida y precisión.

9. Realiza una predicción en las dos primeras muestras del conjunto de prueba.

## Notas extra
¡Gracias por llegar al final del readme, agradezco tu tiempo, espero tengas un buen día!