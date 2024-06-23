# NLP ToolBox :rocket:

NLP ToolBox es una suite completa de herramientas de Procesamiento de Lenguaje Natural (NLP) que incluye análisis de sentimientos, traducción, respuesta a preguntas y resumen de textos. Esta aplicación está construida utilizando Streamlit y modelos preentrenados de Hugging Face Transformers.

## Descripción

NLP ToolBox permite a los usuarios realizar varias tareas de NLP en reseñas de automóviles u otros textos ingresados. Las funcionalidades incluyen:
- **Análisis de Sentimientos**: Detecta si una reseña es positiva o negativa.
- **Traducción**: Traduce texto del inglés al español.
- **Respuesta a Preguntas**: Responde preguntas basadas en un contexto proporcionado.
- **Resumir Textos**: Genera un resumen conciso de textos largos.

## Instalación

Para instalar y configurar NLP ToolBox, sigue estos pasos:

1. **Clona el repositorio:**

    ```bash
    git clone https://github.com/adrianinfantes/NLPToolBox.git
    cd NLPToolBox
    ```

2. **Instala las dependencias usando Poetry:**

    ```bash
    poetry install
    ```

3. **Configura Streamlit:**

    Asegúrate de que `Streamlit` esté instalado y configurado correctamente. Puedes instalarlo usando Poetry o pip:

    ```bash
    poetry add streamlit
    ```

4. **Configura el archivo `pyproject.toml`:**

    Actualiza el nombre del proyecto en `pyproject.toml` si es necesario.

    ```toml
    [tool.poetry]
    name = "nlp-toolbox"
    version = "0.1.0"
    description = "A comprehensive suite of NLP tools including sentiment analysis, translation, question answering, and summarization."
    authors = ["Your Name <your.email@example.com>"]
    ```

## Uso

Para ejecutar la aplicación Streamlit, navega al directorio del proyecto y ejecuta el siguiente comando:

```bash
streamlit run app.py
