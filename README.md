# AI Image Analysis Agent

Un agente de inteligencia artificial basado en LangGraph para analizar imágenes de dashboards y detectar anomalías usando GPT-4o Vision.

## Características

- **Análisis de Imágenes**: Utiliza GPT-4o Vision para analizar dashboards y detectar anomalías
- **Workflow con LangGraph**: Implementa un flujo de trabajo robusto y escalable
- **Persistencia de Datos**: Guarda resultados en formato JSON con metadatos
- **Gestión de Archivos**: Mueve automáticamente las imágenes procesadas
- **Monitoreo**: Incluye herramientas de monitoreo y reportes
- **Docker Ready**: Configuración completa para despliegue en contenedores

## Estructura del Proyecto

```
ai_services_compliance/
├── src/
│   ├── __init__.py
│   ├── agent.py          # Agente principal con LangGraph
│   └── main.py           # Script de ejecución principal
├── assets/
│   └── images/           # Imágenes a analizar
│       └── processed/    # Imágenes ya procesadas
├── envs/
│   ├── data/             # Resultados de análisis (JSON)
│   ├── run_analysis.sh   # Script de ejecución
│   ├── monitoring.py     # Herramientas de monitoreo
│   ├── Dockerfile        # Configuración Docker
│   └── docker-compose.yml # Orquestación Docker
├── requirements.txt      # Dependencias Python
├── .env                 # Variables de entorno (crear manualmente)
└── README.md
```

## Configuración

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar API Key

Crear archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=tu_api_key_aqui
```

### 3. Preparar Imágenes

Colocar las imágenes a analizar en la carpeta `assets/images/`.

## Uso

### Ejecución Local

```bash
# Usando el script shell
./envs/run_analysis.sh

# O directamente con Python
python src/main.py
```

### Ejecución con Docker

```bash
# Construir y ejecutar con docker-compose
cd envs
docker-compose up --build

# O construir manualmente
docker build -f envs/Dockerfile -t ai-agent .
docker run -e OPENAI_API_KEY=tu_key ai-agent
```

## Funcionamiento del Agente

1. **Lectura de Imágenes**: Lee todas las imágenes de `assets/images/`
2. **Análisis con GPT-4o**: Analiza cada imagen buscando:
   - Estado general del sistema
   - Métricas y variables importantes
   - Anomalías detectadas
   - Información relevante
3. **Persistencia**: Guarda resultados en `envs/data/` con formato:
   ```json
   {
     "id": "analysis_20241201_143022_image1",
     "file": "dashboard1.png",
     "content": {
       "estado_general": "normal",
       "metricas_principales": ["CPU", "Memory", "Network"],
       "anomalias_detectadas": ["High CPU usage"],
       "informacion_relevante": "Dashboard shows normal operation",
       "resumen": "System operating normally with minor CPU spike"
     },
     "datetime": "2024-12-01T14:30:22"
   }
   ```
4. **Movimiento de Archivos**: Mueve imágenes procesadas a `assets/images/processed/`

## Monitoreo

Para generar reportes de monitoreo:

```bash
python envs/monitoring.py
```

## Herramientas del Agente

El agente incluye las siguientes herramientas:

- **Análisis de Imágenes**: GPT-4o Vision para análisis detallado
- **Gestión de Archivos**: Movimiento automático de archivos procesados
- **Persistencia de Datos**: Almacenamiento estructurado de resultados
- **Manejo de Errores**: Gestión robusta de errores y excepciones
- **Logging**: Registro detallado de operaciones

## Formato de Salida

Los análisis se guardan en archivos JSON con la siguiente estructura:

```json
{
  "id": "identificador_unico",
  "file": "nombre_archivo_original",
  "content": {
    "estado_general": "normal|advertencia|critico",
    "metricas_principales": ["lista", "de", "metricas"],
    "anomalias_detectadas": ["lista", "de", "anomalias"],
    "informacion_relevante": "descripción detallada",
    "resumen": "resumen ejecutivo"
  },
  "datetime": "2024-12-01T14:30:22"
}
```

## Requisitos

- Python 3.8+
- OpenAI API Key
- Dependencias listadas en `requirements.txt`

## Licencia

Este proyecto está diseñado para uso interno y de cumplimiento de servicios de IA.
