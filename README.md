# Movie Review Classifier

Clasificador binario de sentimiento (positivo/negativo) para reseñas de películas, construido haciendo *fine-tuning* de **DistilBERT** sobre el dataset **IMDB** con la librería 🤗 Transformers.

## Descripción

El proyecto toma el modelo preentrenado `distilbert-base-uncased` y le agrega una cabeza de clasificación de secuencias (`AutoModelForSequenceClassification`, `num_labels=2`) para predecir si una reseña de película es positiva (`1`) o negativa (`0`).

Todo el flujo de trabajo —carga de datos, tokenización, entrenamiento y evaluación— está implementado en un único notebook: [movie_review_classifier.ipynb](movie_review_classifier.ipynb).

## Dataset

Se utiliza el dataset [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) de Hugging Face:

| Split          | Ejemplos |
|----------------|---------:|
| `train`        |   25.000 |
| `test`         |   25.000 |
| `unsupervised` |   50.000 |

Cada ejemplo tiene un campo `text` (la reseña) y un campo `label` (`0` = negativa, `1` = positiva).

## Modelo y pipeline

1. **Tokenización**: `AutoTokenizer` de `distilbert-base-uncased`, con `padding='max_length'`, `truncation=True` y `max_length=256`.
2. **Modelo**: `distilbert-base-uncased` + cabeza de clasificación (`classifier`, `pre_classifier`) inicializada desde cero para la tarea de 2 clases.
3. **Métrica**: *accuracy*, calculada con la librería `evaluate`.
4. **Entrenamiento**: se usa `Trainer` / `TrainingArguments` de Transformers en dos etapas:
   - Una corrida rápida sobre un subconjunto (1.000 ejemplos de train / 500 de eval) para validar que el pipeline funciona end-to-end (2 épocas, `learning_rate=2e-5`, `batch_size=16`).
   - Una corrida sobre el dataset completo (25.000 train / 25.000 test, 3 épocas) con los mismos hiperparámetros.

### Resultado de la corrida de validación

Con el subconjunto reducido (1.000/500 ejemplos, 2 épocas) se obtuvo:

- `training_loss`: **0.463**
- `train_runtime`: ~70.6s
- `train_samples_per_second`: ~28.3

> Nota: el entrenamiento sobre el dataset completo (celda final del notebook) está implementado pero el notebook actual no incluye las métricas finales de esa corrida ni el guardado del modelo entrenado — ver [Estado actual y próximos pasos](#estado-actual-y-próximos-pasos).

## Requisitos

- Python 3.9+
- Jupyter (Notebook o Lab)
- Dependencias instaladas desde la primera celda del notebook:

```bash
pip install transformers datasets evaluate accelerate
```

Se recomienda además tener PyTorch instalado (requerido por `transformers`) y, si se dispone de GPU, los drivers/CUDA correspondientes para acelerar el entrenamiento.

## Uso

1. Cloná el repositorio e instalá las dependencias:

   ```bash
   pip install transformers datasets evaluate accelerate jupyter
   ```

2. Abrí el notebook:

   ```bash
   jupyter notebook movie_review_classifier.ipynb
   ```

3. Ejecutá las celdas en orden. La primera corrida (dataset descargará ~25MB vía `datasets` y el checkpoint de `distilbert-base-uncased`, ~268MB) puede tardar unos minutos según la conexión.

4. Ajustá los hiperparámetros en `TrainingArguments` (épocas, batch size, learning rate) según el hardware disponible. El entrenamiento sobre el dataset completo es significativamente más pesado que la corrida de validación sobre el subconjunto chico.

## Estructura del proyecto

```
movie-review-classifier/
├── README.md
└── movie_review_classifier.ipynb   # Pipeline completo: datos, tokenización, entrenamiento y evaluación
```

## Estado actual y próximos pasos

Este es un proyecto en desarrollo. Pendientes identificados en el notebook actual:

- [ ] Registrar y reportar las métricas finales (`accuracy`, `loss`) de la corrida sobre el dataset completo.
- [ ] Guardar el modelo y tokenizer entrenados (`trainer.save_model()` / `tokenizer.save_pretrained()`) para poder reutilizarlos sin reentrenar.
- [ ] Agregar un script o celda de inferencia para clasificar reseñas nuevas (por ejemplo con `pipeline("text-classification", ...)`).
- [ ] Congelar las dependencias en un `requirements.txt` o `pyproject.toml` para reproducibilidad.
- [ ] Evaluar el modelo final sobre el split `test` completo y reportar matriz de confusión / métricas adicionales (precision, recall, F1).

## Créditos

- Dataset: [IMDB Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (Maas et al., 2011).
- Modelo base: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) (Sanh et al., 2019).
- Librerías: [🤗 Transformers](https://github.com/huggingface/transformers), [🤗 Datasets](https://github.com/huggingface/datasets), [🤗 Evaluate](https://github.com/huggingface/evaluate).
