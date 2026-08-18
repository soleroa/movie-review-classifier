# Movie Review Classifier

Binary sentiment classifier (positive/negative) for movie reviews, built by *fine-tuning* **DistilBERT** on the **IMDB** dataset with the 🤗 Transformers library.

## Description

The project takes the pretrained `distilbert-base-uncased` model and adds a sequence classification head (`AutoModelForSequenceClassification`, `num_labels=2`) to predict whether a movie review is positive (`1`) or negative (`0`).

The whole workflow —data loading, tokenization, training, and evaluation— is implemented in a single notebook: [movie_review_classifier.ipynb](movie_review_classifier.ipynb).

## Dataset

The [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) dataset from Hugging Face is used:

| Split          | Examples |
|----------------|---------:|
| `train`        |   25,000 |
| `test`         |   25,000 |
| `unsupervised` |   50,000 |

Each example has a `text` field (the review) and a `label` field (`0` = negative, `1` = positive).

## Model and pipeline

1. **Tokenization**: `AutoTokenizer` from `distilbert-base-uncased`, with `padding='max_length'`, `truncation=True`, and `max_length=256`.
2. **Model**: `distilbert-base-uncased` + classification head (`classifier`, `pre_classifier`) initialized from scratch for the 2-class task.
3. **Metric**: *accuracy*, computed with the `evaluate` library.
4. **Training**: `Trainer` / `TrainingArguments` from Transformers is used in two stages:
   - A quick run on a subset (1,000 train / 500 eval examples) to validate that the pipeline works end-to-end (2 epochs, `learning_rate=2e-5`, `batch_size=16`).
   - A run on the full dataset (25,000 train / 25,000 test, 3 epochs) with the same hyperparameters.

### Validation run result

With the reduced subset (1,000/500 examples, 2 epochs) the following was obtained:

- `training_loss`: **0.463**
- `train_runtime`: ~70.6s
- `train_samples_per_second`: ~28.3

> Note: training on the full dataset (final cell of the notebook) is implemented, but the current notebook does not include the final metrics for that run nor does it save the trained model — see [Current status and next steps](#current-status-and-next-steps).

## Requirements

- Python 3.9+
- Jupyter (Notebook or Lab)
- Dependencies installed from the first cell of the notebook:

```bash
pip install transformers datasets evaluate accelerate
```

It is also recommended to have PyTorch installed (required by `transformers`) and, if a GPU is available, the corresponding drivers/CUDA to speed up training.

## Usage

1. Clone the repository and install the dependencies:

   ```bash
   pip install transformers datasets evaluate accelerate jupyter
   ```

2. Open the notebook:

   ```bash
   jupyter notebook movie_review_classifier.ipynb
   ```

3. Run the cells in order. The first run (the dataset will download ~25MB via `datasets` and the `distilbert-base-uncased` checkpoint, ~268MB) may take a few minutes depending on your connection.

4. Adjust the hyperparameters in `TrainingArguments` (epochs, batch size, learning rate) according to the available hardware. Training on the full dataset is significantly heavier than the validation run on the small subset.

## Project structure

```
movie-review-classifier/
├── README.md
└── movie_review_classifier.ipynb   # Full pipeline: data, tokenization, training, and evaluation
```

## Current status and next steps

This is a work-in-progress project. Identified pending items in the current notebook:

- [ ] Record and report the final metrics (`accuracy`, `loss`) for the run on the full dataset.
- [ ] Save the trained model and tokenizer (`trainer.save_model()` / `tokenizer.save_pretrained()`) to be able to reuse them without retraining.
- [ ] Add an inference script or cell to classify new reviews (e.g. with `pipeline("text-classification", ...)`).
- [ ] Pin dependencies in a `requirements.txt` or `pyproject.toml` for reproducibility.
- [ ] Evaluate the final model on the full `test` split and report a confusion matrix / additional metrics (precision, recall, F1).

## Credits

- Dataset: [IMDB Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (Maas et al., 2011).
- Base model: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) (Sanh et al., 2019).
- Libraries: [🤗 Transformers](https://github.com/huggingface/transformers), [🤗 Datasets](https://github.com/huggingface/datasets), [🤗 Evaluate](https://github.com/huggingface/evaluate).
