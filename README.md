# Email Spam/Ham Classifier

A modular, reproducible machine learning pipeline for classifying emails as **Spam** (`1`) or **Ham** (`0`) using regex text normalization, TF-IDF vectorization, and Multinomial Naive Bayes.

---

## 1. Project Structure

```text
spam-classifier/
├── config/
│   └── config.yaml                 # Centralized paths, column names, and hyperparameters
│
├── dataset/
│   ├── raw/                        # Place raw input CSV here
│   └── processed/                  # Auto-generated cleaned dataset
│
├── artifacts/
│   ├── spam_classifier.joblib      # Serialized end-to-end preprocessing + model pipeline
│   ├── metrics.json                # Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
│   ├── confusion_matrix.png        # Saved confusion matrix heatmap
│   └── error_analysis.csv          # Misclassified test samples with predicted probabilities
│
├── src/
│   └── spam_classifier/
│       ├── __init__.py
│       ├── config.py               # Typed YAML configuration loader
│       ├── exceptions.py           # Domain-specific exception hierarchy
│       ├── logger.py               # Timestamped console logging
│       ├── preprocessing.py        # Alphanumeric regex cleaner & batch transformer
│       ├── data.py                 # Schema validation, stats logging, empty-row filtering
│       ├── model.py                # Pipeline construction, evaluation, and artifact export
│       ├── train.py                # Reproducible training & evaluation CLI entrypoint
│       └── predict.py              # Standalone inference service & CLI entrypoint
│
├── tests/
│   ├── test_preprocessing.py       # Unit tests for regex normalization edge cases
│   ├── test_data.py                # Data contract and label validation tests
│   └── test_pipeline.py            # End-to-end training, serialization, and inference tests
│
└── README.md
```

---

## 2. Setup & Installation

### Note
Running this project requires Python Interpreter, pip and python-venv installed

### 1. Clone the repository and enter the project directory

```bash
git clone https://github.com/shreeharipsychx86/week1-training.git
cd spam-classifier
```

### 2. Create and activate a virtual environment

#### For Windows
```bash
python -m venv .venv
source .venv/bin/activate
```

#### For Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies within the virtual environment

```bash
pip install -r requirements.txt
```

### 4. Place the raw dataset

Ensure the dataset is located at the path configured in `config/config.yaml` (default: `dataset/raw/email_data.csv`) with two columns:

- Message: Raw email subject/body string
- Category: Ground-truth class (`ham` or `spam`)

---

## 3. Reproducing the ML Result

### Step 1: Run the Automated Test Suite

Verify that text normalization, data guardrails, pipeline serialization, and inference work before running training:

#### For Linux

```bash
PYTHONPATH=src pytest -v
```

#### For Windows

```bash
set PYTHONPATH=src && pytest -v
```

### Step 2: Run Training & Evaluation

Execute the end-to-end training pipeline using the deterministic seed (`random_state: 42`) and stratified 80/20 split defined in `config/config.yaml`.

#### For Linux

```bash
PYTHONPATH=src python3 -m spam_classifier.train --config config/config.yaml
```

#### For Windows

```bash
set PYTHONPATH=src && python -m spam_classifier.train --config config/config.yaml
```

### Output

- `dataset/processed/processed_emails.csv` - Cleaned, non-empty dataset with binary labels (`0 = ham`, `1 = spam`)
- `artifacts/spam_classifier.joblib` - Serialized `Pipeline(clean -> tfidf -> mnb)`
- `artifacts/metrics.json` - Full classification report, ROC-AUC, PR-AUC, and confusion matrix
- `artifacts/confusion_matrix.png` - Visual confusion matrix heatmap
- `artifacts/error_analysis.csv` - Every misclassified test email alongside its predicted `spam_probability`

### Step 3: Run Single-Email Inference

Test the saved `.joblib` artifact on new, unseen raw email text.

#### For Linux

```bash
PYTHONPATH=src python3 -m spam_classifier.predict "URGENT! Claim your FREE $1000 gift card now!"
```

#### For Windows

```bash
set PYTHONPATH=src && python -m spam_classifier.predict "URGENT! Claim your FREE $1000 gift card now!"
```

### Output

```text
The given email is "URGENT! Claim your FREE $1000 gift card now!" with a probability of 94.80 %
```

---

## 4. Why This Model and Metric Were Selected

### Model Selection: TF-IDF + Multinomial Naive Bayes (MultinomialNB)

**Strong inductive bias for sparse text:**  
Email bodies represented via TF-IDF produce high-dimensional, sparse term-frequency matrices. Multinomial Naive Bayes models conditional word-feature likelihoods directly, making it a fast, well-calibrated baseline on bag-of-words representations without overfitting small-to-medium datasets.

**Deterministic & low-latency serving:**  
Training completes in milliseconds and inference runs in sub-millisecond time with minimal memory overhead, ideal for high-throughput email filtering before escalating ambiguous messages to heavier models.

**Zero training-serving skew:**  
Wrapping the custom regex cleaner (`FunctionTransformer(clean_batch)`), `TfidfVectorizer`, and `MultinomialNB` inside a single scikit-learn `Pipeline` guarantees raw user text undergoes identical preprocessing at training and inference time.

### Metric Selection: Precision, Recall, F1 (Spam) & PR-AUC over Accuracy

#### Why Accuracy is Misleading

Real-world email datasets are heavily class-imbalanced (typically ~85–90% ham vs. ~10–15% spam). A trivial classifier that predicts ham for every email achieves ~87% accuracy while catching 0% of spam.

#### Precision vs. Recall Trade-Off

- **Precision (Spam)** measures what fraction of emails flagged as spam were actually spam-critical because False Positives (legitimate work/personal emails sent to the spam folder) carry high user-experience cost.

- **Recall (Spam)** measures what fraction of actual spam was caught-controlling False Negatives reaching the inbox.

- **F1-Score (Spam)** provides the primary single-number harmonic balance between Precision and Recall, while **PR-AUC (Average Precision)** evaluates ranking quality across all probability thresholds under class imbalance.

---

## 5. Assumptions & Known Limitations

### Assumptions

- Input files are UTF-8 encoded CSVs containing binary `"ham"` and `"spam"` labels.

- Stratified splitting (`stratify=y`) preserves the exact training/test class ratio across runs.

### Limitations

- **Punctuation & casing loss:** Per baseline specification, regex cleaning removes currency symbols (`$`), exclamation marks (`!`), and URLs, discarding non-alphanumeric spam cues.

- **Bag-of-words independence:** Multinomial Naive Bayes ignores word order and semantic context, making it vulnerable to subtle spear-phishing or synonym substitution.

- **Content-only signal:** Email header metadata is not modeled in this baseline.