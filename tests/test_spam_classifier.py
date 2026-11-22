import pytest
from spam_classifier.preprocessing import clean
from spam_classifier.config import AppConfig, PathConfig, DataConfig, TrainConfig
from spam_classifier.train import run_training
from spam_classifier.predict import EmailClassifier
from spam_classifier.exceptions import InferenceInputError, DataValidationError


from pathlib import Path
import pandas as pd

def test_clean() -> None:
    
    assert clean("WIN   $500\n\tNOW!!!") == "win 500 now"
    assert clean("!!!@@@###") == ""

def test_training() -> None:
    path = Path("sample")
    path.mkdir(parents=True,exist_ok=True)
    raw_csv = path / "raw.csv"
    print(raw_csv)
    df = pd.DataFrame({
            "text": [
                "Free cash prize claim now",
                "Urgent lottery winner click here",
                "Win free gift voucher today",
                "Cheap pills order online now",
                "Exclusive free deal limited offer",
                "Meeting moved to 3pm tomorrow",
                "Please review the quarterly report",
                "Are we still on for lunch",
                "Attached are the lecture notes",
                "Can you send the invoice details",
                "",
            ],
            "label": [
                "spam", "spam", "spam", "spam", "spam",
                "ham", "ham", "ham", "ham", "ham",
                "ham",
            ],
        }
    )
    df.to_csv(raw_csv, index=False, index_label=False)

    cfg = AppConfig(paths=PathConfig(raw_csv, path / "processed.csv", path / "model.joblib", path / "metrics.json", path / "cm.png", path / "errors.csv"),
                    data=DataConfig(text_column="text",label_column="label"),
                    train=TrainConfig(test_size=0.4, random_state=42, max_features=500,alpha=1.0))
    
    run_training(cfg)

    assert cfg.paths.model.exists()
    assert cfg.paths.metrics.exists()
    assert cfg.paths.confusion_matrix.exists()

    saved_processed = pd.read_csv(cfg.paths.processed_data)
    assert len(saved_processed) == 10

    classifier = EmailClassifier(cfg.paths.model)
    output = classifier.predict("WIN!!!   FREE   CASH   NOW!!!")
    assert output["class"] in {"ham", "spam"}
    assert "%" in output["prob"]

    with pytest.raises(InferenceInputError):
        classifier.predict("   !!!   ")

def test_label() -> None:
    path = Path("sample")
    test_csv = path / "test.csv"
    pd.DataFrame({"text": ["hello"], "label": ["unknown"]}).to_csv(test_csv, index=False,index_label=False)
    cfg = AppConfig(
        paths=PathConfig(
            raw_data=test_csv,
            processed_data=Path("sample") / "p.csv",
            model=Path("sample") / "m.joblib",
        ),
        data=DataConfig(text_column="text", label_column="label"),
        train=TrainConfig(test_size=0.2, random_state=42, max_features=100, alpha=1.0),
    )

    with pytest.raises(DataValidationError):
        run_training(cfg)