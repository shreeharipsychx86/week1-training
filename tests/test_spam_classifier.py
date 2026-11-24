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
        "Please review the quarterly report before Friday",
        "Congratulations, you have been selected for a special reward",
        "Could you review my code changes when you have time",
        "Win a free gift voucher today",
        "Thank you for your feedback on the proposal",
        "Exclusive deal available for a limited time",
        "Let's schedule a call for next week",
        "Claim your reward before the offer expires",
        "The client approved the final document",
        "Free membership available if you sign up today",
        "Please find the meeting agenda attached",
        "Urgent lottery winner notification, respond now",
        "Meeting moved to 3 PM tomorrow",
        "Earn money fast from the comfort of your home",
        "Attached are the lecture notes from today's session",
        "You have been pre-approved for a loan",
        "Reminder about the team stand-up tomorrow morning",
        "Click here to receive your free smartphone",
        "Can you send the invoice details by end of day",
        "Limited-time discount available on all products",
        "Happy birthday and best wishes for the year ahead",
        "Double your income with this simple method",
        "Are we still on for lunch tomorrow",
        "Congratulations, you've won a luxury vacation package",
        "Here are the minutes from today's meeting",
        "Cheap medicines available online with quick delivery",
        "Please confirm your availability for the workshop",
        "Free cash prize waiting for you",
        "The project deadline has been extended by two days",
        ""
        ],
        "label": [
            "ham","spam","ham","spam","ham","spam","ham","spam","ham","spam","ham","spam","ham",
            "spam","ham","spam","ham","spam","ham","spam","ham","spam","ham","spam","ham","spam"
            ,"ham","spam","ham","ham"
        ],
        }
    )
    df.to_csv(raw_csv, index=False, index_label=False)

    cfg = AppConfig(paths=PathConfig(raw_csv, path / "processed.csv", path / "model.joblib", path / "metrics.json", path / "cm.png", path / "errors.csv"),
                    data=DataConfig(text_column="text",label_column="label"),
                    train=TrainConfig(test_size=0.4, random_state=42, max_features=500,alpha=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]))
    
    run_training(cfg)

    assert cfg.paths.model.exists()
    assert cfg.paths.metrics.exists()
    assert cfg.paths.confusion_matrix.exists()

    saved_processed = pd.read_csv(cfg.paths.processed_data)
    assert len(saved_processed) == 29

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
        train=TrainConfig(test_size=0.2, random_state=42, max_features=100, alpha=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
    )

    with pytest.raises(DataValidationError):
        run_training(cfg)