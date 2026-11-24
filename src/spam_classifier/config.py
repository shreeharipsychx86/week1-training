from dataclasses import dataclass
from pathlib import Path
import yaml
from spam_classifier.logger import get_logger
from spam_classifier.exceptions import ConfigError

@dataclass(frozen=True)
class PathConfig:
    raw_data: Path
    processed_data: Path
    model: Path
    metrics: Path = Path("artifacts/metrics.json")
    confusion_matrix: Path = Path("artifacts/confusion_matrix.png")
    error_analysis: Path = Path("artifacts/error_analysis.csv")

@dataclass(frozen=True)
class DataConfig:
    text_column: str
    label_column: str

@dataclass(frozen=True)
class TrainConfig:
    test_size: float
    random_state: int
    max_features: int
    alpha: list[float]


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig
    data: DataConfig
    train: TrainConfig

def load_config(config_path: Path = Path("config/config.yaml")) -> AppConfig:
    if not config_path.exists():
        raise ConfigError(f"No configuration file with .yaml was found: {config_path}")

    with config_path.open(mode="r") as f:
        config = yaml.safe_load(f) or {}

    try:
        p = config["paths"]
        d = config["data"]
        t = config["train"]
        logger = get_logger(__name__)
        return AppConfig(paths=PathConfig(raw_data=Path(p["raw_data"]),processed_data=Path(p["processed_data"]),model=Path(p["model"]),metrics=Path(p.get("metrics","artifacts/metrics.json")),confusion_matrix=Path(p.get("confusion_matrix","artifacts/metrics.json")), error_analysis=Path(p.get("error_analysis","artifacts/error_analysis.csv"))),
                        data = DataConfig(text_column=d["text_column"],label_column=d["label_column"]),
                        train=TrainConfig(test_size=t["test_size"], random_state=t["random_state"], max_features=t["max_features"],alpha=t["alpha"]))

    except KeyError as err:
        raise ConfigError(f"Missing configuration key in {config_path} file") from err
