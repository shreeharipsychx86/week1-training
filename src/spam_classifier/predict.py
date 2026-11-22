from pathlib import Path
from spam_classifier.exceptions import ModelArtifactError, InferenceInputError
from spam_classifier.logger import get_logger
from spam_classifier.preprocessing import clean
from spam_classifier.config import load_config
import joblib
import argparse

logger = get_logger(__name__)

class EmailClassifier:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise ModelArtifactError(f"The model joblib file does not exists {model_path}")

        self.pipeline = joblib.load(model_path)

    def predict(self, text: str) -> dict[str,str]:
        if not isinstance(text,str) or not clean(text):
            raise InferenceInputError("Invalid Input: The text should contain some alphabets or digits")

        pred_class = int(self.pipeline.predict([text])[0])
        pred_prob = self.pipeline.predict_proba([text])[0]
        email_class = "ham" if pred_class == 0 else "spam" 
        output_prob = pred_prob[pred_class]
        logger.info("The given email is %s with a probability of %d %%",email_class, round(int(output_prob * 100), 2))

        return {"class":email_class, "prob":f"{round(int(output_prob * 100), 2)} %"}

def main() -> None:
    parser = argparse.ArgumentParser(description="Classify whether an email is spam or ham")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Path to YAML configuration file")
    parser.add_argument("--text", type=str, help="Enter the email content")

    args = parser.parse_args()
    cfg = load_config(args.config)
    classifier = EmailClassifier(cfg.paths.model)
    classifier.predict(args.text)


if __name__ == "__main__":
    main()