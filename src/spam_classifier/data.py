from pathlib import Path
import pandas as pd

from spam_classifier.config import DataConfig
from spam_classifier.exceptions import DataValidationError
from spam_classifier.logger import get_logger
from spam_classifier.preprocessing import clean

logger = get_logger(__name__)

LABEL_MAPPING = {"ham": 0, "spam": 1}

def load_raw_data(raw_path: Path, data_cfg: DataConfig) -> pd.DataFrame:
    if not raw_path.exists():
        raise DataValidationError(f"Raw dataset not found at {raw_path}")

    df = pd.read_csv(raw_path)
    required_col = {data_cfg.text_column, data_cfg.label_column}
    missing_col = required_col - set(df.columns)
    if missing_col:
        raise DataValidationError(f"Dataset missing required columns: {missing_col}")

    df = df[[data_cfg.text_column, data_cfg.label_column]].copy()

    logger.info("Loaded dataset from %s with rows=%d and cols=%d",raw_path,df.shape[0],df.shape[1])
    return df

def log_dataset_statistics(df: pd.DataFrame, data_cfg: DataConfig):
    logger.info("Dataset summary statistics:")
    logger.info("No of rows: %d", df.shape[0])
    logger.info("No of rows: %d", df.shape[1])
    logger.info("Null rows: %s", df.isna().sum())


def preprocess_and_save(df: pd.DataFrame, processed_path: Path, data_cfg: DataConfig) -> pd.DataFrame:
    text_col = data_cfg.text_column
    label_col = data_cfg.label_column

    cleaned = df.dropna().copy()
    cleaned[text_col] = cleaned[text_col].str.lower()
    cleaned[label_col] = cleaned[label_col].str.lower()

    cleaned = cleaned[(cleaned[text_col] != "") | (cleaned[label_col] != "")]

    invalid_labels = set(cleaned[label_col].unique()) - set(LABEL_MAPPING.keys())
   
    if invalid_labels:
        raise DataValidationError(f"Unexpected label values found: {invalid_labels}")
    
    numSpam = cleaned.loc[cleaned[label_col] == "spam", label_col].count()
    numHam = cleaned.loc[cleaned[label_col] == "ham", label_col].count()

    
    logger.info("No of spam messages: %d", numSpam)
    logger.info("No of ham messages: %d", numHam)

    cleaned[text_col] = cleaned[text_col].apply(clean)
    cleaned = cleaned[cleaned[text_col] != ""].reset_index(drop=True)
    cleaned[label_col] = cleaned[label_col].map(LABEL_MAPPING).astype(int)

    if cleaned.empty:
        raise DataValidationError("Dataset is empty after cleaning and filtering.")

    processed_path.parent.mkdir(parents=True,exist_ok=True)
    cleaned.to_csv(processed_path,index=False,index_label=False)
    logger.info("Saved %d processed rows to %s", len(cleaned), processed_path)
    return cleaned