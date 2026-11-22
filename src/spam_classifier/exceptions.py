class SpamClassifierError(Exception):
    """Base exception for the spam classifier application."""

class ConfigError(SpamClassifierError):
    """config.yaml is missing required keys"""

class DataValidationError(SpamClassifierError):
    """Input data is missing columns, has invalid labels, or is empty"""

class ModelArtifactError(SpamClassifierError):
    """Raised when serialized joblib pipeline cannot be found or loaded"""

class InferenceInputError(SpamClassifierError):
    """Raised when user text contains no alphanumeric characters"""