import re
from typing import Iterable

_PUNCTUATION_RE = re.compile(r"[^A-Za-z\s\d]")
_WHITESPACE_RE = re.compile(r"\s+")

def clean(text: str) -> str:

    if not isinstance(text,str):
        return ""
    
    text = text.lower()
    
    # Remove punctuations
    text = _PUNCTUATION_RE.sub(r"[^A-Za-z\s\d]","",text)

    # Remove unnecessary whitespaces, tabs, newlines
    text = _WHITESPACE_RE.sub(r"\s+"," ",text).strip()

    return text

def clean_batch(texts: Iterable[str]) -> list[str]:
    """Wrapper used inside sklearn FunctionTransformer"""
    return [clean(t) for t in texts]