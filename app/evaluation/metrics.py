"""WER / CER with an explicit, versioned text normalization.

Why normalize: references are often upper-case with no punctuation (LibriSpeech), while models such as Whisper
return cased, punctuated text with digits. Without normalization those formatting differences are scored as word
errors. The normalization name is stored with every result, so scores from different normalizations are never
mixed (the leaderboard and the reports group on it).

Normalizations
  basic-v1        NFKC, lowercase, drop punctuation (keeping in-word apostrophes), collapse whitespace.
                  Known limits: "10" vs "ten" and "colour" vs "color" count as errors. Used for every score
                  before Part H.
  whisper-en-v1   Whisper's EnglishTextNormalizer (vendored, app/evaluation/whisper_norm): spelled-out numbers
                  become digits on both sides, British spellings become American, fillers (uh, um) are dropped.
                  Bracketed text such as "(laughs)" is removed, as in Whisper's own evaluation.
                  Plus two guards: curly apostrophes are made straight first, and punctuation-only tokens are
                  dropped. The usual normalizer for comparing with published Whisper results. Default since Part H.
"""
import re
import unicodedata
from functools import lru_cache
from typing import Callable

import jiwer

DEFAULT_NORMALIZATION = "whisper-en-v1"
NORMALIZATION_VERSION = DEFAULT_NORMALIZATION  # kept for older imports

_NOT_WORD = re.compile(r"[^\w\s']", re.UNICODE)
_EDGE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")
_SPACES = re.compile(r"\s+")


def normalize_basic_v1(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = _NOT_WORD.sub(" ", text)        # drop punctuation, keep apostrophes inside words (don't, it's)
    text = _EDGE_APOSTROPHE.sub(" ", text)  # but not quote marks around words
    return _SPACES.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _whisper_en() -> Callable[[str], str]:
    from app.evaluation.whisper_norm import EnglishTextNormalizer
    return EnglishTextNormalizer()


_HAS_WORD_CHAR = re.compile(r"\w")


def normalize_whisper_en_v1(text: str) -> str:
    """Whisper's EnglishTextNormalizer plus two guards, both part of this version's definition:
    curly apostrophes become straight first (the vendored code turns U+2019 into a space, so "don\u2019t" would
    become "don t"), and tokens with no letter or digit left (a lone "." from "...") are dropped."""
    out = _whisper_en()(text.replace("\u2019", "'").replace("\u2018", "'"))
    return " ".join(tok for tok in out.split() if _HAS_WORD_CHAR.search(tok))


NORMALIZERS: dict[str, Callable[[str], str]] = {
    "basic-v1": normalize_basic_v1,
    "whisper-en-v1": normalize_whisper_en_v1,
}


class UnknownNormalization(ValueError):
    pass


def get_normalizer(name: str) -> Callable[[str], str]:
    try:
        return NORMALIZERS[name]
    except KeyError:
        raise UnknownNormalization(f"unknown normalization '{name}'; known: {sorted(NORMALIZERS)}") from None


def normalize_text(text: str, normalization: str = DEFAULT_NORMALIZATION) -> str:
    return get_normalizer(normalization)(text)


def compute_metrics(reference: str, hypothesis: str, normalization: str = DEFAULT_NORMALIZATION) -> dict | None:
    """Returns None when the reference is empty after normalization (WER is undefined then).

    The error counts are kept so results can be aggregated correctly across files:
    dataset WER = total errors / total reference words, NOT the mean of per-file WERs.
    """
    norm = get_normalizer(normalization)
    ref, hyp = norm(reference), norm(hypothesis)
    if not ref:
        return None
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    return {
        "wer": w.wer,
        "cer": c.cer,
        "ref_words": w.hits + w.substitutions + w.deletions,
        "word_substitutions": w.substitutions,
        "word_deletions": w.deletions,
        "word_insertions": w.insertions,
        "ref_chars": c.hits + c.substitutions + c.deletions,
        "char_substitutions": c.substitutions,
        "char_deletions": c.deletions,
        "char_insertions": c.insertions,
        "normalization": normalization,
    }
