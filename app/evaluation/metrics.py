"""WER / CER with an explicit, versioned text normalisation.

Why normalise: references are often upper-case with no punctuation (LibriSpeech), while models such as Whisper
return cased, punctuated text. Without normalisation that formatting difference is scored as word errors.
The version string is stored with every result, so scores from different normalisations are never mixed up.
Known limits of "basic-v1": numbers are not expanded ("10" vs "ten") and there is no spelling normalisation.
"""
import re
import unicodedata

import jiwer

NORMALIZATION_VERSION = "basic-v1"

_NOT_WORD = re.compile(r"[^\w\s']", re.UNICODE)
_EDGE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
    text = _NOT_WORD.sub(" ", text)        # drop punctuation, keep apostrophes inside words (don't, it's)
    text = _EDGE_APOSTROPHE.sub(" ", text)  # but not quote marks around words
    return _SPACES.sub(" ", text).strip()


def compute_metrics(reference: str, hypothesis: str) -> dict | None:
    """Returns None when the reference is empty after normalisation (WER is undefined then).

    The error counts are kept so results can be aggregated correctly across files:
    dataset WER = total errors / total reference words, NOT the mean of per-file WERs.
    """
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
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
        "normalization": NORMALIZATION_VERSION,
    }
