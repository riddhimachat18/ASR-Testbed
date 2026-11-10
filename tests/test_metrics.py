import pytest

from app.evaluation.metrics import (DEFAULT_NORMALIZATION, NORMALIZERS, UnknownNormalization, compute_metrics,
                                    normalize_text)


@pytest.mark.parametrize("raw,expected", [
    ("Hello, World!", "hello world"),
    ("  THE   cat\n sat ", "the cat sat"),
    ("It's  Don\u2019t stop", "it's don't stop"),     # apostrophes inside words are kept
    ("he said 'yes' loudly", "he said yes loudly"),    # quote marks around words are not
    ("well-known (really)", "well known really"),
    ("", ""),
])
def test_normalize_text_basic_v1(raw, expected):
    assert normalize_text(raw, "basic-v1") == expected


def test_formatting_differences_are_not_errors():
    m = compute_metrics("THE CAT SAT ON THE MAT", "The cat sat on the mat.")
    assert m["wer"] == 0.0 and m["cer"] == 0.0


def test_known_wer_counts():
    m = compute_metrics("the cat sat on the mat", "the cat sit on mat")   # 1 substitution + 1 deletion
    assert m["wer"] == pytest.approx(2 / 6)
    assert (m["word_substitutions"], m["word_deletions"], m["word_insertions"], m["ref_words"]) == (1, 1, 0, 6)
    assert m["normalization"] == DEFAULT_NORMALIZATION


def test_empty_hypothesis_is_all_deletions():
    m = compute_metrics("a b c", "")
    assert m["wer"] == 1.0 and m["word_deletions"] == 3


def test_insertions_can_push_wer_above_one():
    assert compute_metrics("a", "a b c d")["wer"] == pytest.approx(3.0)


def test_empty_reference_is_undefined_not_a_number():
    assert compute_metrics("", "anything") is None
    assert compute_metrics("...!", "anything") is None   # nothing left after normalisation


def test_cer_counts_characters():
    m = compute_metrics("kitten", "sitting")   # classic: 2 substitutions + 1 insertion
    assert m["char_substitutions"] == 2 and m["char_insertions"] == 1 and m["ref_chars"] == 6
    assert m["cer"] == pytest.approx(3 / 6)


# ── whisper-en-v1 (Part H) ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("TWENTY ONE colours", "21 colors"),                 # spelled numbers -> digits, British -> American
    ("He paid ten dollars in nineteen sixty", "he paid $10 in 1960"),
    ("It's  Don\u2019t stop", "it is do not stop"),      # curly apostrophe handled like a straight one
    ("well-known (laughs) really", "well known really"), # bracketed text removed, as in Whisper's evaluation
    ("...!", ""),                                         # punctuation-only tokens dropped
])
def test_normalize_text_whisper_en_v1(raw, expected):
    assert normalize_text(raw, "whisper-en-v1") == expected


def test_default_is_whisper_en_v1():
    assert DEFAULT_NORMALIZATION == "whisper-en-v1" and set(NORMALIZERS) == {"basic-v1", "whisper-en-v1"}


def test_number_formatting_is_an_error_only_under_basic_v1():
    ref, hyp = "THE TEN COMMANDMENTS", "The 10 commandments."
    assert compute_metrics(ref, hyp, "basic-v1")["word_substitutions"] == 1
    m = compute_metrics(ref, hyp, "whisper-en-v1")
    assert m["wer"] == 0.0 and m["normalization"] == "whisper-en-v1"


def test_british_spelling_is_not_an_error_under_whisper_en_v1():
    assert compute_metrics("the colour of honour", "the color of honor", "whisper-en-v1")["wer"] == 0.0
    assert compute_metrics("the colour of honour", "the color of honor", "basic-v1")["word_substitutions"] == 2


def test_unknown_normalization_is_rejected():
    with pytest.raises(UnknownNormalization):
        compute_metrics("a", "a", "basic-v9")
