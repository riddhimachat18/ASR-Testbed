"""Whisper's English text normalizer, vendored unchanged from openai-whisper 20250625 (MIT, see LICENSE).

Vendored rather than installed so the scoring behaviour is pinned in this repository: changing it means a new
normalization version string in app/evaluation/metrics.py, never a silent library upgrade.
Needs the `regex` and `more-itertools` packages.
"""
from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizer
