"""Text comparison used both for verifying generated lines and for evaluation."""

import re


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    """Character error rate (the standard metric, used in evaluation)."""
    return levenshtein(pred.strip(), truth.strip()) / max(1, len(truth.strip()))


def letters(text: str) -> str:
    """Only letters and digits, lower-cased."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def loose_cer(pred: str, truth: str) -> float:
    """CER on letters and digits only, ignoring case, spaces and punctuation.

    Used to check that a generated piece says what it should. TrOCR's spacing and punctuation
    habits (e.g. 'chosen ,') shouldn't count as errors, but missing or wrong letters should.
    """
    t = letters(truth)
    return levenshtein(letters(pred), t) / max(1, len(t))
