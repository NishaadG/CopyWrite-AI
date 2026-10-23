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


def missing_letters(pred: str, truth: str) -> int:
    """How many letters/digits of `truth` have no counterpart in `pred` (deletions in the
    best alignment, case/spaces/punctuation ignored).

    A missing word or a dropped letter shows up here even when the overall error rate looks
    small (a 4-word piece missing one word is only ~0.25 CER, but 3-6 letters missing).
    """
    p, t = letters(pred), letters(truth)
    # dp[i][j] = (edit distance, deletions) aligning t[:i] with p[:j]; ties prefer fewer deletions
    dp = [[(0, 0)] * (len(p) + 1) for _ in range(len(t) + 1)]
    for j in range(1, len(p) + 1):
        dp[0][j] = (j, 0)
    for i in range(1, len(t) + 1):
        dp[i][0] = (i, i)
        for j in range(1, len(p) + 1):
            sub = dp[i - 1][j - 1]
            dele = dp[i - 1][j]
            ins = dp[i][j - 1]
            dp[i][j] = min((sub[0] + (t[i - 1] != p[j - 1]), sub[1]),
                           (dele[0] + 1, dele[1] + 1),
                           (ins[0] + 1, ins[1]))
    return dp[-1][-1][1]


def run_on_words(pred: str, truth: str) -> int:
    """Words read beyond the text's word count, when there are also extra letters.

    Catches the model writing on after the text ('while the the', 'and a a'). Requiring extra
    letters too means OCR splitting one word in two ('b alances') doesn't count."""
    words = lambda s: [w for w in s.split() if letters(w)]
    if len(letters(pred)) <= len(letters(truth)):
        return 0
    return max(0, len(words(pred)) - len(words(truth)))


def extra_marks(pred: str, truth: str) -> int:
    """Punctuation-like marks read beyond those in the text (stray dots, ticks, colons)."""
    marks = lambda s: len(re.sub(r"[A-Za-z0-9\s]", "", s))
    return max(0, marks(pred) - marks(truth))


def loose_cer(pred: str, truth: str) -> float:
    """CER on letters and digits only, ignoring case, spaces and punctuation.

    Used to check that a generated piece says what it should. TrOCR's spacing and punctuation
    habits (e.g. 'chosen ,') shouldn't count as errors, but missing or wrong letters should.
    """
    t = letters(truth)
    return levenshtein(letters(pred), t) / max(1, len(t))
