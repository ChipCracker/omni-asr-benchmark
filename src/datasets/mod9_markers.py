"""mod9 reference clean-up: drop the annotator marker words (2026-10-09).

The mod9 release stores its non-speech marks as plain lower-case words inside the
transcript -- ``musik``, ``geräusch``, ``fremdsprache``, ``unverständlich``, ``leeresaudio``
(also ``leeres audio``) and ``keinaudio`` -- with no bracket or casing left to tell them from
the spoken words ``Musik``, ``Geräusch`` ... No model can transcribe a marker, so a scorer that
keeps them charges every model a deletion per mark (0.57 WER points for SelOSS v4; 2 882
``musik`` marks alone). :func:`clean_reference` removes the marks with a context rule:

* ``leeresaudio`` / ``keinaudio`` / ``leeres audio`` / ``kein audio`` are always marks;
* ``musik``, ``geräusch``, ``fremdsprache``, ``unverständlich`` are marks UNLESS the
  neighbouring word makes them a spoken word: a determiner/adjective/preposition before
  (``die musik``, ``elektronische musik``, ``eine fremdsprache``, ``völlig unverständlich``)
  or a verb/preposition after (``musik ist``, ``fremdsprache lernen``). Repeated marks
  (``musik musik musik``) never satisfy the rule and go.

Clips whose reference is empty after the clean-up (578 marker-only clips) and the 17
``dies ist blindtext`` placeholders are excluded (``manifests/mod9_exclude_markers.txt``,
built by ``scripts/mod9_marker_exclude.py``) -- there is no reference to score against.
Applied by ``src/datasets/mod9.py`` for new runs and by ``scripts/refresh_references.py`` to
stored results.
"""
from __future__ import annotations

import re

ALWAYS = {"leeresaudio", "keinaudio"}
TWO_WORD = {("leeres", "audio"), ("kein", "audio")}

_DET = {"die", "der", "den", "dem", "des", "das", "dieser", "diese", "dieses", "diesem", "diesen",
        "jene", "jener", "welche", "welcher", "welches", "solche", "solcher", "seine", "seiner",
        "seinem", "seinen", "ihre", "ihrer", "ihrem", "ihren", "meine", "meiner", "meinem", "meinen",
        "unsere", "unserer", "unserem", "unseren", "eure", "eurer", "deine", "deiner", "keine",
        "keiner", "keinem", "keinen", "eine", "einer", "einem", "einen", "ein", "kein", "viel",
        "wenig", "mehr", "weniger", "jede", "jeder", "jedes", "jedem", "andere", "anderen", "anderer"}
_PREP = {"zur", "zum", "mit", "von", "an", "über", "ohne", "aus", "in", "für", "durch", "bei",
         "beim", "als", "wie", "nur", "auch", "vor", "nach", "gegen", "unter", "zwischen", "um"}
_ADJ_MUSIK = {"deutsche", "deutschen", "deutscher", "elektronische", "elektronischen", "elektronischer",
              "klassische", "klassischen", "klassischer", "neue", "neuen", "neuer", "moderne", "modernen",
              "alte", "alten", "alter", "gute", "guten", "guter", "schlechte", "schlechten", "laute",
              "lauten", "leise", "leisen", "schöne", "schönen", "eigene", "eigenen", "ganze", "ganzen",
              "englische", "englischen", "amerikanische", "amerikanischen", "türkische", "türkischen",
              "traditionelle", "traditionellen", "populäre", "populären", "viel", "wenig", "live"}
_VERB_MUSIK = {"mag", "mögen", "mochte", "mochten", "macht", "machen", "mache", "machte", "machten",
               "gemacht", "höre", "hören", "hört", "hörte", "hörten", "gehört", "spiele", "spielen",
               "spielt", "spielte", "gespielt", "liebe", "lieben", "liebt", "produziere", "produzieren",
               "produziert", "schreibe", "schreiben", "schreibt", "komponiert", "komponieren"}
MUSIK_PREV = _DET | _PREP | _ADJ_MUSIK | _VERB_MUSIK
MUSIK_NEXT = {"ist", "war", "wird", "wurde", "hat", "hatte", "kommt", "kam", "macht", "machen", "gemacht",
              "hören", "hört", "gehört", "spielt", "spielen", "gespielt", "läuft", "lief", "zu", "von",
              "aus", "für", "als", "gibt", "gab", "bedeutet", "heißt", "sein", "bleibt", "klingt"}
GER_PREV = _DET | {"lautes", "leises", "komisches", "seltsames", "dumpfes", "starkes", "leichtes",
                   "lauten", "leisen", "so", "als", "welches"}
GER_NEXT = {"gehört", "ist", "war", "hören", "machen", "macht", "gemacht", "kommt", "kam", "zu", "von"}
UNV_PREV = {"ist", "war", "wäre", "wird", "sind", "bleibt", "völlig", "ganz", "total", "sehr", "so",
            "mir", "uns", "ihm", "ihr", "ihnen", "einfach", "ziemlich", "oft", "schon", "auch", "nicht",
            "fast", "immer", "teilweise", "eigentlich", "irgendwie", "leider", "manchmal", "recht",
            "etwas", "völlig", "komplett", "relativ", "absolut", "meist", "meistens", "für"}
UNV_NEXT = {"ist", "war", "wäre", "für", "gesprochen", "bleibt", "sind", "gemacht", "wird", "geworden",
            "finde", "fand"}
FS_PREV = _DET | {"zur", "als", "in", "zweite", "dritte", "erste", "andere", "anderen", "fremde", "neue",
                  "neuen", "weitere", "weiteren"}
FS_NEXT = {"lernen", "lernt", "lerne", "gelernt", "zu", "ist", "war", "sprechen", "spricht", "gesprochen",
           "beherrschen", "beherrscht", "unterrichten", "unterrichtet", "unterricht", "können", "kann"}

_BLINDTEXT_RE = re.compile(r"\bblindtext\b")


def is_blindtext(text: str) -> bool:
    return bool(_BLINDTEXT_RE.search(text.lower()))


def clean_reference(text: str) -> str:
    """Return ``text`` without the mod9 marker words (see module docstring)."""
    toks = text.split()
    out = []
    n = len(toks)
    i = 0
    while i < n:
        w = toks[i]
        wl = w.lower()
        prev = toks[i - 1].lower() if i else ""
        nxt = toks[i + 1].lower() if i + 1 < n else ""
        if wl in ALWAYS:
            i += 1
            continue
        if (wl, nxt) in TWO_WORD:
            i += 2
            continue
        keep = True
        if wl == "musik":
            keep = prev in MUSIK_PREV or nxt in MUSIK_NEXT
        elif wl == "geräusch":
            keep = prev in GER_PREV or nxt in GER_NEXT
        elif wl == "unverständlich":
            keep = prev in UNV_PREV or nxt in UNV_NEXT
        elif wl == "fremdsprache":
            keep = prev in FS_PREV or nxt in FS_NEXT
        if keep:
            out.append(w)
        i += 1
    return " ".join(out)
