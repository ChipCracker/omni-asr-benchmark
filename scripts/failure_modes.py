#!/usr/bin/env python3
"""Fehlerkategorien eines Benchmark-Laufs: Wortausrichtung Hyp/Ref (Scorer-Normalisierung),
jede Edit-Operation bekommt eine Kategorie. Ausgabe: Tabelle je Kategorie (gesamt + je Datensatz),
Top-Verwechslungen, Beispiele.  python3 failure_modes.py "SelOSS-RNNT_105M_v4_Beam_5,_Blank-Strafe_0.5"
"""
import collections
import glob
import json
import os
import re
import sys

sys.path[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
from benchmark.metrics import normalize_text  # noqa: E402

MODEL = sys.argv[1]
RESDIR = sys.argv[2] if len(sys.argv) > 2 else "results_tuda_cv"

FILLERS = {"äh", "ähm", "hm", "hmm", "mh", "mhm", "mmh", "ähh", "öh", "öhm", "eh", "ehm", "em", "ah", "oh", "uh", "uhm", "ehh", "ne", "gell", "halt", "also", "ja", "sozusagen", "naja", "genau", "quasi", "irgendwie"}
HARD_FILLERS = {"äh", "ähm", "hm", "hmm", "mh", "mhm", "mmh", "ähh", "öh", "öhm", "eh", "ehm", "em", "ah", "uh", "uhm", "ehh"}
ARTICLES = {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines", "kein", "keine", "keinen", "keinem", "keiner"}
PRONOUNS = {"ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "dich", "ihn", "uns", "euch", "mir", "dir", "ihm", "ihnen", "mein", "meine", "meinen", "meinem", "meiner", "dein", "deine", "sein", "seine", "seinen", "seinem", "seiner", "unser", "unsere", "unseren", "unserem", "unserer", "euer", "eure", "ihrer", "ihren", "ihrem", "ihre", "man", "sich", "wer", "was", "wen", "wem", "wessen", "dies", "diese", "diesen", "diesem", "dieser", "dieses", "jene", "jener", "welche", "welcher", "welches", "welchen", "welchem"}
PREP_CONJ = {"in", "im", "an", "am", "auf", "aus", "bei", "beim", "mit", "nach", "von", "vom", "zu", "zum", "zur", "über", "unter", "vor", "hinter", "neben", "zwischen", "durch", "für", "gegen", "ohne", "um", "bis", "seit", "ab", "und", "oder", "aber", "denn", "sondern", "dass", "ob", "weil", "wenn", "als", "wie", "da", "so", "doch", "noch", "nur", "auch", "schon", "dann", "jetzt", "hier", "dort", "nicht", "mal", "sehr", "ganz", "zwar", "etwa", "eben"}
AUX = {"ist", "sind", "war", "waren", "bin", "bist", "seid", "hat", "haben", "habe", "hast", "hatte", "hatten", "wird", "werden", "werde", "wirst", "wurde", "wurden", "kann", "können", "könnte", "könnten", "muss", "müssen", "müsste", "soll", "sollen", "sollte", "will", "wollen", "wollte", "darf", "dürfen", "mag", "würde", "würden", "gibt", "gab"}
FUNCTION = ARTICLES | PRONOUNS | PREP_CONJ | AUX
MARKERS = {"musik", "geräusch", "fremdsprache", "leeresaudio", "unverständlich", "häs", "lachen", "husten", "räuspern", "rauschen"}
# Umgangssprachliche Kurzform (Referenz woertlich) -> Standardform (Hypothese bereinigt)
COLLOQ = {"ne": {"eine", "nein", "nicht"}, "n": {"ein", "einen", "den"}, "nen": {"einen"}, "nem": {"einem"}, "ner": {"einer"},
          "net": {"nicht"}, "nit": {"nicht"}, "ned": {"nicht"}, "nix": {"nichts"}, "is": {"ist"}, "isch": {"ist", "ich"},
          "gibts": {"gibt"}, "ok": {"okay"}, "grade": {"gerade"}, "ma": {"mal", "wir"}, "i": {"ich"}, "d": {"das", "die", "der"},
          "s": {"es", "das"}, "mei": {"mein"}, "des": {"das"}, "dat": {"das"}, "wat": {"was"}, "mer": {"wir", "mir"}, "mir": {"wir"},
          "nee": {"nein"}, "nö": {"nein"}, "jo": {"ja"}, "ham": {"haben"}, "hamma": {"haben"}, "kannste": {"kannst"}, "haste": {"hast"},
          "biste": {"bist"}, "willste": {"willst"}, "sowas": {"so"}, "halt": set(), "gell": set(), "ah": set(), "oh": set(),
          "dran": {"daran"}, "drin": {"darin"}, "drauf": {"darauf"}, "drum": {"darum"}, "rum": {"herum"}, "raus": {"heraus", "raus"},
          "rein": {"herein", "hinein"}, "runter": {"herunter", "hinunter"}, "rauf": {"herauf", "hinauf"}, "rüber": {"herüber", "hinüber"},
          "bissl": {"bisschen"}, "bisserl": {"bisschen"}, "bissel": {"bisschen"}, "mal": {"einmal"}, "einmal": {"mal"}, "nja": {"ja"},
          "hab": {"habe"}, "glaub": {"glaube"}, "würd": {"würde"}, "wär": {"wäre"}, "hätt": {"hätte"}, "könnt": {"könnte"},
          "müsst": {"müsste"}, "sag": {"sage"}, "mach": {"mache"}, "komm": {"komme"}, "geh": {"gehe"}, "seh": {"sehe"}, "find": {"finde"}}


def colloquial(r, h):
    """Referenz woertlich/umgangssprachlich, Hypothese Standardform (oder umgekehrt)."""
    if h in COLLOQ.get(r, ()) or r in COLLOQ.get(h, ()):
        return True
    # apokopiertes Verb-e: hab/habe, glaub/glaube, wuerd/wuerde (Stamm >= 3)
    if (h == r + "e" or r == h + "e") and len(min(r, h, key=len)) >= 3:
        return True
    # vorangestelltes Kurzwort "ne/n" etc. ist oben; Dialekt-Endungen -a/-e: "heut/heute"
    return False
NUMBER_WORDS = re.compile(r"^(null|eins?|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf|hundert|tausend|million|milliarde|komma|prozent|uhr|erste|zweite|dritte|vierte|fünfte|sechste|siebte|achte|neunte|zehnte|zwanzig|dreissig|vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig)")


def lev(a, b):
    """Zeichen-Levenshtein."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align(ref, hyp):
    """Wort-Levenshtein mit Rueckverfolgung -> Liste (op, ref_wort|None, hyp_wort|None)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None)); i -= 1
        else:
            ops.append(("I", None, hyp[j - 1])); j -= 1
    ops.reverse()
    return ops


def is_number(w):
    return bool(NUMBER_WORDS.match(w)) or any(ch.isdigit() for ch in w)


def same_stem(a, b):
    """Gleicher Stamm, andere Endung: gemeinsamer Praefix >= 4 und >= 60 % des kuerzeren Worts, Rest <= 3 Zeichen."""
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    short = min(len(a), len(b))
    return k >= 4 and k >= 0.6 * short and max(len(a), len(b)) - k <= 3


def categorize(ops, ref_freq):
    """Jede Nicht-C-Operation -> Kategorie. Kontextregeln: Zusammen-/Getrenntschreibung,
    Wiederholungen, lange Auslassungen."""
    cats = []
    n = len(ops)
    used = [False] * n
    # 1) Getrennt-/Zusammenschreibung: hyp-Wort == ref_i + ref_{i+1} (oder umgekehrt), in Fenster von 3 Ops
    for i in range(n):
        if used[i] or ops[i][0] == "C":
            continue
        window = ops[i:i + 3]
        rw = [o[1] for o in window if o[1]]
        hw = [o[2] for o in window if o[2]]
        if len(rw) >= 2 and len(hw) >= 1 and "".join(rw[:2]) == hw[0]:
            for k in range(i, min(n, i + 3)):
                if ops[k][0] != "C":
                    used[k] = True
            cats.append(("Zusammenschreibung (Ref 2 Wörter, Hyp 1)", " ".join(rw[:2]), hw[0]))
            continue
        if len(hw) >= 2 and len(rw) >= 1 and "".join(hw[:2]) == rw[0]:
            for k in range(i, min(n, i + 3)):
                if ops[k][0] != "C":
                    used[k] = True
            cats.append(("Getrenntschreibung (Ref 1 Wort, Hyp 2)", rw[0], " ".join(hw[:2])))
            continue
    # 2) Lange Auslassungen (>= 4 D in Folge)
    i = 0
    while i < n:
        if ops[i][0] == "D" and not used[i]:
            j = i
            while j < n and ops[j][0] == "D":
                j += 1
            if j - i >= 4:
                words = [ops[k][1] for k in range(i, j)]
                for k in range(i, j):
                    used[k] = True
                nm = sum(w in MARKERS for w in words)
                if nm:
                    cats.append(("Referenz-Marker (musik, geräusch, fremdsprache, häs …)", " ".join(words), "", nm))
                if j - i - nm >= 4:
                    cats.append(("Ausgelassene Passage (≥ 4 Wörter am Stück)", " ".join(w for w in words if w not in MARKERS), "", j - i - nm))
                elif j - i - nm:
                    for w in words:
                        if w not in MARKERS:
                            cats.append(("Inhaltswort weggelassen" if w not in FUNCTION else "Funktionswort weggelassen", w, ""))
            i = j
        else:
            i += 1
    # 3) Einzeloperationen
    prev_ref = None
    hyp_words = [o[2] for o in ops if o[2]]
    halluz = "präsident" in hyp_words and not any(o[1] == "präsident" for o in ops)
    for k, (op, r, h) in enumerate(ops):
        if op == "C":
            prev_ref = r
            continue
        if used[k]:
            prev_ref = r or prev_ref
            continue
        if r in MARKERS or h in MARKERS:
            cats.append(("Referenz-Marker (musik, geräusch, fremdsprache, häs …)", r or "", h or ""))
            prev_ref = r or prev_ref
            continue
        if halluz and op in ("I", "S") and h in ("herr", "präsident"):
            cats.append(("Halluzination »Herr Präsident« (kurze/leise Clips)", r or "", h))
            prev_ref = r or prev_ref
            continue
        if op == "S" and colloquial(r, h):
            cats.append(("Umgangssprache → Standardform (hab/habe, ne/eine, net/nicht)", r, h))
            prev_ref = r
            continue
        if op == "D":
            if r in HARD_FILLERS:
                cats.append(("Füllwort weggelassen (äh, ähm, hm)", r, ""))
            elif r == prev_ref or (k + 1 < n and ops[k + 1][1] == r):
                cats.append(("Wiederholung/Stottern bereinigt", r, ""))
            elif r in FILLERS:
                cats.append(("Diskurspartikel weggelassen (ja, also, halt, ne)", r, ""))
            elif r in FUNCTION:
                cats.append(("Funktionswort weggelassen", r, ""))
            elif len(r) <= 2:
                cats.append(("Kurzes Wort weggelassen", r, ""))
            else:
                cats.append(("Inhaltswort weggelassen", r, ""))
        elif op == "I":
            if h in HARD_FILLERS:
                cats.append(("Füllwort eingefügt", "", h))
            elif h in FUNCTION or h in FILLERS:
                cats.append(("Funktionswort eingefügt", "", h))
            else:
                cats.append(("Inhaltswort eingefügt", "", h))
        else:  # S
            if r in FUNCTION and h in FUNCTION:
                cats.append(("Funktionswort verwechselt (der/den/dem, ein/einen …)", r, h))
            elif is_number(r) or is_number(h):
                cats.append(("Zahl/Zahlwort", r, h))
            elif "-" in r or "-" in h:
                cats.append(("Bindestrich-Schreibung", r, h))
            elif same_stem(r, h):
                cats.append(("Flexion/Endung (gleicher Stamm)", r, h))
            elif r in HARD_FILLERS or h in HARD_FILLERS:
                cats.append(("Füllwort verwechselt", r, h))
            elif lev(r, h) <= max(1, round(0.34 * max(len(r), len(h)))):
                if ref_freq.get(r, 0) <= 2:
                    cats.append(("Seltenes Wort/Name lautähnlich ersetzt", r, h))
                else:
                    cats.append(("Lautähnliche Verwechslung (häufiges Wort)", r, h))
            elif ref_freq.get(r, 0) <= 2:
                cats.append(("Seltenes Wort/Name anders ersetzt", r, h))
            elif r in FUNCTION or h in FUNCTION:
                cats.append(("Funktionswort gegen Inhaltswort", r, h))
            else:
                cats.append(("Sonstige Substitution", r, h))
        prev_ref = r or prev_ref
    return cats


def main():
    files = sorted(glob.glob(os.path.join(RESDIR, MODEL + "__*.json")))
    if not files:
        sys.exit("keine Dateien für " + MODEL)
    per_ds = {}
    all_refs = collections.Counter()
    for f in files:
        d = json.load(open(f))
        rows = []
        for s in d["per_sample"]:
            ref = normalize_text(s["references"][d.get("primary_reference", "ref")]).split()
            hyp = normalize_text(s["hypothesis"] or "").split()
            rows.append((ref, hyp, (s.get("extra") or {}).get("utt_id") or s.get("index")))
            all_refs.update(ref)
        per_ds[d["dataset"]] = rows
    total = collections.Counter(); by_ds = collections.defaultdict(collections.Counter)
    pairs = collections.defaultdict(collections.Counter); examples = collections.defaultdict(list)
    ref_words = collections.Counter(); errors_total = 0
    for ds, rows in per_ds.items():
        for ref, hyp, uid in rows:
            ref_words[ds] += len(ref)
            ops = align(ref, hyp)
            cats = categorize(ops, all_refs)
            for c in cats:
                w = c[3] if len(c) > 3 else 1
                total[c[0]] += w; by_ds[ds][c[0]] += w; errors_total += w
                pairs[c[0]][(c[1], c[2])] += 1
                if len(examples[c[0]]) < 6 and c[1] and c[2]:
                    examples[c[0]].append((ds, c[1], c[2]))
    print(f"Modell: {MODEL}   Datensätze: {len(per_ds)}   Ref-Wörter: {sum(ref_words.values())}   Fehler (S+D+I): {errors_total}")
    print("\n== Kategorien gesamt (Anteil an allen Fehlern, Anteil an Ref-Wörtern = Beitrag zur WER in Punkten)")
    nref = sum(ref_words.values())
    for cat, n in total.most_common():
        print(f"{n:7d}  {100*n/errors_total:5.1f} %  {100*n/nref:5.2f} WER-Pkt  {cat}")
    print("\n== Je Datensatz: WER-Beitrag der Top-5-Kategorien (Punkte)")
    for ds in per_ds:
        nw = ref_words[ds]; tot = sum(by_ds[ds].values())
        top = ", ".join(f"{c.split(' (')[0]} {100*n/nw:.1f}" for c, n in by_ds[ds].most_common(5))
        print(f"{ds:22s} WER≈{100*tot/nw:5.1f}  {top}")
    print("\n== Häufigste Paare je Kategorie (Ref → Hyp, Anzahl)")
    for cat, _ in total.most_common():
        top = pairs[cat].most_common(8)
        items = "; ".join(f"{r or '∅'}→{h or '∅'} {n}" for (r, h), n in top)
        print(f"- {cat}: {items}")


if __name__ == "__main__":
    main()
