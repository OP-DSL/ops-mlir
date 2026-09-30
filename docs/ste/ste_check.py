#!/usr/bin/env python3
"""Lightweight checker for the Simplified Technical English (ASD-STE100) versions of the documents.

It is not an official STE checker. It tests the rules that a program can test, and it prints
the lines a writer must read again.

    ste_check.py FILE.md [FILE.md ...]            style report (sentences, paragraphs, words)
    ste_check.py --compare ORIGINAL.md STE.md     fidelity report (code blocks, numbers, headings)

Style rules tested (limits follow ASD-STE100 writing rules 1.x to 3.x):
  * a sentence has at most 25 words (20 words if it is an instruction);
  * a paragraph has at most 6 sentences;
  * no semicolons, no contractions, no Latin abbreviations (e.g., i.e., etc.);
  * no passive voice, no perfect tense, no unapproved words from a short list;
  * the verbs "may", "should", "might" and the conjunctions "since", "once", "while" are flagged.
Code blocks, inline code, tables of numbers and headings are not tested as prose. Each inline code
span counts as one word.
"""
import re
import sys
from collections import Counter

# words that STE does not approve, with the preferred word
UNAPPROVED = {
    "utilize": "use", "utilise": "use", "perform": "do", "performs": "does", "performed": "did",
    "ensure": "make sure", "ensures": "makes sure", "obtain": "get", "obtains": "gets",
    "provide": "give", "provides": "gives", "execute": "run", "executes": "runs", "executed": "ran",
    "execution": "run (or operation)", "prior": "before", "subsequently": "then", "therefore": "so / because of this",
    "however": "but", "additional": "more", "approximately": "about", "via": "through / with",
    "whereas": "but", "furthermore": "also", "moreover": "also", "thus": "so", "hence": "so",
    "optimal": "best", "numerous": "many", "typically": "usually", "currently": "now",
    "require": "need", "requires": "needs", "required": "needed", "enable": "let / start", "enables": "lets",
    "multiple": "many", "various": "different", "sufficient": "enough", "per": "for each",
    "eliminate": "remove", "eliminates": "removes", "avoid": "do not / prevent", "avoids": "prevents",
    "ability": "can", "attempt": "try", "attempts": "tries", "determine": "find / decide",
    "determines": "finds / decides", "indicate": "show", "indicates": "shows", "contain": "have",
    "contains": "has", "consist": "have", "consists": "has", "generate": "make", "generates": "makes",
    "generated": "made", "utilizing": "using", "implement": "make / build", "implements": "makes",
    "implemented": "made", "correspond": "match", "corresponds": "matches", "reduce": "decrease",
}
# Some of these words are technical names in this project (a "reduction", the "reduce" operation).
# The writer can keep them when they name the thing. They stay in the report as low-priority hints.
SOFT = {"generate", "generates", "generated", "implement", "implements", "implemented", "reduce",
        "contain", "contains", "avoid", "avoids", "per", "execute", "executes", "executed", "execution",
        "multiple", "enable", "enables", "require", "requires", "required", "eliminate", "eliminates"}

MODALS = {"may": "can (ability) / must (obligation)", "should": "must / do", "might": "can / could",
          "shall": "must / will", "would": "will"}
CONJ = {"since": "because", "once": "after / when", "while": "when / during", "whilst": "when",
        "although": "but", "though": "but"}
LATIN = re.compile(r"\b(e\.g\.|i\.e\.|etc\.?|vs\.?|cf\.|viz\.|et al\.)", re.I)
CONTRACTION = re.compile(r"\b(\w+(n't|'re|'ve|'ll|'d)|it's|that's|there's|what's|let's|here's|he's|she's)\b", re.I)
PASSIVE = re.compile(r"\b(is|are|was|were|be|been|being)\s+(\w+\s+)?(\w+ed|made|done|given|shown|written|run|seen|known|taken|"
                     r"built|kept|held|sent|set|put|read|found|chosen|stored|left|used|called|passed|needed|based)\b", re.I)
PERFECT = re.compile(r"\b(has|have|had)\s+(\w+\s+)?(\w+ed|made|done|given|shown|written|seen|known|taken|built|kept|"
                     r"held|sent|put|read|found|chosen|left|been)\b", re.I)
PROGRESSIVE = re.compile(r"\b(is|are|was|were|be|been)\s+\w+ing\b", re.I)
ING_OK = {"string", "thing", "nothing", "something", "anything", "everything", "during", "morning", "ping",
          "ring", "king", "bring", "spring", "sing", "wing", "swing", "string", "ceiling", "sibling",
          "padding", "loading", "mapping", "caching", "inlining", "outlining", "scheduling", "tiling",
          "profiling", "timing", "encoding", "binding", "building", "casting", "warning", "setting",
          "settings", "heading", "ending", "beginning", "meaning", "pointing", "pairing", "hashing"}
IMPERATIVE = {"run", "set", "make", "use", "open", "put", "do", "add", "remove", "read", "write", "call", "install",
              "build", "clone", "check", "start", "stop", "send", "get", "give", "find", "enter", "type", "select",
              "copy", "create", "export", "source", "push", "pull", "submit", "wait", "keep", "look", "print",
              "compare", "apply", "try", "change", "replace", "turn", "make", "ensure"}

CODE_SPAN = re.compile(r"`[^`]*`")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def strip_inline(text):
    text = LINK.sub(r"\1", text)
    text = CODE_SPAN.sub("CODE", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def blocks(path):
    """Yield (kind, first_line_number, text) for prose paragraphs, list items, table rows and headings."""
    lines = open(path, encoding="utf-8").read().split("\n")
    fence = False
    para, start = [], 0

    def flush():
        nonlocal para, start
        if para:
            yield ("para", start, " ".join(para))
        para = []

    for n, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            yield from flush()
            fence = not fence
            continue
        if fence:
            continue
        s = line.strip()
        if not s:
            yield from flush()
            continue
        if s.startswith("#"):
            yield from flush()
            yield ("heading", n, s.lstrip("# "))
            continue
        if s.startswith("|"):
            yield from flush()
            if not re.match(r"^\|[\s:|-]+\|?$", s):
                for cell in s.strip("|").split("|"):
                    yield ("cell", n, cell.strip())
            continue
        if s.startswith(">"):
            s = s.lstrip("> ")
        m = re.match(r"^(\*|-|\d+\.)\s+(.*)", s)
        if m:
            yield from flush()
            para, start = [m.group(2)], n
            yield from flush()
            continue
        # continuation of a list item or paragraph
        if not para:
            start = n
        para.append(s)
    yield from flush()


def sentences(text):
    text = strip_inline(text)
    parts = re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9(\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def words(sentence):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-./]*", sentence)


def style_report(path, verbose=True):
    issues = []
    stats = Counter()
    for kind, n, text in blocks(path):
        if kind == "heading":
            continue
        sents = sentences(text)
        if kind == "para" and len(sents) > 6:
            issues.append((n, "paragraph", f"{len(sents)} sentences (max 6)"))
        for s in sents:
            w = words(s)
            if not w:
                continue
            stats["sentences"] += 1
            limit = 20 if w[0].lower() in IMPERATIVE and kind == "para" else 25
            if kind == "cell" and len(w) <= 8:
                continue  # short table cells are labels
            if len(w) > limit:
                issues.append((n, "length", f"{len(w)} words (max {limit}): {s[:110]}"))
            low = s.lower()
            if ";" in s:
                issues.append((n, "semicolon", s[:110]))
            if CONTRACTION.search(s):
                issues.append((n, "contraction", CONTRACTION.search(s).group(0)))
            if LATIN.search(s):
                issues.append((n, "latin", LATIN.search(s).group(0)))
            for m in PASSIVE.finditer(s):
                issues.append((n, "passive?", m.group(0)))
            for m in PERFECT.finditer(s):
                issues.append((n, "perfect?", m.group(0)))
            for m in PROGRESSIVE.finditer(s):
                issues.append((n, "progressive", m.group(0)))
            for t in re.findall(r"[a-z]+", low):
                if t in UNAPPROVED:
                    tag = "soft-word" if t in SOFT else "word"
                    issues.append((n, tag, f"'{t}' -> {UNAPPROVED[t]}"))
                elif t in MODALS:
                    issues.append((n, "modal", f"'{t}' -> {MODALS[t]}"))
                elif t in CONJ:
                    issues.append((n, "conjunction", f"'{t}' -> {CONJ[t]}"))
                elif t.endswith("ing") and len(t) > 5 and t not in ING_OK:
                    issues.append((n, "ing-word", t))
    counts = Counter(k for _, k, _ in issues)
    print(f"== {path}: {stats['sentences']} sentences")
    hard = {k: v for k, v in counts.items() if k not in ("soft-word", "ing-word")}
    print("   hard issues: " + (", ".join(f"{k} {v}" for k, v in sorted(hard.items())) or "none"))
    print("   hints: " + ", ".join(f"{k} {counts[k]}" for k in ("soft-word", "ing-word") if counts[k]) or "none")
    if verbose:
        for n, k, msg in sorted(issues):
            if k in ("soft-word", "ing-word") and verbose != "all":
                continue
            print(f"   {n:5d} {k:12s} {msg}")
    return counts


def fenced(path):
    out, cur, fence = [], [], False
    for line in open(path, encoding="utf-8").read().split("\n"):
        if line.strip().startswith("```"):
            if fence:
                out.append("\n".join(cur))
                cur = []
            fence = not fence
            continue
        if fence:
            cur.append(line)
    return out


def prose_numbers(path):
    nums = Counter()
    for kind, n, text in blocks(path):
        if kind == "heading":
            continue
        text = LINK.sub(r"\1", text)
        for m in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?%?|\d+(?:\.\d+)?x\b", text):
            nums[m.replace(",", "")] += 1
    return nums


def compare(original, ste):
    problems = 0
    a, b = fenced(original), fenced(ste)
    if len(a) != len(b):
        print(f"code blocks: {len(a)} in the original, {len(b)} in the STE version")
        problems += 1
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x.strip() != y.strip():
            print(f"code block {i} differs ({x.strip().splitlines()[0][:60]!r})")
            problems += 1
    ha = [t for k, n, t in blocks(original) if k == "heading"]
    hb = [t for k, n, t in blocks(ste) if k == "heading"]
    if len(ha) != len(hb):
        print(f"headings: {len(ha)} in the original, {len(hb)} in the STE version")
        problems += 1
    na, nb = prose_numbers(original), prose_numbers(ste)
    missing = [k for k in na if k not in nb]
    if missing:
        print(f"{len(missing)} numbers of the original do not appear in the STE version: {' '.join(sorted(missing)[:60])}")
        problems += 1
    print(f"fidelity: {'OK' if not problems else str(problems) + ' problem group(s)'}"
          f" ({len(a)} code blocks, {len(ha)} headings, {len(na)} distinct numbers in the original)")
    return problems


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    if args[0] == "--compare":
        sys.exit(1 if compare(args[1], args[2]) else 0)
    verbose = True
    if args[0] == "--summary":
        verbose, args = False, args[1:]
    elif args[0] == "--all":
        verbose, args = "all", args[1:]
    for f in args:
        style_report(f, verbose)
