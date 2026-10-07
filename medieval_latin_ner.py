#!/usr/bin/env python3
"""
Medieval Latin Span-NER – easy-to-use wrapper for ERCDiDip/medieval-latin-span-ner

Downloads the custom architecture (span_ner_model.py) and the weights from the
Hugging Face Hub, loads the model and tags 19 diplomatic entity types.

Command line:
    python medieval_latin_ner.py charter.txt                    # table to stdout
    python medieval_latin_ner.py charter.txt --json out.json    # save as JSON
    python medieval_latin_ner.py charter.txt --html out.html    # colour-highlighted HTML
    python medieval_latin_ner.py --text "Ego Heinricus comes ..."

Python:
    from medieval_latin_ner import MedievalLatinNER
    ner = MedievalLatinNER()
    entities = ner.predict("Ego Heinricus comes ...", threshold=0.5)
"""

import argparse
import html
import importlib.util
import json
import sys

REPO_ID = "ERCDiDip/medieval-latin-span-ner"
WEIGHTS_FILE = "pytorch_model.bin"
CODE_FILE = "span_ner_model.py"

# Defaults used in the paper (best test threshold t=0.5, nested entities allowed)
DEFAULT_THRESHOLD = 0.5

LABEL_COLORS = {
    "PER": "#f4a261", "ACTOR": "#e9c46a", "TITLE": "#f6bd60", "REL": "#f28482",
    "LOC": "#84a59d", "INS": "#90be6d", "NAT": "#43aa8b", "EST": "#a7c957",
    "PROP": "#bc9c64", "LEG": "#b5a3d6", "TRANS": "#9d8df1", "TIM": "#8ecae6",
    "DAT": "#4cc9f0", "MON": "#ffd166", "TAX": "#ef8354", "COM": "#d4a373",
    "NUM": "#cdb4db", "MEA": "#bde0fe", "RELIC": "#ffafcc",
}


def _import_architecture(path):
    """Import the downloaded span_ner_model.py as a module."""
    spec = importlib.util.spec_from_file_location("span_ner_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["span_ner_model"] = module
    spec.loader.exec_module(module)
    return module


class MedievalLatinNER:
    """Loads the span-NER model once; call .predict(text) as often as needed."""

    def __init__(self, device=None, repo_id=REPO_ID, verbose=True):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        log = print if verbose else (lambda *a, **k: None)

        log(f"[1/3] Downloading architecture and weights from {repo_id} (cached after first run) ...")
        code_path = hf_hub_download(repo_id, CODE_FILE)
        weights_path = hf_hub_download(repo_id, WEIGHTS_FILE)
        self.arch = _import_architecture(code_path)

        log("[2/3] Loading tokenizers (xlm-roberta-large, bge-m3) ...")
        cfg = self.arch.Config()
        self.text_tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)
        self.label_tokenizer = AutoTokenizer.from_pretrained(cfg.LABEL_MODEL)

        log(f"[3/3] Building model on {self.device} ...")
        self.model = self.arch.SpanNERModel(cfg).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if verbose and (missing or unexpected):
            log(f"      note: {len(missing)} missing / {len(unexpected)} unexpected keys")
        self.model.eval()
        self.labels = list(self.arch.LABEL_DICT.keys())
        self.label_descriptions = dict(self.arch.LABEL_DICT)

    def predict(self, text, threshold=DEFAULT_THRESHOLD, flat_ner=False, window=150, overlap=40):
        """Return a list of entities sorted by position.

        Each entity: {label, text, start_char, end_char, start_word, end_word, score}.
        flat_ner=False keeps nested entities (e.g. PER inside ACTOR), as in the paper.

        Long texts are processed in overlapping windows of `window` word/punctuation tokens.
        (span_ner_model.py chunks by 512 *words*, but the XLM-R tokenizer truncates at 512
        *subwords*, which would leave everything after ~250 words of a charter untagged.)
        """
        toks = self.arch.char_tokenize(text)
        n = len(toks)
        if n == 0:
            return []
        step = max(1, window - overlap)
        merged = {}
        for ws in range(0, n, step):
            we = min(ws + window, n)
            c0, c1 = toks[ws]["start"], toks[we - 1]["end"]
            for e in self.model.predict(text[c0:c1], self.label_tokenizer, self.text_tokenizer,
                                        threshold, flat_ner, self.device):
                e = dict(e, start_char=e["start_char"] + c0, end_char=e["end_char"] + c0,
                         start_word=e["start_word"] + ws, end_word=e["end_word"] + ws)
                key = (e["start_char"], e["end_char"], e["label"])
                if key not in merged or e["score"] > merged[key]["score"]:
                    merged[key] = e
            if we == n:
                break
        ents = list(merged.values())
        if flat_ner:  # resolve overlaps across windows: highest score wins
            taken, flat = set(), []
            for e in sorted(ents, key=lambda e: -e["score"]):
                cover = set(range(e["start_word"], e["end_word"] + 1))
                if not cover & taken:
                    taken |= cover
                    flat.append(e)
            ents = flat
        return sorted(ents, key=lambda e: (e["start_char"], -(e["end_char"] - e["start_char"])))

    def predict_file(self, path, **kwargs):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        return text, self.predict(text, **kwargs)


def to_html(text, entities, title="Medieval Latin NER"):
    """Colour-highlighted HTML (innermost label shown; outer labels in tooltip)."""
    bounds = sorted({0, len(text), *[e["start_char"] for e in entities], *[e["end_char"] for e in entities]})
    parts = []
    for a, b in zip(bounds, bounds[1:]):
        covering = [e for e in entities if e["start_char"] <= a and e["end_char"] >= b]
        seg = html.escape(text[a:b])
        if not covering:
            parts.append(seg)
            continue
        covering.sort(key=lambda e: e["end_char"] - e["start_char"])
        inner = covering[0]
        tip = " > ".join(f'{e["label"]} ({e["score"]:.2f})' for e in reversed(covering))
        parts.append(f'<mark style="background:{LABEL_COLORS.get(inner["label"], "#ddd")}" title="{tip}">'
                     f'{seg}<sub>{inner["label"]}</sub></mark>')
    legend = " ".join(f'<mark style="background:{c}">{l}</mark>' for l, c in LABEL_COLORS.items())
    return (f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f'<body style="font-family:Georgia,serif;max-width:60em;margin:2em auto;line-height:2">'
            f'<h2>{html.escape(title)}</h2><p>{legend}</p><hr><p>{"".join(parts)}</p></body>')


def print_table(entities):
    print(f'{"LABEL":<7}{"SCORE":<7}{"CHARS":<12}TEXT')
    for e in entities:
        snippet = e["text"] if len(e["text"]) <= 90 else e["text"][:87] + "..."
        print(f'{e["label"]:<7}{e["score"]:<7.2f}{e["start_char"]}-{e["end_char"]:<8}{snippet}')


def main():
    p = argparse.ArgumentParser(description="Named entity recognition for medieval Latin charters.")
    p.add_argument("file", nargs="?", help="UTF-8 text file with a Latin charter")
    p.add_argument("--text", help="analyse this string instead of a file")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="score threshold (default 0.5)")
    p.add_argument("--flat", action="store_true", help="disallow nested entities")
    p.add_argument("--labels", nargs="+", help="only report these labels, e.g. --labels PER LOC INS")
    p.add_argument("--json", help="write entities to this JSON file")
    p.add_argument("--html", help="write highlighted HTML to this file")
    p.add_argument("--device", help="cpu / cuda (default: auto)")
    args = p.parse_args()

    if not args.file and not args.text:
        p.error("give a text file or --text")
    text = args.text if args.text else open(args.file, encoding="utf-8").read().strip()

    ner = MedievalLatinNER(device=args.device)
    entities = ner.predict(text, threshold=args.threshold, flat_ner=args.flat)
    if args.labels:
        entities = [e for e in entities if e["label"] in set(args.labels)]

    print_table(entities)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"text": text, "threshold": args.threshold, "entities": entities}, f, ensure_ascii=False, indent=2)
        print(f"JSON saved: {args.json}")
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(to_html(text, entities))
        print(f"HTML saved: {args.html}")


if __name__ == "__main__":
    main()
