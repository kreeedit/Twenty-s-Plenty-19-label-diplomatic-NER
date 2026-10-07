# Twenty's Plenty – Medieval Latin Span-NER

Named entity recognition for **Medieval Latin charters** with 19 fine-grained diplomatic entity types
(persons, titles, places, institutions, legal clauses, dates, money, measures, ...).

| | |
|---|---|
| Model | [`ERCDiDip/medieval-latin-span-ner`](https://huggingface.co/ERCDiDip/medieval-latin-span-ner) (MIT) |
| Training data | [Zenodo 10.5281/zenodo.19009431](https://zenodo.org/records/19009431) (CC BY 4.0) – 458 annotated sentences from 20 charters of Monasterium.net (AT-StiAH) |
| Paper | *Twenty's Plenty: Semantic Scaffolding and Span Architecture for 19-Label NER in Medieval Latin Charters*, NLP4DH 2026, pp. 236–241 (`2026.nlp4dh-1.22.pdf`) |
| Test performance | Overlap F1 **83.4 %**, Exact F1 **67.7 %** (threshold 0.5) |

## Contents

| File | Purpose |
|---|---|
| `medieval_latin_ner.py` | Ready-to-use script / Python module: downloads the model and tags text (CLI, JSON, HTML output) |
| `medieval_latin_ner_demo.ipynb` | Demo notebook: highlighted charter, entity tables, label statistics, threshold effect |
| `PannHActa_1416_VIII_10_charter.txt` | Demo charter: Sigismund to Abbot Demetrius of Tihany, 1416 Aug 10 |
| `2026.nlp4dh-1.22.pdf` | The paper |

## Installation

```bash
pip install torch transformers huggingface_hub
pip install jupyter matplotlib pandas   # only for the notebook
```

The first run downloads the weights (~4.5 GB, cached afterwards) plus the tokenizers of
`FacebookAI/xlm-roberta-large` and `BAAI/bge-m3`. A GPU is used automatically if available; CPU works but is slower.

## Usage

### Command line

```bash
python medieval_latin_ner.py PannHActa_1416_VIII_10_charter.txt                  # table on stdout
python medieval_latin_ner.py charter.txt --json entities.json --html view.html  # save results
python medieval_latin_ner.py charter.txt --labels PER ACTOR LOC INS             # filter labels
python medieval_latin_ner.py --text "Ego Heinricus comes de Hals ..."           # inline text
```

Options: `--threshold` (default 0.5), `--flat` (no nested entities), `--device cpu|cuda`.

### Python

```python
from medieval_latin_ner import MedievalLatinNER

ner = MedievalLatinNER()
entities = ner.predict("Ego Heinricus comes de Hals dedi ecclesie Sancti Petri ...", threshold=0.5)
for e in entities:
    print(e["label"], e["score"], e["text"], e["start_char"], e["end_char"])
```

Every entity is a dict with `label`, `text`, `score`, `start_char`, `end_char`, `start_word`, `end_word`.
By default entities may be **nested** (e.g. a `PER` inside an `ACTOR`), as in the paper's annotation scheme.

### Notebook

```bash
jupyter notebook medieval_latin_ner_demo.ipynb   # spaCy-style highlighted charter, label table, statistics
```

## Labels

The 19 entity types, grouped into the four documentary layers of the annotation scheme. The English descriptions are the label prompts used by the model (*semantic scaffolding*, see the paper); they are also available as `ner.label_descriptions`.

| Label | Layer | Description (prompt given to the model) |
|---|---|---|
| `PER` | persons and roles | individual person name without any titles or roles, strictly the given name or family name |
| `ACTOR` | persons and roles | full noun phrase referring to a person including their name plus noble title, profession, geographic origin, or social status |
| `TITLE` | persons and roles | social rank, noble title, ecclesiastical office, profession, or papal rank such as comes, abbas, episcopus |
| `REL` | persons and roles | word or phrase indicating family, kinship, marriage, or social relationship like filius, uxor, frater |
| `LOC` | places and landscape | geographical place, settlement, city, diocese, region, or named territory |
| `INS` | places and landscape | monastery, abbey, church, cell, or religious order functioning as a corporate and legal body |
| `NAT` | places and landscape | natural landscape feature such as a river, stream, forest, mountain, or valley |
| `EST` | property and legal content | short physical plot of land, estate, farm, meadows, woods, vineyards, or courtyards |
| `PROP` | property and legal content | detailed boundary description of a property, grange, estate, or island including past owners, movables, and immovables |
| `LEG` | property and legal content | legal clause declaring rights, conditions, penalties, permissions, or papal commands |
| `TRANS` | property and legal content | verb or phrase denoting a core transaction, confirmation, transfer, sale, gift, or donation |
| `TIM` | time and value | time period, duration, general dating formula, indiction, or papal/royal regnal year |
| `DAT` | time and value | specific calendar date, precise year of incarnation often starting with Anno or Datum, or named liturgical feast day |
| `MON` | time and value | money, currency, coin, or monetary value such as libra, solidus, denarius, uncia, or marca |
| `TAX` | time and value | customary toll, legal tax, tithe, exaction, lucrum camere, or tribute paid to an authority |
| `COM` | time and value | harvested crops, food, physical goods, salt, wine, wax, gold, wood, or animals traded or given |
| `NUM` | time and value | number written as a word or roman numeral, including fractions and quantities |
| `MEA` | time and value | unit of measurement for land, volume, or weight such as mansus, carratas, aratrum, or talentum |
| `RELIC` | time and value | holy relic, cross, altar, or sacred object of veneration within a church |

## How it works

A custom **bi-encoder span model**: XLM-RoBERTa-large encodes the text, spans (up to 80 tokens) are represented by
start/end tokens, multi-head attention pooling and a width embedding, and scored against frozen BGE-M3 embeddings of
English label descriptions ("semantic scaffolding"). Training used an asymmetric focal + Dice loss with InfoNCE and
hard-negative mining. Because of this custom architecture the standard `pipeline()` API does not work – use
`medieval_latin_ner.py`, which loads the architecture file from the model repository.

## Long texts

`MedievalLatinNER.predict()` splits long charters into overlapping windows (`window=150`, `overlap=40` tokens) and
merges the results. This is necessary because the chunking in the repository's `span_ner_model.py` counts 512 *words*,
while the XLM-R tokenizer truncates at 512 *subwords*: with the raw `SpanNERModel.predict()`, everything after roughly the
first 250 words of a charter stays untagged. (The demo charter has 460 tokens: the raw call stopped at character 1818 of 3186.)

## Limitations

The training data are 13th-century Austrian and Central European charters. Other periods, chanceries and
orthographies (such as the 15th-century royal Hungarian mandate in the demo) can give lower accuracy, particularly for
rare labels (`PROP`, `EST`, `TAX`, ...). Check the results before using them in research.

## Citation

```bibtex
@inproceedings{kovacs2026twenty,
  title     = {Twenty's Plenty: Semantic Scaffolding and Span Architecture for 19-Label NER in Medieval Latin Charters},
  author    = {Kovács, Tamás and Consolo, Giuseppe and Vogeler, Georg},
  booktitle = {Proceedings of the 6th International Conference on Natural Language Processing for the Digital Humanities},
  pages     = {236--241},
  year      = {2026}
}

@dataset{consolo2026ner,
  title     = {Named Entity Recognition Dataset for Medieval Latin Charters},
  author    = {Consolo, Giuseppe and Kovács, Tamás and Vogeler, Georg},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19009431}
}
```
