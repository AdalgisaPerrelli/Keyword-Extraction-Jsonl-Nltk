# Keyword and Key Sentence Extraction from JSONL with Python and NLTK

This project implements a text‑mining pipeline to extract key words and key sentences from documents stored in JSON Lines (`.jsonl`) format. 
It combines classic preprocessing (lowercasing, cleaning, stemming, stopword removal) with three scoring methods: term frequency, keyword matching and POS‑tag patterns.
The work was carried out as a group project within a Data Processing and Analysis course.

---

## Features

- Load documents from `.jsonl` into a `pandas.DataFrame`.
- Text preprocessing:
  - lowercasing
  - removal of unwanted characters with regular expressions
  - tokenization
  - stemming with `PorterStemmer`
  - custom stopword removal from a text file
- Term‑frequency–based ranking of words per document.
- Sentence importance scores based on:
  - presence of predefined keywords from a JSON/JSONL file
  - combinations of POS tags using NLTK.

---

## Project Structure

- `kp5k.jsonl` – input JSONL file containing the documents.
- `stopwords.txt` – custom stopword list (one word per line).
- `test.json` – JSON/JSONL file containing keywords in a `keywords` field.
- `main.py` – main script with:
  - file loading
  - preprocessing
  - scoring
  - extraction and printing of top sentences.

---

## Requirements

- Python 3.8+
- Recommended packages:
  - `pandas`
  - `jsonlines`
  - `tqdm`
  - `nltk`
