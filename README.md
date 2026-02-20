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
- `main.py` (or notebook) – main script with:
  - file loading
  - preprocessing
  - scoring
  - extraction and printing of top sentences.

Adjust file names and paths to match your local setup.

---

## Requirements

- Python 3.8+
- Recommended packages:
  - `pandas`
  - `jsonlines`
  - `tqdm`
  - `nltk`

## Files

- `code_Keyword-Extraction.py`: Main script containing the full pipeline: JSONL loading, preprocessing, scoring (TF, keywords, POS patterns) and printing of top sentences per document
- `kp5k.jsonl`: Input JSON Lines file. Each line is a JSON object representing one document, with a `sents` field that contains a list of sentences or text units.
- `stopwords.txt`: Text file with custom stopwords, one token per line, used to filter out frequent but uninformative words after stemming/lowercasing.
- `test.json`: JSON or JSON Lines file containing keyword definitions. Each record includes a `keywords` field (e.g. phrases separated by `;`) used to score sentences by keyword presence.
