# LSTM-CNN Text Classification

This repository contains a simple Keras model for binary text classification using:
Embedding + CNN + LSTM.

## Notes
- The script expects a pandas DataFrame `df` with:
  - `df.v2` = input text
  - `df.v1` = labels
- You need to load your dataset (e.g., CSV) into `df` before running.
