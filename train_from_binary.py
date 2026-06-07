#!/usr/bin/env python3
"""
Quick training script that reads a binary file produced by `preprocess.py` (using
`out/metadata.json` schema) and trains a small MLP to replicate the ESP32 training
behaviour for quick experiments.

Usage: python train_from_binary.py [path/to/merge_file.bin] [path/to/metadata.json]
If paths are omitted the script will pick the first file under `./out/merge` and
`./out/metadata.json`.

This script uses scikit-learn's MLPClassifier with partial_fit to emulate
incremental training rounds. It expects the metadata to contain `schema`,
`label_values` and `label_map` (label_map: original->encoded 1-based) as
produced by the preprocessor.

Defaults tuned from your request (can be changed at the top):
    epochs = 1
    rounds = 3
    layers = [32, 200, 100, 50, 25, 18]
    learningRateWeights = 0.00625
    learningRateBiases = 0.00125  # not used separately (sklearn doesn't expose)
    seed = 42

"""
import os
import sys
import json
import struct
import glob
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Default hyperparameters (change here if desired) ---
DEFAULT_EPOCHS = 1
DEFAULT_ROUNDS = 3
DEFAULT_LAYERS = [32, 200, 100, 50, 25, 18]
DEFAULT_ACTIVATIONS = [1, 1, 1, 1, 6]  # not directly used; approximated by relu
DEFAULT_LR_WEIGHTS = 0.00625
DEFAULT_LR_BIASES = 0.00125
DEFAULT_SEED = 42
# --------------------------------------------------------

TYPE_TO_STRUCT = {
    'int8': ('b', 1),
    'uint8': ('B', 1),
    'int32': ('<i', 4),
    'float32': ('<f', 4),
}


def load_metadata(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    schema = meta.get('schema', [])
    # bytes_per_row top-level if present
    bpr = meta.get('bytes_per_row')
    label_values = meta.get('label_values', [])
    label_map = meta.get('label_map', None)
    return schema, bpr, label_values, label_map


def compute_row_size_from_schema(schema):
    s = 0
    for c in schema:
        s += int(c.get('bytes', 0))
    return s


def parse_binary_file(bin_path, schema, bytes_per_row=None, label_column_name='activityID', label_map=None):
    if bytes_per_row is None:
        bytes_per_row = compute_row_size_from_schema(schema)
    # determine which columns are inputs and which is label
    label_col = None
    timestamp_cols = set(['timestamp'])
    feature_cols = []
    # Build parse info: list of (name,type,offset,bytes,struct_fmt)
    parse_cols = []
    for c in schema:
        name = c['name']
        t = c['type']
        offset = int(c['offset'])
        b = int(c['bytes'])
        if t not in TYPE_TO_STRUCT:
            raise ValueError(f"Unsupported type in schema: {t}")
        fmt = TYPE_TO_STRUCT[t][0]
        parse_cols.append((name, t, offset, b, fmt))
        if name == label_column_name:
            label_col = name
        elif name in timestamp_cols:
            # skip timestamp from features
            pass
        else:
            feature_cols.append(name)

    if label_col is None:
        raise ValueError('Label column not found in schema')

    # open and read rows
    X_list = []
    y_list = []
    with open(bin_path, 'rb') as f:
        idx = 0
        while True:
            row = f.read(bytes_per_row)
            if not row or len(row) < bytes_per_row:
                break
            # parse label
            # find label col parse info
            for (name, t, offset, b, fmt) in parse_cols:
                if name == label_col:
                    # fmt might be '<i' which is fine for struct.unpack_from
                    try:
                        val = struct.unpack_from(fmt, row, offset)[0]
                    except struct.error:
                        # fallback: read bytes manually
                        val = 0
                    label_raw = int(val)
                    break
            # handle encoded labels: preprocessor used 1-based encoding
            if label_map is not None:
                # encoded label stored in binary is 1..N, 0 means no-label
                if label_raw == 0:
                    # skip unlabeled row
                    idx += 1
                    continue
                label_idx = label_raw - 1  # 0-based class index
            else:
                # label_raw is original activity id; map to index via label_values
                # label_values may not be present; we'll skip if not found
                raise ValueError('label_map missing in metadata; this script expects encoded labels')

            # parse features
            feats = []
            for (name, t, offset, b, fmt) in parse_cols:
                if name == label_col or name in timestamp_cols:
                    continue
                # unpack value
                try:
                    val = struct.unpack_from(fmt, row, offset)[0]
                except struct.error:
                    val = 0.0
                # convert ints to float
                if t in ('int8', 'uint8', 'int32'):
                    feats.append(float(val))
                else:
                    feats.append(float(val))
            if (idx < 5):
                print(feats)
            X_list.append(feats)
            y_list.append(label_idx)
            idx += 1

    if len(X_list) == 0:
        raise RuntimeError('No labeled rows found in binary (maybe all labels are 0)')

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def build_and_train(X, y, layers, epochs=1, rounds=1, lr=0.00625, seed=42):
    # layers: full vector [input, h1, h2, ..., output]
    input_size = layers[0]
    output_size = layers[-1]
    hidden = tuple(layers[1:-1])

    if X.shape[1] != input_size:
        print(f"Warning: metadata input size {input_size} != actual {X.shape[1]}. Using actual size.")
        input_size = X.shape[1]

    classes = np.unique(y)
    # sklearn MLPClassifier requires classes for partial_fit
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', solver='sgd',
                        learning_rate_init=lr, max_iter=1, warm_start=True, random_state=seed)

    print(f"Training: rounds={rounds}, epochs_per_round={epochs}, data_shape={X.shape}, num_classes={len(classes)}")

    for r in range(rounds):
        print(f"=== Round {r+1}/{rounds} ===")
        # shuffle
        rng = np.random.RandomState(seed + r)
        perm = rng.permutation(len(X))
        Xs = X[perm]
        ys = y[perm]
        for e in range(epochs):
            # use partial_fit on whole dataset
            if r == 0 and e == 0:
                mlp.partial_fit(Xs, ys, classes=classes)
            else:
                mlp.partial_fit(Xs, ys)
            preds = mlp.predict(Xs)
            acc = accuracy_score(ys, preds)
            print(f" round {r+1} epoch {e+1}: accuracy={acc:.4f}")

    return mlp


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('--help', '-h'):
        print("""
Usage: python train_from_binary.py [BINARY_FILE] [METADATA_FILE]

Train a small MLP classifier on binary data produced by preprocess.py.
This script emulates the ESP32 on-device training process using scikit-learn.

Arguments:
  BINARY_FILE    Path to the binary dataset file (e.g., out/merge/merged.bin).
                 Defaults to the first .bin file found in ./out/merge/.
  METADATA_FILE  Path to the metadata JSON file describing the schema.
                 Defaults to ./out/metadata.json.

Options:
  -h, --help     Show this help message and exit.

Description:
  The script reads the binary file according to the schema in metadata.json.
  It expects the schema to define column types and offsets.
  The model trained is an MLPClassifier with partial_fit to simulate
  incremental learning.
  
  Hyperparameters (epochs, layers, learning rate) are configured at the top
  of this script.
""")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Using default paths (./out/merge/*.bin, ./out/metadata.json)...")
        # continue to default logic
    
    # CLI args: optional bin file and metadata
    bin_path = None
    meta_path = None
    if len(sys.argv) >= 2:
        bin_path = sys.argv[1]
    if len(sys.argv) >= 3:
        meta_path = sys.argv[2]

    if meta_path is None:
        meta_path = os.path.join('out', 'metadata.json')
    if bin_path is None:
        # pick first merge file
        merges = glob.glob(os.path.join('out', 'merge', '*_processed.bin'))
        if not merges:
            merges = glob.glob(os.path.join('out', 'merge', '*.bin'))
        if not merges:
            print('No merge binaries found in out/merge')
            sys.exit(1)
        bin_path = merges[0]

    print(f"Using binary: {bin_path}")
    print(f"Using metadata: {meta_path}")

    schema, bpr, label_values, label_map = load_metadata(meta_path)
    if not schema:
        print('Warning: schema is empty in metadata. Cannot parse binary reliably.')
    if bpr is None:
        row_size = compute_row_size_from_schema(schema)
    else:
        row_size = bpr

    print(f"Row size: {row_size} bytes")
    X, y = parse_binary_file(bin_path, schema, bytes_per_row=row_size, label_map=label_map)
    print(f"Parsed dataset: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

    # train
    mlp = build_and_train(X, y, layers=DEFAULT_LAYERS, epochs=DEFAULT_EPOCHS, rounds=DEFAULT_ROUNDS, lr=DEFAULT_LR_WEIGHTS, seed=DEFAULT_SEED)

    # evaluate on training set
    preds = mlp.predict(X)
    print('Final training accuracy:', accuracy_score(y, preds))
    print(classification_report(y, preds))
    print('Confusion matrix:')
    print(confusion_matrix(y, preds))

    # save model using joblib
    try:
        import joblib
        joblib.dump(mlp, 'trained_model.joblib')
        print('Saved model to trained_model.joblib')
    except Exception as e:
        print('Could not save model (joblib missing?):', e)


if __name__ == '__main__':
    main()
