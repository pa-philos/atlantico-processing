import os
import glob
import json
import pandas as pd
import numpy as np
from src.loader import DataLoader
from src.pipeline import create_preprocessing_pipeline

import argparse

# ----------------------
# Configuration (edit as needed)
# ----------------------
# Input directories
PROTOCOL_DIR = './dataset/Protocol'
OPTIONAL_DIR = './dataset/Optional'
# Output directory
OUT_DIR = './out'
# Pipeline sampling strategy: 'downsample' or 'interpolate'
SAMPLING_STRATEGY = 'downsample'
# Regex to select columns: timestamp, activityID, heart_rate, and IMU channels
REGEX_PATTERN = r'^(activityID|heart_rate|.*_(temp|acc16_[xyz]|gyro_[xyz]|mag_[xyz]))$'

# Maximum rows per output file (None to disable)
# If DOUBLE_DATASET is True, this applies to EACH of the two output files.
MAX_OUTPUT_ROWS = 10000
# Output sampling strategy: 'uniform' (evenly spaced) or 'proportional_random'
OUTPUT_SAMPLING_STRATEGY = 'uniform'  # or 'proportional_random'
# Optional random seed for reproducibility when using proportional_random
RANDOM_SEED = 42
# Enable double dataset (split each result into 2 files)
DOUBLE_DATASET = False

def parse_args():
    global OUT_DIR, MAX_OUTPUT_ROWS, OUTPUT_SAMPLING_STRATEGY, DOUBLE_DATASET
    parser = argparse.ArgumentParser(description='Preprocess dataset for ESP32.')
    parser.add_argument('--double', action='store_true', help='Enable double dataset mode (split outputs into two files)')
    parser.add_argument('--max-rows', type=int, default=MAX_OUTPUT_ROWS, help=f'Max rows per output file (default: {MAX_OUTPUT_ROWS})')
    parser.add_argument('--out-dir', type=str, default=OUT_DIR, help=f'Output directory (default: {OUT_DIR})')
    
    args = parser.parse_args()
    DOUBLE_DATASET = args.double
    if args.max_rows is not None:
        MAX_OUTPUT_ROWS = args.max_rows
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    print(f"Configuration: DOUBLE_DATASET={DOUBLE_DATASET}, MAX_OUTPUT_ROWS={MAX_OUTPUT_ROWS}, OUT_DIR={OUT_DIR}")



def export_binary_with_metadata(df, base_filename):
    """
    Exports DataFrame to a binary file with mixed types and a JSON metadata file.
    """
    # We will pack columns to save space for ESP32 readers.
    # Mapping decisions:
    # - activityID -> int8
    # - timestamp -> int32 (milliseconds)
    # - heart_rate -> uint8 (0..255)
    # - all IMU numeric channels -> float32
    type_map = {}
    out_df = df.copy()

    # Convert timestamp to milliseconds and store as int32
    if 'timestamp' in out_df.columns:
        out_df['timestamp'] = (out_df['timestamp'].astype(float) * 1000.0).round().astype('int32')
        type_map['timestamp'] = 'int32'

    for col in out_df.columns:
        if col == 'activityID':
            out_df[col] = out_df[col].fillna(0).astype('int8')
            type_map[col] = 'int8'
        elif col == 'timestamp':
            # already handled
            continue
        else:
            out_df[col] = out_df[col].astype('float32')
            type_map[col] = 'float32'

    # Build binary by row in native endianness (little-endian typical on ESP32)
    # We'll construct a bytes object by concatenating per-column numpy representations
    bin_filename = f"{base_filename}.bin"
    with open(bin_filename, 'wb') as f:
        # iterate rows and write packed binary
        for row in out_df.itertuples(index=False, name=None):
            for (col, val) in zip(out_df.columns, row):
                t = type_map[col]
                if t == 'int8':
                    f.write(np.int8(val).tobytes())
                elif t == 'uint8':
                    f.write(np.uint8(val).tobytes())
                elif t == 'int32':
                    f.write(np.int32(val).tobytes())
                elif t == 'float32':
                    f.write(np.float32(val).tobytes())
                else:
                    raise ValueError(f"Unsupported type {t} for column {col}")

    # Return schema info (do not write per-file JSON here)
    # Compute offsets and total bytes per row
    dtype_info = {
        'int8': {'size': 1, 'c_type': 'int8_t'},
        'uint8': {'size': 1, 'c_type': 'uint8_t'},
        'int32': {'size': 4, 'c_type': 'int32_t'},
        'float32': {'size': 4, 'c_type': 'float'},
    }

    current_offset = 0
    columns_meta = []
    for col in out_df.columns:
        t = type_map[col]
        info = dtype_info[t]
        columns_meta.append({
            'name': col,
            'type': t,
            'c_type': info['c_type'],
            'bytes': info['size'],
            'offset': current_offset
        })
        current_offset += info['size']

    row_size = current_offset
    print(f"Exported {bin_filename} ({len(out_df)} rows, {row_size} bytes/row)")
    return {
        'filename': os.path.basename(bin_filename),
        'rows': len(out_df),
        'columns': columns_meta,
        'total_bytes_per_row': row_size
    }

def main():
    parse_args()

    # Verify input directories
    if not os.path.exists(PROTOCOL_DIR):
        print(f"Error: Protocol directory '{PROTOCOL_DIR}' not found.")
        return
    print(f"Initializing DataLoaders for {PROTOCOL_DIR} and {OPTIONAL_DIR}...")
    loader_protocol = DataLoader(PROTOCOL_DIR)
    loader_optional = DataLoader(OPTIONAL_DIR)

    # Find subject ids from both folders
    protocol_files = glob.glob(os.path.join(PROTOCOL_DIR, 'subject*.dat'))
    optional_files = glob.glob(os.path.join(OPTIONAL_DIR, 'subject*.dat')) if os.path.exists(OPTIONAL_DIR) else []
    subject_ids = set()
    for p in protocol_files:
        subject_ids.add(int(os.path.basename(p).replace('subject','').replace('.dat','')))
    for p in optional_files:
        subject_ids.add(int(os.path.basename(p).replace('subject','').replace('.dat','')))
    subject_ids = sorted(subject_ids)
    if not subject_ids:
        print("No subject files found in protocol or optional folders.")
        return
    print(f"Found subjects: {subject_ids}")

    # Define pipeline
    print(f"Creating pipeline with strategy: {SAMPLING_STRATEGY}...")
    # Note: We pass None for selected_columns to ensure regex is used
    pipeline = create_preprocessing_pipeline(
        selected_columns=None,
        selected_columns_regex=REGEX_PATTERN,
        sampling_strategy=SAMPLING_STRATEGY
    )
    print("Collecting label set from processed data (first pass)...")
    label_values_set = set()
    for subject_id in subject_ids:
        # Protocol
        protocol_path = os.path.join(PROTOCOL_DIR, f'subject{subject_id}.dat')
        if os.path.exists(protocol_path):
            df = loader_protocol.load_subject(subject_id)
            if df is not None and not df.empty:
                processed = pipeline.fit_transform(df)
                if 'activityID' in processed.columns:
                    vals = processed['activityID'].dropna().unique().tolist()
                    for v in vals:
                        label_values_set.add(int(v))

        # Optional
        optional_path = os.path.join(OPTIONAL_DIR, f'subject{subject_id}.dat')
        if os.path.exists(optional_path):
            df = loader_optional.load_subject(subject_id)
            if df is not None and not df.empty:
                processed = pipeline.fit_transform(df)
                if 'activityID' in processed.columns:
                    vals = processed['activityID'].dropna().unique().tolist()
                    for v in vals:
                        label_values_set.add(int(v))

        # Merge
        # load both (if present) and merge
        proto_df = loader_protocol.load_subject(subject_id) if os.path.exists(os.path.join(PROTOCOL_DIR, f'subject{subject_id}.dat')) else None
        opt_df = loader_optional.load_subject(subject_id) if os.path.exists(os.path.join(OPTIONAL_DIR, f'subject{subject_id}.dat')) else None
        if (proto_df is not None and not proto_df.empty) or (opt_df is not None and not opt_df.empty):
            dfs = []
            if proto_df is not None:
                dfs.append(proto_df)
            if opt_df is not None:
                dfs.append(opt_df)
            merged_df = pd.concat(dfs, ignore_index=True)
            if 'timestamp' in merged_df.columns:
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            processed = pipeline.fit_transform(merged_df)
            if 'activityID' in processed.columns:
                vals = processed['activityID'].dropna().unique().tolist()
                for v in vals:
                    label_values_set.add(int(v))

    # Build compact encoding mapping (original label -> encoded index) and reverse mapping
    label_list = sorted(list(label_values_set))
    # Use 1-based encoded labels so 0 can represent "no label" in datasets
    label_map = {int(v): (i + 1) for i, v in enumerate(label_list)}
    # label_list is original labels ordered by encoded index

    # Prepare output directories
    os.makedirs(OUT_DIR, exist_ok=True)
    out_protocol = os.path.join(OUT_DIR, 'protocol')
    out_optional = os.path.join(OUT_DIR, 'optional')
    out_merged = os.path.join(OUT_DIR, 'merge')
    os.makedirs(out_protocol, exist_ok=True)
    os.makedirs(out_optional, exist_ok=True)
    os.makedirs(out_merged, exist_ok=True)

    # Global metadata collection
    all_files_meta = []
    schema_written = None
    label_column_name = None

    for subject_id in subject_ids:
        print(f"\n=== Subject {subject_id} ===")

        # Helper to process a source dataframe and write to target dir with a tag
        def process_and_write(df, target_dir, tag):
            nonlocal schema_written
            nonlocal label_column_name
            if df is None or df.empty:
                print(f"No data for subject {subject_id} ({tag})")
                return
            processed = pipeline.fit_transform(df)
            print(f"Processed ({tag}) shape: {processed.shape}")

            # Remap activityID to compact encoding if present
            if 'activityID' in processed.columns:
                label_column_name = 'activityID'
                # map original labels to encoded indices; missing/NaN -> 0
                processed['activityID'] = processed['activityID'].fillna(0).astype(int).map(lambda v: label_map.get(int(v), 0)).astype('int8')

            # Optionally trim dataset to at most MAX_OUTPUT_ROWS using configured sampling strategy
            limit_rows = MAX_OUTPUT_ROWS
            if DOUBLE_DATASET and MAX_OUTPUT_ROWS is not None:
                limit_rows = MAX_OUTPUT_ROWS * 2

            if limit_rows is not None and len(processed) > limit_rows:
                n = len(processed)
                keep = limit_rows
                if OUTPUT_SAMPLING_STRATEGY == 'uniform':
                    # evenly spaced indices across the dataset
                    indices = np.linspace(0, n - 1, num=keep, dtype=int)
                    processed = processed.iloc[indices].reset_index(drop=True)
                    print(f"Trimmed ({tag}) to {keep} rows from {n} rows using uniform sampling")
                elif OUTPUT_SAMPLING_STRATEGY == 'proportional_random':
                    # Stratified random sampling proportional to label frequencies (preserves label distribution)
                    rng = np.random.RandomState(RANDOM_SEED) if RANDOM_SEED is not None else np.random
                    if 'activityID' in processed.columns:
                        total = n
                        # compute exact quotas
                        counts = processed['activityID'].value_counts().to_dict()
                        quotas = {}
                        remainders = {}
                        for lbl, cnt in counts.items():
                            exact = (cnt / total) * keep
                            q = int(np.floor(exact))
                            quotas[lbl] = q
                            remainders[lbl] = exact - q
                        assigned = sum(quotas.values())
                        remaining = keep - assigned
                        # distribute remaining by largest remainders
                        if remaining > 0:
                            sorted_lbls = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
                            for i in range(remaining):
                                lbl = sorted_lbls[i % len(sorted_lbls)][0]
                                quotas[lbl] += 1

                        # sample per label
                        selected_idx = []
                        for lbl, q in quotas.items():
                            if q <= 0:
                                continue
                            subset_idx = processed[processed['activityID'] == lbl].index.values
                            if len(subset_idx) <= q:
                                selected = list(subset_idx)
                            else:
                                selected = list(rng.choice(subset_idx, size=q, replace=False))
                            selected_idx.extend(selected)

                        # If rounding/clamping caused fewer samples than needed, fill randomly from remaining
                        if len(selected_idx) < keep:
                            remaining_pool = list(set(processed.index.values) - set(selected_idx))
                            need = keep - len(selected_idx)
                            if len(remaining_pool) <= need:
                                selected_idx.extend(remaining_pool)
                            else:
                                selected_idx.extend(list(rng.choice(remaining_pool, size=need, replace=False)))

                        # Preserve original ordering as much as possible by sorting selected indices
                        selected_idx_sorted = sorted(selected_idx)
                        processed = processed.loc[selected_idx_sorted].reset_index(drop=True)
                        print(f"Trimmed ({tag}) to {keep} rows from {n} rows using proportional random sampling")
                    else:
                        # no labels available, fallback to random sampling
                        processed = processed.sample(n=keep, random_state=RANDOM_SEED).reset_index(drop=True)
                        print(f"Trimmed ({tag}) to {keep} rows from {n} rows using random sampling (no labels)")
                else:
                    raise ValueError(f"Unknown OUTPUT_SAMPLING_STRATEGY: {OUTPUT_SAMPLING_STRATEGY}")

            if DOUBLE_DATASET:
                # Split processed dataframe into two halves
                half_point = len(processed) // 2
                df1 = processed.iloc[:half_point].reset_index(drop=True)
                df2 = processed.iloc[half_point:].reset_index(drop=True)

                # Export Part 1
                out_base_1 = os.path.join(target_dir, f'subject{subject_id}_processed_1')
                meta1 = export_binary_with_metadata(df1, out_base_1)
                
                # Export Part 2
                out_base_2 = os.path.join(target_dir, f'subject{subject_id}_processed_2')
                meta2 = export_binary_with_metadata(df2, out_base_2)

                # Add meta for both
                meta_entry1 = {
                    'subject': subject_id,
                    'tag': tag,
                    'filename': os.path.relpath(meta1['filename'], start='.'),
                    'rows': meta1['rows'],
                    'bytes_per_row': meta1['total_bytes_per_row'],
                    'total_bytes': meta1['rows'] * meta1['total_bytes_per_row']
                }
                all_files_meta.append(meta_entry1)

                meta_entry2 = {
                    'subject': subject_id,
                    'tag': tag,
                    'filename': os.path.relpath(meta2['filename'], start='.'),
                    'rows': meta2['rows'],
                    'bytes_per_row': meta2['total_bytes_per_row'],
                    'total_bytes': meta2['rows'] * meta2['total_bytes_per_row']
                }
                all_files_meta.append(meta_entry2)

                # store schema once (from first one is fine)
                if schema_written is None:
                    schema_written = meta1['columns']

            else:
                out_base = os.path.join(target_dir, f'subject{subject_id}_processed')
                meta = export_binary_with_metadata(processed, out_base)
                # attach subject and tag
                meta_entry = {
                    'subject': subject_id,
                    'tag': tag,
                    'filename': os.path.relpath(meta['filename'], start='.'),
                    'rows': meta['rows'],
                    'bytes_per_row': meta['total_bytes_per_row'],
                    'total_bytes': meta['rows'] * meta['total_bytes_per_row']
                }
                all_files_meta.append(meta_entry)
                # store schema once
                if schema_written is None:
                    schema_written = meta['columns']


        # Protocol
        protocol_path = os.path.join(PROTOCOL_DIR, f'subject{subject_id}.dat')
        protocol_df = None
        if os.path.exists(protocol_path):
            protocol_df = loader_protocol.load_subject(subject_id)
            process_and_write(protocol_df, out_protocol, 'protocol')

        # Optional
        optional_path = os.path.join(OPTIONAL_DIR, f'subject{subject_id}.dat')
        optional_df = None
        if os.path.exists(optional_path):
            optional_df = loader_optional.load_subject(subject_id)
            process_and_write(optional_df, out_optional, 'optional')

        # Merge (concatenate and sort by timestamp)
        if protocol_df is not None or optional_df is not None:
            dfs = []
            if protocol_df is not None:
                dfs.append(protocol_df)
            if optional_df is not None:
                dfs.append(optional_df)
            merged_df = pd.concat(dfs, ignore_index=True)
            # sort by timestamp if present
            if 'timestamp' in merged_df.columns:
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            process_and_write(merged_df, out_merged, 'merge')

    # Once done, write a single metadata JSON in out/
    master_meta = {
        'schema': schema_written or [],
        'files': all_files_meta
    }
    # Compute top-level bytes_per_row from schema (sum of column bytes)
    if schema_written:
        try:
            bytes_per_row = sum([int(c.get('bytes', 0)) for c in schema_written])
        except Exception:
            bytes_per_row = None
    else:
        bytes_per_row = None
    master_meta['bytes_per_row'] = bytes_per_row
    # Add label information (single shared JSON)
    # `label_list` contains original labels ordered by encoded index (0..N-1)
    master_meta['label_column'] = label_column_name or 'activityID'
    master_meta['label_values'] = label_list
    # map original label -> encoded index (store keys as strings for JSON)
    master_meta['label_map'] = { str(k): v for k, v in label_map.items() }
    # Compute input_size: number of feature columns (exclude label and timestamp)
    timestamp_cols = set(['timestamp'])
    if schema_written:
        try:
            label_col_name = label_column_name or 'activityID'
            input_size = 0
            for c in schema_written:
                name = c.get('name')
                if name == label_col_name:
                    continue
                if name in timestamp_cols:
                    continue
                input_size += 1
        except Exception:
            input_size = None
    else:
        input_size = None
    master_meta['input_size'] = input_size
    master_path = os.path.join(OUT_DIR, 'metadata.json')
    with open(master_path, 'w') as f:
        json.dump(master_meta, f, indent=2)
    print(f"Wrote master metadata to {master_path}")

if __name__ == '__main__':
    main()
