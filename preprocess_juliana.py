import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


INPUT_DIR = './data_juliana'
OUT_DIR = './out_juliana'
COMBINED_CSV_NAME = 'combined_raw.csv'
PREPROCESSED_CSV_NAME = 'combined_normalized.csv'
METADATA_NAME = 'metadata.json'
BIN_NAME = 'combined_normalized.bin'
SHARD_COUNTS = (5, 10)
RANDOM_SEED = 42

TIMESTAMP_COLUMN = 'timestamp'
TARGET_COLUMN = 'ocupada'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocess the Juliana CSV dataset.')
    parser.add_argument('--input-dir', type=str, default=INPUT_DIR, help=f'Input directory with CSV files (default: {INPUT_DIR})')
    parser.add_argument('--out-dir', type=str, default=OUT_DIR, help=f'Output directory (default: {OUT_DIR})')
    parser.add_argument('--combined-name', type=str, default=COMBINED_CSV_NAME, help=f'Filename for the combined raw CSV (default: {COMBINED_CSV_NAME})')
    parser.add_argument('--normalized-name', type=str, default=PREPROCESSED_CSV_NAME, help=f'Filename for the normalized CSV (default: {PREPROCESSED_CSV_NAME})')
    parser.add_argument('--metadata-name', type=str, default=METADATA_NAME, help=f'Filename for the metadata JSON (default: {METADATA_NAME})')
    parser.add_argument('--binary-name', type=str, default=BIN_NAME, help=f'Filename for the binary export (default: {BIN_NAME})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help=f'Random seed for deterministic sharding (default: {RANDOM_SEED})')
    parser.add_argument('--shard-counts', type=int, nargs='+', default=list(SHARD_COUNTS), help='Shard counts to export, for example: 5 10')
    parser.add_argument('--skip-binary', action='store_true', help='Only export CSV files and metadata, without writing the binary file')
    parser.add_argument('--skip-csv', action='store_true', help='Only export binary files and metadata, without writing the CSV files')
    return parser.parse_args()


def load_and_combine_csvs(input_dir: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {input_dir}')

    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        frame['source_file'] = os.path.basename(csv_file)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def add_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f'Missing required column: {TIMESTAMP_COLUMN}')

    result = df.copy()
    timestamp = pd.to_datetime(result[TIMESTAMP_COLUMN], errors='coerce')
    if timestamp.isna().any():
        bad_rows = int(timestamp.isna().sum())
        raise ValueError(f'Failed to parse {bad_rows} timestamp values')

    result['current_time'] = (
        timestamp.dt.hour * 3600
        + timestamp.dt.minute * 60
        + timestamp.dt.second
        + timestamp.dt.microsecond / 1_000_000.0
    )
    result['current_day_of_week'] = timestamp.dt.dayofweek.astype('int64')
    result = result.drop(columns=[TIMESTAMP_COLUMN])

    feature_columns = [column for column in result.columns if column != TARGET_COLUMN]
    ordered_columns = [
        'current_time',
        'current_day_of_week',
        *[column for column in result.columns if column.startswith('average_')],
        TARGET_COLUMN,
    ]
    ordered_columns = [column for column in ordered_columns if column in result.columns]

    return result[ordered_columns], feature_columns


def min_max_normalize(df: pd.DataFrame, feature_columns: List[str]) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    normalized = df.copy()
    stats: Dict[str, Dict[str, float]] = {}

    for column in feature_columns:
        series = pd.to_numeric(normalized[column], errors='coerce')
        if series.isna().any():
            raise ValueError(f'Column {column} contains non-numeric values after preprocessing')

        min_value = float(series.min())
        max_value = float(series.max())
        stats[column] = {'min': min_value, 'max': max_value}

        if max_value == min_value:
            normalized[column] = 0.0
        else:
            normalized[column] = (series - min_value) / (max_value - min_value)

    normalized[TARGET_COLUMN] = pd.to_numeric(normalized[TARGET_COLUMN], errors='coerce').fillna(0).astype('uint8')

    return normalized, stats


def export_binary_with_metadata(df: pd.DataFrame, base_filename: str) -> Dict[str, object]:
    type_map: Dict[str, str] = {}
    out_df = df.copy()

    for column in out_df.columns:
        if column == TARGET_COLUMN:
            out_df[column] = pd.to_numeric(out_df[column], errors='coerce').fillna(0).astype('uint8')
            type_map[column] = 'uint8'
        else:
            out_df[column] = pd.to_numeric(out_df[column], errors='coerce').astype('float32')
            type_map[column] = 'float32'

    bin_filename = f'{base_filename}.bin'
    with open(bin_filename, 'wb') as handle:
        for row in out_df.itertuples(index=False, name=None):
            for column, value in zip(out_df.columns, row):
                if column == TARGET_COLUMN:
                    handle.write(np.uint8(value).tobytes())
                else:
                    handle.write(np.float32(value).tobytes())

    dtype_info = {
        'uint8': {'size': 1, 'c_type': 'uint8_t'},
        'float32': {'size': 4, 'c_type': 'float'},
    }

    current_offset = 0
    columns_meta = []
    for column in out_df.columns:
        column_type = type_map[column]
        info = dtype_info[column_type]
        columns_meta.append(
            {
                'name': column,
                'type': column_type,
                'c_type': info['c_type'],
                'bytes': info['size'],
                'offset': current_offset,
            }
        )
        current_offset += info['size']

    print(f'Exported {bin_filename} ({len(out_df)} rows, {current_offset} bytes/row)')
    return {
        'filename': os.path.basename(bin_filename),
        'rows': len(out_df),
        'columns': columns_meta,
        'total_bytes_per_row': current_offset,
    }


def stratified_shard_indices(labels: pd.Series, shard_count: int, seed: int) -> List[np.ndarray]:
    if shard_count <= 0:
        raise ValueError('shard_count must be positive')

    rng = np.random.default_rng(seed)
    shard_buckets: List[List[int]] = [[] for _ in range(shard_count)]

    for label_value in sorted(labels.dropna().unique().tolist()):
        label_indices = labels[labels == label_value].index.to_numpy()
        if len(label_indices) == 0:
            continue
        shuffled = label_indices.copy()
        rng.shuffle(shuffled)
        for position, row_index in enumerate(shuffled):
            shard_buckets[position % shard_count].append(int(row_index))

    return [np.array(sorted(bucket), dtype=int) for bucket in shard_buckets]


def export_shards(df: pd.DataFrame, out_dir: str, shard_count: int, seed: int, args: argparse.Namespace) -> Dict[str, object]:
    shard_dir = os.path.join(out_dir, f'shards_{shard_count}')
    os.makedirs(shard_dir, exist_ok=True)

    shard_indices = stratified_shard_indices(df[TARGET_COLUMN], shard_count, seed)
    shard_entries = []
    shard_schema = None
    shard_bytes_per_row = None

    for shard_number, indices in enumerate(shard_indices, start=1):
        shard_df = df.iloc[indices].reset_index(drop=True)
        shard_name = f'shards_{shard_count}_{shard_number:02d}'
        csv_path = os.path.join(shard_dir, f'{shard_name}.csv')
        bin_base = os.path.join(shard_dir, shard_name)

        if not args.skip_csv:
            shard_df.to_csv(csv_path, index=False)
            
        binary_meta = export_binary_with_metadata(shard_df, bin_base)
        if shard_schema is None:
            shard_schema = binary_meta['columns']
            shard_bytes_per_row = binary_meta['total_bytes_per_row']

        shard_entries.append(
            {
                'shard': shard_number,
                'rows': int(len(shard_df)),
                'csv': os.path.basename(csv_path),
                'bin': binary_meta['filename'],
                'bytes_per_row': binary_meta['total_bytes_per_row'],
                'label_counts': {str(k): int(v) for k, v in shard_df[TARGET_COLUMN].value_counts().sort_index().items()},
            }
        )

    return {
        'shard_count': shard_count,
        'directory': os.path.relpath(shard_dir, out_dir),
        'shards': shard_entries,
        'schema': shard_schema,
        'bytes_per_row': shard_bytes_per_row,
    }


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    combined = load_and_combine_csvs(args.input_dir)
    source_files = sorted(combined['source_file'].unique().tolist())
    print(f'Loaded {len(combined)} rows from {len(source_files)} CSV files')

    combined = combined.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    combined_export = combined.drop(columns=['source_file'])
    combined_output_path = os.path.join(args.out_dir, args.combined_name)
    if not args.skip_csv:
        combined_export.to_csv(combined_output_path, index=False)
        print(f'Wrote combined raw CSV to {combined_output_path}')

    processed, feature_columns = add_time_features(combined_export)
    normalized, normalization_stats = min_max_normalize(processed, feature_columns)

    normalized_output_path = os.path.join(args.out_dir, args.normalized_name)
    if not args.skip_csv:
        normalized.to_csv(normalized_output_path, index=False)
        print(f'Wrote normalized CSV to {normalized_output_path}')

    # Prepare metadata with schema and label_column at the top for faster/easier parsing on ESP32
    metadata = {
        'label_column': TARGET_COLUMN,
        'schema': [], # placeholder, filled below
        'bytes_per_row': 0, # placeholder, filled below
        'input_dir': os.path.abspath(args.input_dir),
        'combined_csv': args.combined_name,
        'normalized_csv': args.normalized_name,
        'feature_columns': feature_columns,
        'normalization': normalization_stats,
        'rows': int(len(normalized)),
        'source_files': source_files,
        'seed': int(args.seed),
    }

    binary_schema = None
    if not args.skip_binary:
        binary_base = os.path.join(args.out_dir, os.path.splitext(args.binary_name)[0])
        binary_meta = export_binary_with_metadata(normalized, binary_base)
        binary_schema = binary_meta['columns']
        metadata['binary'] = {
            'filename': binary_meta['filename'],
            'rows': binary_meta['rows'],
            'bytes_per_row': binary_meta['total_bytes_per_row'],
            'schema': binary_schema,
        }

    shard_counts = []
    for shard_count in args.shard_counts:
        shard_counts.append(export_shards(normalized, args.out_dir, int(shard_count), int(args.seed), args))
    metadata['shards'] = shard_counts

    if binary_schema is None:
        for shard_group in shard_counts:
            shard_schema = shard_group.get('schema')
            if shard_schema:
                binary_schema = shard_schema
                break

    metadata['schema'] = binary_schema or []
    metadata['bytes_per_row'] = sum(column['bytes'] for column in metadata['schema']) if metadata['schema'] else 0

    metadata_path = os.path.join(args.out_dir, args.metadata_name)
    with open(metadata_path, 'w') as handle:
        json.dump(metadata, handle, indent=2)
    print(f'Wrote metadata to {metadata_path}')


if __name__ == '__main__':
    main()
