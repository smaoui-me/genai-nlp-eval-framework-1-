import argparse
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("Missing dependency: install pandas and pyarrow first (pip install pandas pyarrow)")


def convert_file(parquet_path: Path, output_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    df.to_csv(output_path, index=False)
    print(f"Converted: {parquet_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to CSV for easier viewing in VS Code."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Parquet file(s) or directories containing parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        help="Optional output directory for converted CSV files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    parquet_files = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_dir():
            parquet_files.extend(path.rglob("*.parquet"))
        elif path.suffix == ".parquet":
            parquet_files.append(path)
        else:
            print(f"Skipping non-parquet path: {path}")

    if not parquet_files:
        raise SystemExit("No parquet files found to convert.")

    for parquet_file in parquet_files:
        target_dir = out_dir if out_dir else parquet_file.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / (parquet_file.stem + ".csv")
        convert_file(parquet_file, csv_path)


if __name__ == "__main__":
    main()
