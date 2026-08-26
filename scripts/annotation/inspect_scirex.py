"""Print concise, readable previews of normalized SciREX benchmark records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-file", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--example-id")
    parser.add_argument("--doc-id")
    parser.add_argument("--bucket")
    parser.add_argument("--show-entities", action="store_true")
    parser.add_argument("--show-sentences", action="store_true")
    parser.add_argument("--show-sections", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=1000)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.processed_file.open(encoding="utf-8")]
    filtered = [record for record in records if (
        (not args.example_id or record.get("example_id") == args.example_id)
        and (not args.doc_id or record.get("doc_id") == args.doc_id)
        and (not args.bucket or record.get("length_bucket") == args.bucket)
    )]
    if not filtered:
        raise SystemExit("No matching SciREX record found")
    index = args.index if args.index is not None else 0
    if not 0 <= index < len(filtered):
        raise SystemExit(f"--index {index} outside filtered range 0..{len(filtered)-1}")
    record = filtered[index]
    print(f"example_id: {record.get('example_id')}")
    print(f"doc_id: {record.get('doc_id')}")
    print(f"split: {record.get('source_split')}  bucket: {record.get('length_bucket')}")
    print(f"sentences: {record.get('sentence_count')}  characters: {len(record.get('text', ''))}")
    print(f"entities: {len(record.get('entities', []))}  sections: {len(record.get('sections', []))}")
    print("\nTEXT PREVIEW\n" + record.get("text", "")[:args.preview_chars])
    for enabled, key in (
        (args.show_entities, "entities"), (args.show_sentences, "sentences"),
        (args.show_sections, "sections"),
    ):
        if enabled:
            print(f"\n{key.upper()}")
            for item in record.get(key, []):
                print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
