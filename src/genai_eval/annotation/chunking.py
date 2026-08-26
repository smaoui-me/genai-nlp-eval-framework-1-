"""Deterministic contiguous same-document SciREX window sampling."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from .schemas import SCHEMA_VERSION


BUCKET_NAMES = ("short", "medium", "long", "very_long")


def bucket_for_sentence_count(count: int, buckets: dict) -> str | None:
    for name in BUCKET_NAMES:
        spec = buckets[name]
        maximum = spec.get("max_sentences")
        if count >= spec["min_sentences"] and (maximum is None or count <= maximum):
            return name
    return None


def _window_from_document(doc: dict, bucket: str, start_sentence: int, sentence_count: int) -> dict:
    end_sentence = start_sentence + sentence_count
    selected_sentences = doc["sentences"][start_sentence:end_sentence]
    source_token_start = selected_sentences[0]["token_start"]
    source_token_end = selected_sentences[-1]["token_end_exclusive"]
    source_char_start = selected_sentences[0]["start_char"]
    source_char_end = selected_sentences[-1]["end_char"]
    text = doc["text"][source_char_start:source_char_end]

    sentences = [{
        "relative_sentence_index": index,
        "source_sentence_index": sentence["sentence_index"],
        "start_char": sentence["start_char"] - source_char_start,
        "end_char": sentence["end_char"] - source_char_start,
        "text": sentence["text"],
    } for index, sentence in enumerate(selected_sentences)]

    entities = [{
        "entity_id": entity["entity_id"], "label": entity["label"], "text": entity["text"],
        "start_char": entity["start_char"] - source_char_start,
        "end_char": entity["end_char"] - source_char_start,
        "source_token_start": entity["token_start"],
        "source_token_end_exclusive": entity["token_end_exclusive"],
    } for entity in doc["entities"] if (
        entity["token_start"] >= source_token_start
        and entity["token_end_exclusive"] <= source_token_end
    )]

    sections = []
    for section in doc["sections"]:
        if section["token_end_exclusive"] <= source_token_start or section["token_start"] >= source_token_end:
            continue
        clipped_start = max(section["start_char"], source_char_start)
        clipped_end = min(section["end_char"], source_char_end)
        sections.append({
            "source_section_index": section["section_index"],
            "start_char": clipped_start - source_char_start,
            "end_char": clipped_end - source_char_start,
            "source_token_start": section["token_start"],
            "source_token_end_exclusive": section["token_end_exclusive"],
            "clipped_by_window": (
                section["start_char"] < source_char_start or section["end_char"] > source_char_end
            ),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": "scirex",
        "example_id": f"{doc['doc_id']}__sent_{start_sentence:06d}_{end_sentence:06d}",
        "doc_id": doc["doc_id"],
        "source_split": doc["source_split"],
        "length_bucket": bucket,
        "sentence_count": sentence_count,
        "source_document_sentence_count": doc["sentence_count"],
        "source_sentence_start": start_sentence,
        "source_sentence_end_exclusive": end_sentence,
        "source_token_start": source_token_start,
        "source_token_end_exclusive": source_token_end,
        "text": text,
        "sentences": sentences,
        "sections": sections,
        "entities": entities,
    }


def _overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / min(a[1] - a[0], b[1] - b[0])


def build_benchmark(documents: list[dict], config: dict, allow_redistribute: bool = False) -> tuple[list[dict], dict]:
    buckets = config["buckets"]
    seed = int(config.get("seed", 42))
    max_overlap = float(config.get("sampling", {}).get("maximum_overlap_ratio", 0.5))
    rng = random.Random(seed)
    examples = []
    candidate_report = {}
    used_by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)

    # Allocate the hardest windows first. Otherwise many small early windows
    # can fragment documents and make valid 201-sentence windows impossible.
    sampling_order = ("very_long", "long", "medium", "short")
    for bucket_index, bucket in enumerate(sampling_order):
        spec = buckets[bucket]
        target = int(spec["target"])
        minimum = int(spec["min_sentences"])
        maximum = spec.get("max_sentences")
        eligible = [doc for doc in documents if doc["sentence_count"] >= minimum]
        candidate_report[bucket] = {
            "eligible_documents": len(eligible), "target": target, "selected": 0,
        }
        order = eligible[:]
        random.Random(seed + bucket_index).shuffle(order)
        # Across buckets, prefer documents with fewer existing windows. The
        # preceding shuffle gives deterministic random tie-breaking.
        order.sort(key=lambda doc: len(used_by_doc[doc["doc_id"]]))
        selected_keys = set()

        def candidate_for(doc: dict, attempts: int = 200) -> tuple[int, int] | None:
            upper = min(int(maximum), doc["sentence_count"]) if maximum is not None else minimum
            for _ in range(attempts):
                length = rng.randint(minimum, upper) if upper > minimum else minimum
                start = rng.randint(0, doc["sentence_count"] - length)
                end = start + length
                if all(
                    _overlap_ratio((start, end), prior) <= max_overlap
                    for prior in used_by_doc[doc["doc_id"]]
                ):
                    return start, end
            return None

        # First pass: one window per source document for maximum diversity.
        for doc in order:
            if len(selected_keys) >= target:
                break
            candidate = candidate_for(doc)
            if candidate is None:
                continue
            start, end = candidate
            key = (doc["doc_id"], start, end)
            selected_keys.add(key)
            used_by_doc[doc["doc_id"]].append((start, end))
            examples.append(_window_from_document(doc, bucket, start, end - start))

        # General fallback for corpora with fewer eligible documents.
        attempts = 0
        while len(selected_keys) < target and attempts < max(10000, target * 100):
            attempts += 1
            if not eligible:
                break
            doc = rng.choice(eligible)
            candidate = candidate_for(doc, attempts=20)
            if candidate is None:
                continue
            start, end = candidate
            key = (doc["doc_id"], start, end)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            used_by_doc[doc["doc_id"]].append((start, end))
            examples.append(_window_from_document(doc, bucket, start, end - start))

        candidate_report[bucket]["selected"] = len(selected_keys)
        if len(selected_keys) < target and not allow_redistribute:
            raise ValueError(
                f"Bucket {bucket!r} shortfall: selected {len(selected_keys)} of {target}. "
                f"Candidate report: {candidate_report[bucket]}"
            )

    examples.sort(key=lambda item: (
        BUCKET_NAMES.index(item["length_bucket"]), item["source_split"],
        item["doc_id"], item["source_sentence_start"], item["source_sentence_end_exclusive"],
    ))
    if not allow_redistribute and len(examples) != int(config["target_examples"]):
        raise ValueError(f"Expected {config['target_examples']} examples, built {len(examples)}")
    return examples, candidate_report
