"""
Filter the dynamic DPO dataset to remove low-quality preference pairs.

Background (from data inspection on 2026-04-13):
  - REJECTED is on average 56% the length of CHOSEN
  - REJECTED has 1.41 article cites vs CHOSEN's 3.75
  - Article-set Jaccard overlap is only 0.09
  → DPO learns spurious "longer + more citations = better" signals instead of
    genuine accuracy. In ~2/3 of inspected pairs, REJECTED was not clearly worse
    than CHOSEN — sometimes more accurate.

This script applies content-level filters to the JSONL produced by
`src/generate_rejections.py`, keeping only pairs with a clean preference signal.

Usage:
    python -m src.filter_rejections \
        --input  data/gdpr_dynamic_dpo.jsonl \
        --output data/gdpr_dynamic_dpo_filtered.jsonl

Defaults match the analysis: length ratio in [0.6, 1.4], drop subset-of-chosen
rejections, drop near-identical rejections (Jaccard > 0.7).
"""
import argparse
import json
import os
import re
import sys
from . import config


ARTICLE_PATTERN = re.compile(r"Article\s+\d+(?:\(\d+\))?(?:\([a-z]\))?")


def article_set(text: str) -> set:
    return set(ARTICLE_PATTERN.findall(text or ""))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def filter_pairs(records, *, min_ratio: float, max_ratio: float,
                 max_jaccard: float, drop_subset: bool):
    """Yield (record, reason_or_None) for each input record.
    reason_or_None is None when the record passes."""
    for r in records:
        chosen = r.get("output", "")
        rejected = r.get("rejected", "")

        if not chosen.strip() or not rejected.strip():
            yield r, "empty_field"
            continue

        # ── Filter 1: length ratio ──────────────────────────────────────────
        ratio = len(rejected) / max(len(chosen), 1)
        if ratio < min_ratio:
            yield r, f"too_short (ratio={ratio:.2f})"
            continue
        if ratio > max_ratio:
            yield r, f"too_long (ratio={ratio:.2f})"
            continue

        # ── Filter 2: rejected is subset of chosen (just an incomplete copy)
        c_arts = article_set(chosen)
        r_arts = article_set(rejected)
        if drop_subset and r_arts and r_arts.issubset(c_arts):
            yield r, "rejected_articles_subset_of_chosen"
            continue

        # ── Filter 3: too similar (high Jaccard → just a paraphrase) ────────
        j = jaccard(c_arts, r_arts)
        if j > max_jaccard:
            yield r, f"high_jaccard (j={j:.2f})"
            continue

        yield r, None


def main(args):
    if not os.path.isfile(args.input):
        sys.exit(f"Input not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} pairs from {args.input}")

    kept = []
    reasons = {}
    for rec, reason in filter_pairs(
        records,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        max_jaccard=args.max_jaccard,
        drop_subset=args.drop_subset,
    ):
        if reason is None:
            kept.append(rec)
        else:
            tag = reason.split(" ")[0]
            reasons[tag] = reasons.get(tag, 0) + 1

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    print(f"\n{'=' * 60}")
    print(f"FILTER REPORT")
    print(f"{'=' * 60}")
    print(f"  Input:    {len(records)} pairs")
    print(f"  Kept:     {len(kept)} pairs ({100 * len(kept) / len(records):.1f}%)")
    print(f"  Dropped:  {len(records) - len(kept)} pairs")
    if reasons:
        print(f"\n  Drop reasons:")
        for tag, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {tag:40s} {n:4d}  ({100 * n / len(records):.1f}%)")

    # Quality check on kept set
    if kept:
        import statistics
        c_lens = [len(r["output"]) for r in kept]
        r_lens = [len(r["rejected"]) for r in kept]
        c_arts = [len(article_set(r["output"])) for r in kept]
        r_arts_l = [len(article_set(r["rejected"])) for r in kept]
        print(f"\n  Kept-set statistics:")
        print(f"    chosen   len: mean={statistics.mean(c_lens):.0f}, "
              f"median={statistics.median(c_lens):.0f}")
        print(f"    rejected len: mean={statistics.mean(r_lens):.0f}, "
              f"median={statistics.median(r_lens):.0f}")
        print(f"    length ratio (rejected/chosen) mean: "
              f"{statistics.mean(r_lens) / statistics.mean(c_lens):.2f}")
        print(f"    chosen   articles/sample: "
              f"{sum(c_arts) / len(c_arts):.2f}")
        print(f"    rejected articles/sample: "
              f"{sum(r_arts_l) / len(r_arts_l):.2f}")

    print(f"\n  Output written to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter dynamic DPO pairs to remove length/citation bias"
    )
    parser.add_argument(
        "--input", type=str,
        default=config.DYNAMIC_DATASET_PATH,
        help="Input JSONL produced by generate_rejections.py",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(config.DATA_DIR, "gdpr_dynamic_dpo_filtered.jsonl"),
        help="Output JSONL with filtered pairs",
    )
    parser.add_argument(
        "--min_ratio", type=float, default=0.6,
        help="Drop pairs where len(rejected)/len(chosen) < min_ratio (default 0.6)",
    )
    parser.add_argument(
        "--max_ratio", type=float, default=1.4,
        help="Drop pairs where len(rejected)/len(chosen) > max_ratio (default 1.4)",
    )
    parser.add_argument(
        "--max_jaccard", type=float, default=0.7,
        help="Drop pairs whose article-set Jaccard exceeds this (default 0.7)",
    )
    parser.add_argument(
        "--drop_subset", action="store_true", default=True,
        help="Drop pairs where rejected articles ⊆ chosen articles (default True)",
    )
    args = parser.parse_args()
    main(args)
