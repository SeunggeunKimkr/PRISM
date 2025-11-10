import argparse, glob, json, os, re, sys
from typing import List, Tuple, Dict, Any

RANK_RE = re.compile(r"\.rank(\d+)$")

def find_shards(prefix: str) -> List[Tuple[int, str]]:
    """Find files like '<prefix>.rankN' and return [(rank, path), ...] sorted by rank."""
    paths = glob.glob(f"{prefix}.rank*")
    shards = []
    for p in paths:
        m = RANK_RE.search(p)
        if m:
            shards.append((int(m.group(1)), p))
    shards.sort(key=lambda x: x[0])
    return shards

def merge_to_jsonl(shards: List[Tuple[int, str]], out_path: str, encoding: str = "utf-8") -> int:
    """Stream-concatenate shard lines into a single JSONL file. Returns number of lines written."""
    if not shards:
        raise FileNotFoundError("No shard files found to merge.")
    tmp_path = out_path + ".tmp"
    written = 0
    with open(tmp_path, "w", encoding=encoding) as fout:
        for rank, shard in shards:
            with open(shard, "r", encoding=encoding) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
                        written += 1
    os.replace(tmp_path, out_path)
    return written

def load_records(shards: List[Tuple[int, str]], encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """Load all JSONL records from shards into memory (for sorting/deduping/JSON array output)."""
    recs: List[Dict[str, Any]] = []
    for rank, shard in shards:
        with open(shard, "r", encoding=encoding) as fin:
            for line in fin:
                s = line.strip()
                if not s:
                    continue
                try:
                    recs.append(json.loads(s))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSON in {shard}: {e}\nLine: {s[:200]}...") from e
    return recs

def dedupe_by_key(recs: List[Dict[str, Any]], key: str, keep: str = "last") -> List[Dict[str, Any]]:
    """Deduplicate by a key (e.g., 'task_id'). keep='first' or 'last'."""
    seen: Dict[Any, int] = {}
    out: List[Dict[str, Any]] = []
    if keep == "first":
        for r in recs:
            k = r.get(key)
            if k not in seen:
                seen[k] = 1
                out.append(r)
    else:
        # keep last: overwrite and then rebuild order
        tmp: Dict[Any, Dict[str, Any]] = {}
        for r in recs:
            tmp[r.get(key)] = r
        # Preserve relative order of the last occurrences
        keys_in_order = []
        seen2 = set()
        for r in recs:
            k = r.get(key)
            if k in tmp and k not in seen2 and tmp[k] is r:
                keys_in_order.append(k)
                seen2.add(k)
        out = [tmp[k] for k in keys_in_order]
    return out

def main():
    ap = argparse.ArgumentParser(description="Merge LLaDA shard files into JSONL and/or JSON.")
    ap.add_argument("--prefix", required=True,
                    help="Common prefix of shard files (e.g., runs/sample_PRISM_30_remasking_5)")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--jsonl", action="store_true",
                    help="Write merged JSONL file (default if neither --jsonl nor --json is given).")
    ap.add_argument("--json", action="store_true",
                    help="Also write a JSON array file (loads all records into memory).")
    ap.add_argument("--sort-by", choices=["task_id", "none"], default="none",
                    help="Sort JSON array output by a key (only applies to --json).")
    ap.add_argument("--dedupe-by", choices=["task_id", "none"], default="none",
                    help="Deduplicate JSON array output by a key (only applies to --json).")
    ap.add_argument("--keep", choices=["first", "last"], default="last",
                    help="When deduping, keep the first or last occurrence (only with --dedupe-by).")
    ap.add_argument("--delete-shards", action="store_true",
                    help="Delete shard files after successful merge.")
    args = ap.parse_args()

    prefix = args.prefix
    jsonl_out = f"{prefix}.jsonl"
    json_out  = f"{prefix}.json"

    shards = find_shards(prefix)
    if not shards:
        print(f"No shards found matching '{prefix}.rank*'.", file=sys.stderr)
        sys.exit(1)

    # Default to JSONL if neither flag is provided
    do_jsonl = args.jsonl or not (args.jsonl or args.json)
    do_json = args.json

    if do_jsonl:
        print(f"[1/2] Merging {len(shards)} shards -> {jsonl_out} ...")
        n = merge_to_jsonl(shards, jsonl_out, encoding=args.encoding)
        print(f"  Wrote {n} lines to {jsonl_out}")

    if do_json:
        print(f"[2/2] Building JSON array -> {json_out} ... (this loads all records into memory)")
        recs = load_records(shards, encoding=args.encoding)
        if args.dedupe_by != "none":
            recs = dedupe_by_key(recs, args.dedupe_by, keep=args.keep)
            print(f"  After dedupe by '{args.dedupe_by}' (keep {args.keep}): {len(recs)} records")
        if args.sort_by != "none":
            key = args.sort_by
            recs.sort(key=lambda r: r.get(key))
            print(f"  Sorted by '{key}'")

        tmp_path = json_out + ".tmp"
        with open(tmp_path, "w", encoding=args.encoding) as fout:
            json.dump(recs, fout, ensure_ascii=False, indent=2)
        os.replace(tmp_path, json_out)
        print(f"  Wrote {len(recs)} objects to {json_out}")

    if args.delete_shards:
        for _, p in shards:
            try:
                os.remove(p)
            except OSError as e:
                print(f"  Warning: failed to remove {p}: {e}")
        print("Deleted shard files.")

    print("Done.")

if __name__ == "__main__":
    main()
