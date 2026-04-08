#!/usr/bin/env python3
import argparse
import math
import re
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

# 支持两类常见日志格式：
# 1) cur_offset_in_result:123    value:0.456
# 2) cur_offset_in_result,target,i,ioff,joff,output: 123,4,5,6,7,0.456
LINE_PATTERNS = [
    re.compile(
        r"cur_offset_in_result\s*[:=]\s*(?P<offset>[+-]?(?:0[xX][0-9a-fA-F]+|\d+)).*?(?:value|output)\s*[:=]\s*(?P<val>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"cur_offset_in_result\s*,\s*target\s*,\s*i\s*,\s*ioff\s*,\s*joff\s*,\s*output\s*:\s*(?P<offset>[+-]?(?:0[xX][0-9a-fA-F]+|\d+))\s*,[^,]*,[^,]*,[^,]*,[^,]*,\s*(?P<val>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    ),
]

OFFSET_TOKEN_RE = re.compile(r"[+-]?(?:0[xX][0-9a-fA-F]+|\d+)")


def parse_file(path: str) -> DefaultDict[int, List[float]]:
    data: DefaultDict[int, List[float]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            matched = False
            for pat in LINE_PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                matched = True
                try:
                    off = int(m.group("offset"), 0)
                    val = float(m.group("val"))
                except (ValueError, TypeError):
                    continue
                data[off].append(val)
                break

            # 兜底：只要这一行有 cur_offset_in_result，再尝试更宽松提取
            if (not matched) and ("cur_offset_in_result" in line):
                offset_match = OFFSET_TOKEN_RE.search(line)
                nums = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", line)
                if offset_match and nums:
                    try:
                        off = int(offset_match.group(0), 0)
                        val = float(nums[-1])
                        data[off].append(val)
                    except ValueError:
                        pass

    return data


def compare(
    a: Dict[int, List[float]],
    b: Dict[int, List[float]],
    abs_threshold: float,
    rel_threshold: float,
    topk: int,
    show_missing: bool,
) -> int:
    offsets = sorted(set(a.keys()) | set(b.keys()))
    mismatches: List[Tuple[float, int, int, int, float, float, float]] = []
    # tuple: (排序键=max_abs_diff, offset, idx, len_a, va, vb, rel)

    missing_only = 0
    for off in offsets:
        va_list = a.get(off, [])
        vb_list = b.get(off, [])

        if not va_list or not vb_list:
            if show_missing:
                print(
                    f"[MISSING] offset={off} count_a={len(va_list)} count_b={len(vb_list)}"
                )
            missing_only += 1
            continue

        n = min(len(va_list), len(vb_list))
        local_max_abs = -1.0
        local_best = None

        for i in range(n):
            va = va_list[i]
            vb = vb_list[i]
            abs_diff = abs(va - vb)
            denom = max(abs(va), abs(vb), 1e-12)
            rel_diff = abs_diff / denom

            # 触发条件：绝对差超阈值，且相对差也超阈值（rel_threshold=0 时等价只看绝对差）
            if abs_diff >= abs_threshold and rel_diff >= rel_threshold:
                if abs_diff > local_max_abs:
                    local_max_abs = abs_diff
                    local_best = (off, i, len(va_list), len(vb_list), va, vb, abs_diff, rel_diff)

        if local_best is not None:
            off2, idx, len_a, len_b, va, vb, abs_d, rel_d = local_best
            mismatches.append((abs_d, off2, idx, len_a, va, vb, rel_d))

        if len(va_list) != len(vb_list) and show_missing:
            print(
                f"[COUNT] offset={off} count_a={len(va_list)} count_b={len(vb_list)}"
            )

    mismatches.sort(key=lambda x: x[0], reverse=True)

    print("=" * 88)
    print(
        f"Summary: total_offsets={len(offsets)} mismatch_offsets={len(mismatches)} missing_offsets={missing_only}"
    )
    print(
        f"Thresholds: abs>={abs_threshold} rel>={rel_threshold}  (rel=0 means disabled)"
    )
    print("=" * 88)

    shown = mismatches if topk <= 0 else mismatches[:topk]
    for abs_d, off, idx, len_a, va, vb, rel_d in shown:
        print(
            f"[DIFF] offset={off:<8d} idx={idx:<4d} "
            f"a={va:<14.7g} b={vb:<14.7g} "
            f"abs_diff={abs_d:<12.7g} rel_diff={rel_d:.6g}"
        )

    if not mismatches:
        print("No offsets exceed thresholds.")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare values by cur_offset_in_result between two log files."
    )
    p.add_argument("file_a", help="First log file path")
    p.add_argument("file_b", help="Second log file path")
    p.add_argument(
        "--abs-threshold",
        type=float,
        default=1e-2,
        help="Absolute difference threshold (default: 1e-2)",
    )
    p.add_argument(
        "--rel-threshold",
        type=float,
        default=0.0,
        help="Relative difference threshold (default: 0.0, disabled)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=200,
        help="Show top-k largest mismatch offsets (<=0 means show all, default: 200)",
    )
    p.add_argument(
        "--show-missing",
        action="store_true",
        help="Also print offsets that exist in only one file or have unequal counts",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    try:
        a = parse_file(args.file_a)
        b = parse_file(args.file_b)
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 2

    if not a:
        print("Warning: no cur_offset_in_result records found in file_a", file=sys.stderr)
    if not b:
        print("Warning: no cur_offset_in_result records found in file_b", file=sys.stderr)

    return compare(
        a,
        b,
        abs_threshold=args.abs_threshold,
        rel_threshold=args.rel_threshold,
        topk=args.topk,
        show_missing=args.show_missing,
    )


if __name__ == "__main__":
    raise SystemExit(main())
