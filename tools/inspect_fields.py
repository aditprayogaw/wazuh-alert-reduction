# scripts/inspect_fields.py
"""
Diagnostik sekali jalan: field apa saja yang tersedia di alert, per rule, untuk SATU periode.
Hanya mencetak NAMA field dan persentase kemunculannya, tanpa nilai (aman dibagikan).

Jalankan dari folder utama proyek:
    python -m scripts.inspect_fields            (default: 2026-03)
    python -m scripts.inspect_fields 2026-03    (format: 2026 atau 2026-03)

Membaca data/raw/ baik berupa file .json.gz maupun arsip .tar.gz. Agar cepat, file yang namanya
jelas berasal dari periode lain dilewati (bila jalur foldernya memuat tahun dan bulan seperti
.../2026/Mar/...), dan baris yang tidak memuat periode itu tidak di-parse. Ketepatan periode tetap
diperiksa dari field `timestamp` tiap alert.
"""
import argparse
import fnmatch
import gzip
import json
import re
import tarfile
from collections import Counter, defaultdict

from config import DATA_RAW_DIR, JSON_GZ_PATTERN, WATCH_RULE_IDS

DEFAULT_PERIOD = "2026-03"
SAMPLE_PER_RULE = 500
TOP_RULES = 30
# rule yang dipakai di dasar pelabelan; selalu ditampilkan bila ada
ALWAYS_SHOW = ["31153", "31106", "100101", "100102", "100103", "100105", "500550", "500554", "5715"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

count = Counter()      # total alert per rule pada periode itu
sampled = Counter()    # jumlah alert yang field-nya dicatat
keys = defaultdict(Counter)
stats = Counter()      # file dilewati / dibaca
info = {}              # deskripsi dan grup rule (teks tetap dari ruleset, bukan data log)


def flat_keys(d, prefix=""):
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from flat_keys(v, name + ".")
        else:
            yield name


def alert_fields(a):
    """Nama field konteks yang ada pada satu alert."""
    out = []
    for section in ("data", "syscheck"):
        d = a.get(section)
        if isinstance(d, dict):
            out += list(flat_keys(d, f"{section}."))
    if "full_log" in a:
        out.append("full_log")
    return out


def path_may_match(parts, period: str) -> bool:
    """False bila jalur folder memuat tahun/bulan yang jelas berbeda dari periode."""
    bits = period.split("-")
    years = [p for p in parts if re.fullmatch(r"20\d\d", p)]
    if years and bits[0] not in years:
        return False
    if len(bits) > 1:
        months = [p for p in parts if p in MONTHS]
        if months and MONTHS[int(bits[1]) - 1] not in months:
            return False
    return True


def iter_lines(period: str):
    """Baris JSON dari folder (.json.gz) maupun arsip (.tar.gz); hanya yang memuat teks periode."""
    for path in sorted(DATA_RAW_DIR.rglob("*")):
        name = path.name
        if fnmatch.fnmatch(name, JSON_GZ_PATTERN):
            if not path_may_match(path.parts, period):
                stats["file dilewati"] += 1
                continue
            stats["file dibaca"] += 1
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if period in line:
                        yield line
        elif name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r:gz") as tar:
                for member in tar:
                    if not (member.isfile() and fnmatch.fnmatch(member.name.split("/")[-1], JSON_GZ_PATTERN)):
                        continue
                    if not path_may_match(path.parts + tuple(member.name.split("/")), period):
                        stats["file dilewati"] += 1
                        continue
                    stats["file dibaca"] += 1
                    with gzip.open(tar.extractfile(member), "rt", encoding="utf-8") as fh:
                        for line in fh:
                            if period in line:
                                yield line


def scan(period: str) -> int:
    n = 0
    for line in iter_lines(period):
        try:
            a = json.loads(line)
        except ValueError:
            continue
        if not str(a.get("timestamp", "")).startswith(period):
            continue
        rule = a.get("rule", {})
        rid = str(rule.get("id"))
        count[rid] += 1
        if rid not in info:
            groups = rule.get("groups", [])
            info[rid] = (str(rule.get("description", ""))[:80],
                         ",".join(map(str, groups[:5])) if isinstance(groups, list) else str(groups))
        if sampled[rid] < SAMPLE_PER_RULE:
            sampled[rid] += 1
            for k in alert_fields(a):
                keys[rid][k] += 1
        n += 1
    return n


def show(rid: str):
    n = sampled[rid]
    common = [f"{k} ({v / n:.0%})" for k, v in keys[rid].most_common(8)]
    print(f"{rid:>8} | {count[rid]:>8} | {', '.join(common) if common else '(tidak ada field data)'}")
    desc, groups = info.get(rid, ("", ""))
    print(f"           -> {desc} | grup: {groups}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m scripts.inspect_fields")
    ap.add_argument("period", nargs="?", default=DEFAULT_PERIOD, help="2026 atau 2026-03")
    period = ap.parse_args().period

    n = scan(period)
    print(f"Periode {period}: {n:,} alert dibaca | file dibaca: {stats['file dibaca']}, "
          f"dilewati berdasarkan nama folder: {stats['file dilewati']}")
    if not n:
        print(f"Tidak ada alert periode {period} di {DATA_RAW_DIR}. "
              "Cek isi data/raw (.json.gz atau .tar.gz) dan format periode.")
    else:
        print(f"Persen dihitung dari maks {SAMPLE_PER_RULE} alert per rule. Format: rule | total alert | field terbanyak")
        shown = [r for r, _ in count.most_common(TOP_RULES)]
        extra = [str(r) for r in list(WATCH_RULE_IDS) + ALWAYS_SHOW if str(r) in count]
        for rid in shown + [r for r in dict.fromkeys(extra) if r not in shown]:
            show(rid)
        missing = [r for r in ALWAYS_SHOW if r not in count]
        if missing:
            print("Tidak ditemukan pada periode ini:", ", ".join(missing))