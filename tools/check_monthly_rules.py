import tarfile
import gzip
import json
from collections import defaultdict
from pathlib import Path
import time

# Daftar agent yang di-blacklist (case-insensitive substring/exact match)
BLACKLIST_KEYWORDS = {"dfir-iris", "proxy", "dvwa", "soc-1"}

def is_blacklisted(agent_name: str) -> bool:
    if not agent_name:
        return True
    agent_lower = agent_name.lower()
    return any(keyword in agent_lower for keyword in BLACKLIST_KEYWORDS)

def analyze_filtered_rules():
    raw_dir = Path("data/raw")
    archives = sorted(raw_dir.glob("*.tar.gz"))
    print(f"File arsip yang ditemukan: {[a.name for a in archives]}")
    print(f"Blacklist keywords: {list(BLACKLIST_KEYWORDS)}\n")

    month_rules = defaultdict(set)
    month_alerts = defaultdict(int)
    overall_rules = set()
    rule_counts = defaultdict(int)
    rule_descriptions = {}
    rule_levels = {}
    
    total_raw_logs = 0
    total_filtered_logs = 0
    filtered_agents_found = set()
    kept_agents_found = set()

    start_time = time.time()

    for archive in archives:
        print(f"Sedang membaca arsip: {archive.name} ...")
        t0 = time.time()
        archive_kept = 0
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json.gz")]
            for member in members:
                f = tar.extractfile(member)
                if f is None:
                    continue
                with gzip.open(f, "rt", encoding="utf-8", errors="replace") as gz:
                    for line in gz:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            total_raw_logs += 1

                            agent_name = str(record.get("agent", {}).get("name", ""))
                            if is_blacklisted(agent_name):
                                filtered_agents_found.add(agent_name)
                                continue

                            kept_agents_found.add(agent_name)
                            total_filtered_logs += 1
                            archive_kept += 1

                            ts = record.get("timestamp", "")
                            ym = ts[:7] if len(ts) >= 7 else "unknown"

                            rule = record.get("rule", {})
                            r_id = rule.get("id")
                            if r_id is not None:
                                r_id_str = str(r_id)
                                month_rules[ym].add(r_id_str)
                                month_alerts[ym] += 1
                                overall_rules.add(r_id_str)
                                rule_counts[r_id_str] += 1

                                if r_id_str not in rule_descriptions and "description" in rule:
                                    rule_descriptions[r_id_str] = rule.get("description", "-")
                                if r_id_str not in rule_levels and "level" in rule:
                                    rule_levels[r_id_str] = rule.get("level", "-")
                        except Exception:
                            continue
        print(f"  Selesai {archive.name}: {archive_kept:,} log lolos filter dalam {time.time() - t0:.2f} detik.")

    total_time = time.time() - start_time
    
    print("\n" + "=" * 65)
    print("STATUS FILTERING AGENT")
    print("=" * 65)
    print(f"Total Log Mentah Asli    : {total_raw_logs:,} baris")
    print(f"Total Log Setelah Filter : {total_filtered_logs:,} baris ({total_filtered_logs/total_raw_logs*100:.1f}%)")
    print(f"Log Terbuang (Blacklist) : {total_raw_logs - total_filtered_logs:,} baris")
    print(f"Agent yang Dibuang       : {sorted(list(filtered_agents_found))}")
    print(f"Agent yang Digunakan     : {sorted(list(kept_agents_found))}")
    print("=" * 65)

    print("\n" + "=" * 65)
    print(f"{'Tahun-Bulan':<14} | {'Rule Unik':<15} | {'Total Alert':<18}")
    print("=" * 65)
    for ym in sorted(month_rules.keys()):
        print(f"{ym:<14} | {len(month_rules[ym]):<15} | {month_alerts[ym]:<18,}")
    print("-" * 65)

    # Bulan terbanyak
    if month_rules:
        sorted_by_unique = sorted(month_rules.keys(), key=lambda k: len(month_rules[k]), reverse=True)
        print("\nTop 5 Bulan dengan Rule Unik Terbanyak (Setelah Blacklist):")
        for rank, ym in enumerate(sorted_by_unique[:5], 1):
            print(f"  {rank}. {ym} -> {len(month_rules[ym])} rule unik ({month_alerts[ym]:,} alerts)")

    print("\n" + "=" * 65)
    print("KESELURUHAN (SETELAH BLACKLIST)")
    print("=" * 65)
    print(f"Total Rule Unik Keseluruhan : {len(overall_rules)} rule")
    print(f"Total Alert Keseluruhan     : {total_filtered_logs:,} alerts")
    print(f"Total Waktu Pemrosesan      : {total_time:.2f} detik")
    print("=" * 65)

    print("\nTop 10 Rule ID Paling Banyak Muncul:")
    print(f"{'Rule ID':<10} | {'Level':<7} | {'Jumlah Muncul':<15} | {'Deskripsi'}")
    print("-" * 65)
    sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    for r_id, count in sorted_rules[:10]:
        desc = rule_descriptions.get(r_id, "-")
        level = str(rule_levels.get(r_id, "-"))
        desc_truncated = desc[:45] + "..." if len(desc) > 45 else desc
        print(f"{r_id:<10} | {level:<7} | {count:<15,} | {desc_truncated}")
    print("-" * 65)

if __name__ == "__main__":
    analyze_filtered_rules()
