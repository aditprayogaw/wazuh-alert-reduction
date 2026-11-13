import tarfile
import gzip
import json
from collections import defaultdict, Counter
from pathlib import Path
import time

def profile_agents():
    raw_dir = Path("data/raw")
    archives = sorted(raw_dir.glob("*.tar.gz"))
    print(f"Menganalisis profil dan dominasi informasi tiap agent dari arsip: {[a.name for a in archives]} ...\n")

    # Data structures per agent
    agent_counts = Counter()
    agent_decoders = defaultdict(Counter)
    agent_rules = defaultdict(Counter)
    agent_locations = defaultdict(Counter)
    agent_levels = defaultdict(Counter)
    rule_desc_map = {}

    t0 = time.time()
    for archive in archives:
        print(f"Reading {archive.name}...")
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
                            agent_name = str(record.get("agent", {}).get("name", "(unknown)"))
                            agent_counts[agent_name] += 1

                            # Decoder
                            decoder = record.get("decoder", {}).get("name", "(unknown)")
                            agent_decoders[agent_name][decoder] += 1

                            # Location
                            location = record.get("location", "(unknown)")
                            agent_locations[agent_name][location] += 1

                            # Rule
                            rule = record.get("rule", {})
                            r_id = str(rule.get("id", "(unknown)"))
                            r_level = rule.get("level", 0)
                            agent_rules[agent_name][r_id] += 1
                            agent_levels[agent_name][r_level] += 1

                            if r_id not in rule_desc_map and "description" in rule:
                                rule_desc_map[r_id] = rule.get("description", "-")
                        except Exception:
                            continue

    print(f"\nSelesai membaca seluruh log dalam {time.time() - t0:.2f} detik.\n")

    total_all_logs = sum(agent_counts.values())

    print("=" * 80)
    print("PROFIL LENGKAP & KARAKTERISTIK TIAP AGENT (2025 - 2026)")
    print("=" * 80)

    # Urutkan agent berdasarkan total alert terbanyak
    for agent_name, total_alerts in agent_counts.most_common():
        pct = (total_alerts / total_all_logs) * 100
        unique_rules_cnt = len(agent_rules[agent_name])
        print(f"\n▶ AGENT: [{agent_name}]")
        print(f"  Total Alert : {total_alerts:,} baris ({pct:.2f}% dari total) | Variasi Rule: {unique_rules_cnt} unik")
        
        # Top 3 Decoders
        print("  --- Top 3 Decoder (Sumber Log): ---")
        for dec, cnt in agent_decoders[agent_name].most_common(3):
            print(f"      * {dec:<25}: {cnt:,} ({cnt/total_alerts*100:.1f}%)")

        # Top 3 Locations
        print("  --- Top 3 Location (File / Service): ---")
        for loc, cnt in agent_locations[agent_name].most_common(3):
            print(f"      * {loc:<35}: {cnt:,} ({cnt/total_alerts*100:.1f}%)")

        # Top 3 Rule Levels
        print("  --- Distribusi Severity Level Tertinggi: ---")
        for lvl, cnt in agent_levels[agent_name].most_common(3):
            print(f"      * Level {lvl:<2}: {cnt:,} ({cnt/total_alerts*100:.1f}%)")

        # Top 3 Rules
        print("  --- Top 3 Rule ID Dominan: ---")
        for r_id, cnt in agent_rules[agent_name].most_common(3):
            desc = rule_desc_map.get(r_id, "-")
            desc_short = desc[:50] + "..." if len(desc) > 50 else desc
            print(f"      * Rule {r_id:<7} ({cnt:,} alert | {cnt/total_alerts*100:.1f}%): {desc_short}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    profile_agents()

