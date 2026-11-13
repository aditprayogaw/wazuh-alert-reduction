import tarfile
import gzip
import json
from collections import defaultdict
from pathlib import Path
import time

def scan_agents():
    raw_dir = Path("data/raw")
    archives = sorted(raw_dir.glob("*.tar.gz"))
    print(f"File arsip yang ditemukan: {[a.name for a in archives]}\n")

    agent_alert_counts = defaultdict(int)
    agent_rules = defaultdict(set)
    total_logs = 0

    t0 = time.time()
    for archive in archives:
        print(f"Membaca {archive.name}...")
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
                            total_logs += 1
                            agent = record.get("agent", {})
                            agent_name = agent.get("name")
                            if agent_name is None:
                                agent_name = "(no-agent / manager)"
                            else:
                                agent_name = str(agent_name)

                            agent_alert_counts[agent_name] += 1

                            rule_id = record.get("rule", {}).get("id")
                            if rule_id is not None:
                                agent_rules[agent_name].add(str(rule_id))
                        except Exception:
                            continue

    print(f"\nSelesai membaca {total_logs:,} log dalam {time.time() - t0:.2f} detik.\n")
    print("=" * 68)
    print(f"{'No':<4} | {'Nama Agent (agent_name)':<26} | {'Total Alert':<16} | {'Rule Unik':<10}")
    print("=" * 68)

    # Urutkan berdasarkan total alert terbanyak
    sorted_agents = sorted(agent_alert_counts.items(), key=lambda x: x[1], reverse=True)
    for idx, (agent_name, count) in enumerate(sorted_agents, 1):
        unique_rules = len(agent_rules[agent_name])
        pct = (count / total_logs) * 100
        print(f"{idx:<4} | {agent_name:<26} | {count:<10,} ({pct:4.1f}%) | {unique_rules:<10}")

    print("=" * 68)

if __name__ == "__main__":
    scan_agents()

