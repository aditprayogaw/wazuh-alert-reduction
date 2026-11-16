"""
tools/simulate_automation.py
----------------------------
Simulasi pipeline automasi SOC kampus secara real-time / streaming:
1. Log stream masuk satu per satu.
2. Penentuan Triase:
   - Rule Dikenal -> Tabel Kebijakan (Dismiss / Review / Escalate)
   - Rule Baru    -> Model AI Random Forest (Dismiss vs Attention dengan Confidence Guardrail)
3. Deduplikasi Temporal (Sliding Window 10 Menit):
   - First Alert  -> Kirim notifikasi instan ke Chatbot (Telegram/Discord/Slack)
   - Repeated Log -> Diredam (Suppressed), counter diakumulasikan
   - Window Expire-> Kirim pesan ringkasan update agregasi
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Pastikan root project ada di sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CONFIDENCE_THRESHOLD_DISMISS
from src.labeling import load_rule_labels
from src.classification import load_artifacts, encode_features, predict_new_rules_triase


def format_bot_card(header: str, body: dict, border_char="─", color_icon="🔔") -> str:
    lines = [
        f"\n┌{border_char * 60}┐",
        f"│ {color_icon} [CHATBOT TELEGRAM SOC INSTIKI] - {header:<30} │",
        f"├{border_char * 60}┤",
    ]
    for k, v in body.items():
        line_content = f"{k:<18}: {str(v)}"
        lines.append(f"│  • {line_content:<55}│")
    lines.append(f"└{border_char * 60}┘\n")
    return "\n".join(lines)


def run_simulation():
    print("=" * 68)
    print("SIMULASI PIPELINE AUTOMASI SOC KAMPUS (INSTIKI) BERBASIS AI")
    print("=" * 68)
    print("Memuat Tabel Kebijakan (51 Rule) & Model Random Forest...")

    policy_map = load_rule_labels()
    model, encoders = load_artifacts()

    print(f"✅ Tabel Kebijakan siap: {len(policy_map)} rule.")
    print("✅ Model AI & Encoders siap.")
    print("=" * 68)
    print("Menyiapkan skenario streaming log serangan & aktivitas kampus...\n")

    # Skenario Log Stream Realistis di Kampus
    # Format: (detik_ke, rule_id, level, agent, src_ip, decoder, location, groups, desc)
    scenario_logs = [
        # 1. Log Noise Biasa (Rule Dikenal - Dismiss)
        {
            "t": 0,
            "rule_id": "52507",
            "rule_level": 3,
            "agent_name": "sads",
            "src_ip": "10.0.0.12",
            "decoder_name": "json",
            "location": "/var/log/syslog",
            "rule_groups": "system, status",
            "description": "Routine periodic keepalive check",
        },
        # 2. Log Serangan Eksploitasi Kritis Pertama (Rule Dikenal - Escalate)
        {
            "t": 3,
            "rule_id": "203",
            "rule_level": 10,
            "agent_name": "siput",
            "src_ip": "103.112.50.24",
            "decoder_name": "web-accesslog",
            "location": "/var/log/nginx/access.log",
            "rule_groups": "web, attack, exploit",
            "description": "Remote Code Execution / Critical Exploit Attempt",
        },
        # 3. Rentetan 1.500 log serangan yang sama dalam 2 menit berikutnya
        {
            "t": 15,
            "rule_id": "203",
            "rule_level": 10,
            "agent_name": "siput",
            "src_ip": "103.112.50.24",
            "decoder_name": "web-accesslog",
            "location": "/var/log/nginx/access.log",
            "rule_groups": "web, attack, exploit",
            "description": "Remote Code Execution / Critical Exploit Attempt (repeated #2)",
            "repeat_count": 1499,
        },
        # 4. Log Rule Baru Asing yang Belum Pernah Ada di Tabel (Rule Baru -> AI Triage -> Attention)
        {
            "t": 45,
            "rule_id": "99105",  # Rule baru asing
            "rule_level": 8,
            "agent_name": "pusatkarir",
            "src_ip": "45.130.82.12",
            "decoder_name": "apache-errorlog",
            "location": "/var/log/apache2/error.log",
            "rule_groups": "web, attack, sql_injection",
            "description": "New Unknown SQLi signature detected on campus portal",
        },
        # 5. Log Rule Baru Tingkat Rendah/Info (Rule Baru -> AI Triage -> Dismiss)
        {
            "t": 60,
            "rule_id": "99201",  # Rule baru noise
            "rule_level": 2,
            "agent_name": "e-kuesioner",
            "src_ip": "192.168.1.50",
            "decoder_name": "json",
            "location": "/var/log/syslog",
            "rule_groups": "system, status",
            "description": "Periodic system health ping",
        },
    ]

    # In-memory suppression cache: key -> {"start_time": dt, "count": int, "decision": str}
    suppression_cache = {}
    WINDOW_MINUTES = 10

    base_time = datetime(2026, 3, 15, 10, 0, 0)

    for item in scenario_logs:
        current_time = base_time + timedelta(seconds=item["t"])
        rule_id = str(item["rule_id"])
        agent = item["agent_name"]
        ip = item.get("src_ip", "-")
        desc = item["description"]
        rule_key = f"{rule_id}|{agent}|{ip}"
        repeats = item.get("repeat_count", 1)

        time_str = current_time.strftime("%H:%M:%S")

        print(f"\n[STREAM DETIK {item['t']:>2}s - {time_str}] Datang {repeats} log:")
        print(f"   Detail: Rule={rule_id} (Lvl {item['rule_level']}) | Agen={agent} | IP={ip}")

        # 1. Cek Suppression Cache
        if rule_key in suppression_cache:
            cache_entry = suppression_cache[rule_key]
            elapsed_min = (current_time - cache_entry["start_time"]).total_seconds() / 60.0
            if elapsed_min <= WINDOW_MINUTES:
                cache_entry["count"] += repeats
                print(f"   🔕 [SUPPRESSED] Terdeteksi pola identik aktif dalam jendela 10m.")
                print(f"      Status: Log ke-2 s.d. #{cache_entry['count']} DIREDAM. (Chatbot tidak dispam!)")
                continue

        # 2. Jika pola baru, lakukan Triase
        if rule_id in policy_map:
            sumber_triase = "TABEL KEBIJAKAN MANUSIA"
            decision = policy_map[rule_id]
            confidence_str = "100% (Deterministik)"
        else:
            sumber_triase = "MODEL AI RANDOM FOREST (Rule Baru)"
            df_unknown = pd.DataFrame([{
                "rule_level": item["rule_level"],
                "decoder_name": item["decoder_name"],
                "location": item["location"],
                "rule_groups": item["rule_groups"],
            }])
            X_unk = encode_features(df_unknown, encoders)
            y_pred, y_conf = predict_new_rules_triase(model, X_unk, encoders)
            decision = y_pred[0]
            confidence_str = f"{y_conf[0]*100:.1f}%"

        # Simpan ke cache
        suppression_cache[rule_key] = {
            "start_time": current_time,
            "count": repeats,
            "decision": decision,
            "agent": agent,
            "ip": ip,
            "rule_id": rule_id,
            "desc": desc,
        }

        # 3. Eksekusi Aksi Chatbot Berdasarkan Keputusan
        if decision == "Dismiss":
            print(f"   🗑️  [DISMISS] Triase: {sumber_triase} -> {decision} ({confidence_str})")
            print(f"      Aksi: Log noise/info disenyapkan otomatis. Tidak ada notifikasi ke analis.")
        elif decision == "Escalate":
            print(f"   🚨 [ESCALATE] Triase: {sumber_triase} -> {decision} ({confidence_str})")
            card = format_bot_card(
                header="🚨 INSIDEN KRITIS (SEGERA TINDAK)",
                body={
                    "Waktu Deteksi": time_str,
                    "Rule ID": f"{rule_id} (Level {item['rule_level']})",
                    "Target Agen": agent,
                    "Attacker IP": ip,
                    "Deskripsi": desc,
                    "Keputusan": "ESCALATE ⚡",
                    "Sumber": sumber_triase,
                    "Aksi Analis": "[ Blokir IP via Firewall ] [ Isolasi Agen ]",
                },
                color_icon="🚨",
            )
            print(card)
        else:  # Review / Attention
            print(f"   ⚠️  [{decision.upper()}] Triase: {sumber_triase} -> {decision} ({confidence_str})")
            card = format_bot_card(
                header=f"⚠️ PERLU INVESTIGASI ({decision.upper()})",
                body={
                    "Waktu Deteksi": time_str,
                    "Rule ID": f"{rule_id} (Level {item['rule_level']})",
                    "Target Agen": agent,
                    "Attacker IP": ip,
                    "Deskripsi": desc,
                    "Keputusan": f"{decision.upper()} 📋",
                    "Keyakinan AI": confidence_str,
                    "Sumber": sumber_triase,
                    "Aksi Analis": "[ Tandai Sedang Ditinjau ] [ Abaikan ]",
                },
                color_icon="⚠️",
            )
            print(card)

    # 4. Simulasi Jendela 10 Menit Berakhir (Summary Update)
    print("\n" + "=" * 68)
    print("[SIMULASI 10 MENIT KEMUDIAN - 10:10:00]")
    print("Jendela 10 menit berakhir untuk pola yang sedang aktif...")
    print("=" * 68)

    for r_key, c_data in suppression_cache.items():
        if c_data["count"] > 1:
            summary_card = format_bot_card(
                header="📊 RINGKASAN AKUMULASI SERANGAN",
                body={
                    "Pola Serangan": f"Rule {c_data['rule_id']} pada agen '{c_data['agent']}'",
                    "Attacker IP": c_data["ip"],
                    "Total Log Diredam": f"{c_data['count']:,} baris log",
                    "Status": "Serangan mereda / Jendela supresi ditutup",
                    "Hemat Notifikasi": f"{c_data['count'] - 1} pesan spam dicegah ✅",
                },
                border_char="═",
                color_icon="📊",
            )
            print(summary_card)

    print("=" * 68)
    print("SIMULASI SELESAI: Arsitektur automasi terbukti real-time & bebas fatigue!")
    print("=" * 68)


if __name__ == "__main__":
    run_simulation()