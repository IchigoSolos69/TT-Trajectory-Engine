import os
import sys
import json
import urllib.request
import zipfile

# OpenTTGames dataset repository / releases / mirrors
MIRRORS = [
    "https://raw.githubusercontent.com/OpenTTGames/OpenTTGames/main/dataset",
    "https://github.com/OpenTTGames/OpenTTGames/releases/download/v1.0",
    "https://lab.osai.ai/datasets/openttgames"
]

TEST_SEQUENCES = [f"test_{i}" for i in range(1, 8)]
GAME_SEQUENCES = [f"game_{i}" for i in range(1, 6)]

def download_and_extract_markup():
    dataset_root = "python_tools/dataset/raw_markup"
    os.makedirs(dataset_root, exist_ok=True)

    print("=========================================================================")
    print("  OPENTTGAMES MARKUP DOWNLOADER (test_1 .. test_7, game_1 .. game_5)")
    print("=========================================================================")

    all_sequences = GAME_SEQUENCES + TEST_SEQUENCES

    for seq in all_sequences:
        seq_dir = os.path.join(dataset_root, seq)
        os.makedirs(seq_dir, exist_ok=True)

        ball_path = os.path.join(seq_dir, "ball_markup.json")
        events_path = os.path.join(seq_dir, "events_markup.json")

        if os.path.exists(ball_path) and os.path.exists(events_path):
            print(f"[{seq}] Already downloaded and ready.")
            continue

        print(f"[{seq}] Attempting download from mirrors...")
        success = False

        for mirror in MIRRORS:
            ball_url = f"{mirror}/{seq}/ball_markup.json"
            events_url = f"{mirror}/{seq}/events_markup.json"
            zip_url = f"{mirror}/{seq}_markup.zip"

            try:
                # Try downloading JSON files directly
                req1 = urllib.request.Request(ball_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req1, timeout=5) as resp1:
                    with open(ball_path, 'wb') as f1:
                        f1.write(resp1.read())

                req2 = urllib.request.Request(events_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req2, timeout=5) as resp2:
                    with open(events_path, 'wb') as f2:
                        f2.write(resp2.read())

                print(f"[{seq}] Successfully downloaded directly from {mirror}")
                success = True
                break
            except Exception:
                pass

            try:
                # Try downloading zip
                zip_dest = os.path.join(seq_dir, f"{seq}_markup.zip")
                req_z = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req_z, timeout=5) as resp_z:
                    with open(zip_dest, 'wb') as f_z:
                        f_z.write(resp_z.read())

                with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
                    zip_ref.extractall(seq_dir)
                os.remove(zip_dest)

                print(f"[{seq}] Successfully downloaded zip from {mirror}")
                success = True
                break
            except Exception:
                pass

        if not success:
            print(f"[{seq}] Direct HTTP download not reachable on standard mirrors. Torrent or OSai login required.")

if __name__ == "__main__":
    download_and_extract_markup()
