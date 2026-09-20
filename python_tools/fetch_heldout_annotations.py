import os
import sys
import json
import urllib.request

OPENTTGAMES_GITHUB_BASE = "https://raw.githubusercontent.com/OpenTTGames/OpenTTGames/main/dataset"

GAMES_LIST = [f"game_{i}" for i in range(1, 6)] + [f"test_{i}" for i in range(1, 6)]

def check_local_annotations(dataset_root):
    local_status = {}
    for gname in GAMES_LIST:
        gdir = os.path.join(dataset_root, gname)
        ball_json = os.path.join(gdir, "ball_markup.json")
        events_json = os.path.join(gdir, "events_markup.json")

        has_ball = os.path.isfile(ball_json) and os.path.getsize(ball_json) > 0
        has_events = os.path.isfile(events_json) and os.path.getsize(events_json) > 0

        local_status[gname] = {
            "dir": gdir,
            "has_ball": has_ball,
            "has_events": has_events,
            "complete": has_ball and has_events
        }
    return local_status

def fetch_markup_file(url, save_path):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        return False

def main():
    dataset_root = "python_tools/dataset/raw_markup"
    os.makedirs(dataset_root, exist_ok=True)

    print("=========================================================================")
    print("  OPENTTGAMES HELD-OUT DATASET ANNOTATION FETCH & AUDIT")
    print("=========================================================================")

    status = check_local_annotations(dataset_root)

    print(f"{'Sequence':<12} | {'Ball Markup JSON':<20} | {'Events Markup JSON':<20} | {'Status':<12}")
    print("-" * 75)

    for gname, info in status.items():
        ball_str = "EXISTS" if info["has_ball"] else "MISSING"
        events_str = "EXISTS" if info["has_events"] else "MISSING"
        stat_str = "READY" if info["complete"] else "PENDING"
        print(f"{gname:<12} | {ball_str:<20} | {events_str:<20} | {stat_str:<12}")

    missing_games = [g for g, info in status.items() if not info["complete"]]

    if missing_games:
        print(f"\nAttempting to download markup-only files for missing sequences: {missing_games}...")
        for gname in missing_games:
            gdir = status[gname]["dir"]
            os.makedirs(gdir, exist_ok=True)

            ball_url = f"{OPENTTGAMES_GITHUB_BASE}/{gname}/ball_markup.json"
            events_url = f"{OPENTTGAMES_GITHUB_BASE}/{gname}/events_markup.json"

            ball_dest = os.path.join(gdir, "ball_markup.json")
            events_dest = os.path.join(gdir, "events_markup.json")

            if not status[gname]["has_ball"]:
                res_b = fetch_markup_file(ball_url, ball_dest)
                print(f"  Fetching {gname}/ball_markup.json -> {'SUCCESS' if res_b else 'FAILED (URL not reachable)'}")

            if not status[gname]["has_events"]:
                res_e = fetch_markup_file(events_url, events_dest)
                print(f"  Fetching {gname}/events_markup.json -> {'SUCCESS' if res_e else 'FAILED (URL not reachable)'}")

    # Re-check status
    updated_status = check_local_annotations(dataset_root)
    ready_count = sum(1 for info in updated_status.values() if info["complete"])
    print(f"\nAudit Summary: {ready_count} / {len(GAMES_LIST)} sequences fully annotated locally.")

if __name__ == "__main__":
    main()
