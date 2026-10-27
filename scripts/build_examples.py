"""Run the real pipeline once per example chip (name only, against its sample seller) and save the
full result to examples/<seller>/<slug>.json, which the page plays back when a chip is clicked.

    python scripts/build_examples.py                    # every chip, both sellers
    python scripts/build_examples.py vanta/hebbia       # just these
    python scripts/build_examples.py vanta              # every chip for one seller

Each run uses real Tavily searches and Gemini calls. Builds count against their own daily
counters (examples_*_counter.json, git-ignored), not the app's, so rebuilding never uses up
the live site's budget. One process builds them all, so a company shown for both sellers is
searched once. A busy or capped run is never saved. Read the printed email before committing;
re-run a chip until its draft reads cleanly.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as qf  # noqa: E402  (needs the path above)

qf.TAVILY_COUNTER_FILE = "examples_tavily_counter.json"
qf.USAGE_COUNTER_FILE = "examples_usage_counter.json"
qf.TAVILY_DAILY_CAP = int(os.environ.get("EXAMPLES_TAVILY_CAP", "80"))  # Tavily searches per day for builds

# Per-visitor and cache fields describe one live request, not the saved example.
DROP_KEYS = ("_transient", "checks_left_today", "cached")


def build(seller, slug, name):
    qf.start_budget()
    payload, code = qf.run_analysis(name, None, qf.seller_profile(seller), "")
    if code != 200 or payload.get("_transient") or payload.get("reason") in ("busy", "daily_cap"):
        print(f"{seller}/{slug}: {payload.get('error') or payload.get('message')}; not saved")
        return None
    payload = qf.strip_dashes({k: v for k, v in payload.items() if k not in DROP_KEYS})
    path = qf.example_path(seller, slug)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    draft = payload.get("draft") or {}
    print(f"\n=== {seller}/{slug}: {payload.get('verdict')} {payload.get('fit_score') or ''} "
          f"({payload.get('reason') or 'verdict'})")
    if draft:
        print(f"Subject: {draft['subject']}\n{draft['body']}")
    elif payload.get("draft_note"):
        print("No draft:", payload["draft_note"])
    for c in payload.get("candidates") or []:
        print("  candidate:", c)
    return payload


def targets(args):
    """(seller, slug, name) for "seller" or "seller/slug" arguments; every chip when none."""
    chips = [(seller, slug, name) for seller, items in qf.EXAMPLES.items() for slug, name in items]
    if not args:
        return chips
    wanted = [a.strip("/").split("/", 1) for a in args]
    unknown = [a for a, w in zip(args, wanted)
               if not any(w[0] == s and (len(w) == 1 or w[1] == slug) for s, slug, _ in chips)]
    if unknown:
        sys.exit(f"Unknown example(s): {', '.join(unknown)}")
    return [c for c in chips if any(w[0] == c[0] and (len(w) == 1 or w[1] == c[1]) for w in wanted)]


if __name__ == "__main__":
    results = [build(*t) for t in targets(sys.argv[1:])]
    sys.exit(0 if all(results) else 1)
