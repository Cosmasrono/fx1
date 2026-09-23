"""Train the local win-probability model from the command line.

Does the same work as POST /api/model/train, but prints progress. The first
run replays the whole cached history and can take around 15 minutes; later
runs reuse the cached setups and only replay new candles.

Usage:
    python research.py --download --years 5   # once, if backend/data is empty
    python train_model.py
"""
import asyncio
import argparse
import sys

from app import prediction
from app.market import fetch_candles


def main() -> int:
    parser = argparse.ArgumentParser(description="Train from price history and completed paper trades.")
    parser.add_argument("--offline", action="store_true", help="Use cached history and the journal without API calls")
    args = parser.parse_args()
    recent = None
    if not args.offline:
        try:
            recent, _ = asyncio.run(fetch_candles(outputsize=5000))
        except Exception as exc:
            print(f"latest candles unavailable ({type(exc).__name__}: {exc}); "
                  "training on the cached history only")

    last = {"stage": None}

    def progress(stage, done=0, total=0):
        if total:
            print(f"\r{stage}: {done:,}/{total:,} candles", end="", flush=True)
        elif stage != last["stage"]:
            print(f"\n{stage}", flush=True)
        last["stage"] = stage

    try:
        report = prediction.train(recent, progress)
    except ValueError as exc:
        print(f"\n{exc}")
        return 1
    evidence = report["out_of_sample"]
    base, kept = evidence["baseline"], evidence["filtered"]
    print(f"\n\ntrained on {report['samples']:,} setups, {report['data_start'][:10]} -> {report['data_end'][:10]}")
    paper = report["paper_learning"]
    print(f"paper feedback: {paper['used']} completed trades ({paper['wins']} targets, {paper['losses']} stops); excluded {sum(paper['skipped'].values())}")
    print(f"out of sample ({base['trades']:,} {report['population']} in {len(evidence['folds'])} later periods):")
    print(f"  ranking skill (AUC)   : {evidence['auc']}  (0.5 = coin flip)")
    if base["trades"]:
        print(f"  all setups            : win {base['win_rate']:.1%}  avg {base['avg_r']:+.3f}R")
    if kept["trades"]:
        print(f"  setups model allows   : win {kept['win_rate']:.1%}  avg {kept['avg_r']:+.3f}R")
    print(f"  losing setups skipped : {evidence['losses_avoided']:,}  winners given up: {evidence['wins_given_up']:,}")
    if report["filter_helps"]:
        print(f"threshold {report['threshold']:.1%}: setups below it become HOLD")
    else:
        print("the filter did not help out of sample, so it will never block an entry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
