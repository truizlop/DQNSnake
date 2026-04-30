"""Stream JSON scalar metrics from stdin into TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    writer = EventFileWriter(args.logdir)

    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                step = int(payload["step"])
                scalars = payload.get("scalars", {})
                now = time.time()
                for tag, value in scalars.items():
                    summary = summary_pb2.Summary(
                        value=[summary_pb2.Summary.Value(tag=str(tag), simple_value=float(value))]
                    )
                    event = event_pb2.Event(wall_time=now, step=step, summary=summary)
                    writer.add_event(event)
                writer.flush()
            except Exception as exc:  # pragma: no cover - defensive runtime parsing
                sys.stderr.write(f"tensorboard_stream parse error: {exc}\n")
                sys.stderr.flush()
    finally:
        writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
