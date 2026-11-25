"""The live feed (decision 0013): the streaming job publishes each micro-batch online, so the
hosted dashboard shows a real run within seconds.

A hosted dashboard cannot read a laptop's data/stream/ folder. So for a demo, start the job with
--publish-live: every few seconds it uploads the micro-batches it has finished to a public Hugging
Face dataset (common.settings.LIVE_REPO), in the same batch_id=N folders it writes locally, and
without the columns that would give back prices (NOT_PUBLISHED). The dashboard copies that folder
into data/live/ and reads it exactly like a local run.

    hf auth login                                       (once, with a write token)
    python3 src/producers/replay_to_files.py
    python3 src/streaming/job.py --source files --fresh --publish-live

The repository holds:
    current.json                                        {"run_id", "started_utc", "updated_utc", "batches"}
    runs/<run_id>/<signals|event_car|predictions>/batch_id=N/part-0.parquet and _SUCCESS

--publish-live needs --out to be a local folder (not HDFS), and a Hugging Face login.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.settings import LIVE_REPO, STREAM_SUBDIRS

# Columns that are, or would give back, Yahoo's prices, volumes or raw returns: never published.
NOT_PUBLISHED = {"ret", "mkt_ret", "vol_ratio", "timestamp_utc"}
CURRENT = "current.json"
RUNS = "runs"


def public(df: pd.DataFrame) -> pd.DataFrame:
    """The table without any column that could give back prices, volumes or raw returns."""
    return df.drop(columns=[c for c in df.columns if c in NOT_PUBLISHED])


def stage_batch(out_dir: Path, batch_id: int, staging: Path, run_id: str) -> int:
    """Copy one finished micro-batch, minus NOT_PUBLISHED, into staging/runs/<run_id>/.
    Returns how many of the outputs (signals, event CARs, predictions) it had."""
    found = 0
    for sub in STREAM_SUBDIRS.values():
        src = Path(out_dir) / sub / f"batch_id={batch_id}"
        if not (src / "_SUCCESS").exists():
            continue
        dest = staging / RUNS / run_id / sub / f"batch_id={batch_id}"
        dest.mkdir(parents=True, exist_ok=True)
        public(pd.read_parquet(src)).to_parquet(dest / "part-0.parquet", index=False)
        (dest / "_SUCCESS").touch()
        found += 1
    return found


class LivePublisher:
    """Uploads finished micro-batches every `every` seconds, in one commit each time, so a
    fast replay does not hit Hugging Face's limits. Upload errors are printed, not raised:
    the streaming job keeps running, and the batches are tried again next time."""

    def __init__(self, out_dir, repo_id: str = LIVE_REPO, every: float = 15.0):
        from huggingface_hub import HfApi

        self.api, self.repo_id, self.out_dir, self.every = HfApi(), repo_id, Path(out_dir), every
        self.api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.pending: list[int] = []
        self.published = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.push(force=True)                    # current.json first, so the dashboard switches to this run
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        print(f"live feed: run {self.run_id} -> https://huggingface.co/datasets/{repo_id}")

    def add(self, batch_id: int):
        with self.lock:
            self.pending.append(batch_id)

    def loop(self):
        while not self.stop.wait(self.every):
            self.push()

    def push(self, force: bool = False):
        with self.lock:
            batches, self.pending = self.pending, []
        if not batches and not force:
            return
        staging = Path(tempfile.mkdtemp(prefix="calendar-effects-live-"))
        try:
            for b in batches:
                stage_batch(self.out_dir, b, staging, self.run_id)
            (staging / CURRENT).write_text(json.dumps({
                "run_id": self.run_id, "started_utc": self.started,
                "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "batches": self.published + len(batches)}))
            self.api.upload_folder(repo_id=self.repo_id, repo_type="dataset", folder_path=staging,
                                   commit_message=f"live run {self.run_id}: {len(batches)} micro-batches")
            self.published += len(batches)
            if batches:
                print(f"live feed: published {len(batches)} micro-batches ({self.published} in this run)")
        except Exception as e:                   # keep streaming; try these batches again next time
            print(f"live feed: upload failed ({e}); will retry")
            with self.lock:
                self.pending = batches + self.pending
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.push(force=True)


def mirror(cache_dir: Path, repo_id: str = LIVE_REPO) -> dict | None:
    """Copy the current live run into cache_dir/runs/<run_id>/, downloading only new files.

    Returns current.json plus "folder" (the local copy, laid out like data/stream/), or None
    when no run has been published yet. Each batch's _SUCCESS is fetched after its data, so a
    reader never sees half a batch.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    api = HfApi()
    try:
        current = json.loads(Path(hf_hub_download(repo_id, CURRENT, repo_type="dataset",
                                                  force_download=True)).read_text())
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    run = f"{RUNS}/{current['run_id']}"
    files = [f.path for f in api.list_repo_tree(repo_id, path_in_repo=run, recursive=True, repo_type="dataset")
             if getattr(f, "size", None) is not None]
    for path in sorted(files, key=lambda p: (p.rsplit("/", 1)[0], p.endswith("_SUCCESS"))):
        if not (Path(cache_dir) / path).exists():
            hf_hub_download(repo_id, path, repo_type="dataset", local_dir=cache_dir)
    return {**current, "folder": Path(cache_dir) / run}


def age_seconds(current: dict) -> float:
    """Seconds since the live run last published."""
    return time.time() - datetime.fromisoformat(current["updated_utc"]).timestamp()
