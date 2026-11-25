"""The live feed (decision 0013): the streaming job publishes each micro-batch online, so the
hosted dashboard shows a real run within seconds.

A hosted dashboard cannot read a laptop's data/stream/ folder. So the streaming job uploads the
micro-batches it has finished, every few seconds, to a public Hugging Face dataset
(common.settings.LIVE_REPO), in the same batch_id=N folders it writes locally, and without the
columns that would give back prices (NOT_PUBLISHED). The dashboard copies that folder into
data/live/ and reads it exactly like a local run.

It is automatic, with safeguards (publish_check, takes_over):
  - it only starts when this computer is signed in to Hugging Face with write access to the
    team's organisation, and the job writes to a local folder (not HDFS on the cluster);
    otherwise the job prints one line and runs without it, never failing;
  - every run uploads into its own folder, and only becomes the run the dashboard shows when
    no other run is streaming, so a teammate's test cannot take over a demo;
  - each run says who ran it.
--no-publish-live keeps a run private; --publish-live insists (and fails if it cannot publish).

    hf auth login                                       (once, with a write token)
    python3 src/producers/replay_to_files.py
    python3 src/streaming/job.py --source files --fresh

The repository holds:
    current.json        {"run_id", "by", "started_utc", "updated_utc", "batches", "finished"}: the run the
                        dashboard shows; "batches" counts micro-batches with data, "finished" is true
                        once the job stops
    runs/<run_id>/<signals|event_car|predictions>/batch_id=N/part-0.parquet and _SUCCESS

--publish-live needs --out to be a local folder (not HDFS), and a Hugging Face login.
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
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


def publish_check(repo_id: str = LIVE_REPO) -> tuple[bool, str]:
    """Can this computer publish to the live feed? (True, who) or (False, why not)."""
    try:
        from huggingface_hub import HfApi
        me = HfApi().whoami()
    except Exception:
        return False, "this computer is not signed in to Hugging Face (`hf auth login`)"
    if me.get("auth", {}).get("accessToken", {}).get("role") == "read":
        return False, "the Hugging Face token on this computer is read-only"
    owner = repo_id.split("/")[0]
    member = any(o.get("name") == owner and o.get("roleInOrg") in ("admin", "write", "contributor")
                 for o in me.get("orgs", []))
    if owner != me.get("name") and not member:
        return False, f"Hugging Face account {me.get('name')} cannot write to {owner}"
    return True, me.get("name", "")


def takes_over(remote: dict | None, run_id: str) -> bool:
    """May this run become the one the dashboard shows? Only if it already is, or no other run
    is streaming now (so a test run never takes over a demo)."""
    return remote is None or remote.get("run_id") == run_id or run_state(remote) != "live"


def runner_name() -> str:
    """The first name in this computer's git config, so the dashboard can say who is streaming."""
    import subprocess
    name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True).stdout.strip()  # not-config: the git subcommand
    return name.split()[0] if name else "a teammate"


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
        self.by = runner_name()
        self.waiting_for = None             # another run that is streaming, which this one waits behind
        self.pending: list[int] = []
        self.published = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.push(force=True)                    # current.json first, so the dashboard switches to this run
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        print(f"live feed: run {self.run_id} by {self.by} -> https://huggingface.co/datasets/{repo_id}")

    def current(self) -> dict | None:
        """The run the dashboard shows now (current.json), or None if there is none."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
        try:
            return json.loads(Path(hf_hub_download(self.repo_id, CURRENT, repo_type="dataset",
                                                   force_download=True)).read_text())
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None

    def add(self, batch_id: int):
        with self.lock:
            self.pending.append(batch_id)

    def loop(self):
        while not self.stop.wait(self.every):
            self.push()

    def push(self, force: bool = False, finished: bool = False):
        with self.lock:
            batches, self.pending = self.pending, []
        if not batches and not force:
            return
        staging = Path(tempfile.mkdtemp(prefix="calendar-effects-live-"))
        try:
            with_data = sum(stage_batch(self.out_dir, b, staging, self.run_id) > 0 for b in batches)
            remote = self.current()
            if takes_over(remote, self.run_id):
                if self.waiting_for:
                    print(f"live feed: run {self.waiting_for} has ended; the dashboard now shows this run")
                self.waiting_for = None
                (staging / CURRENT).write_text(json.dumps({
                    "run_id": self.run_id, "by": self.by, "started_utc": self.started,
                    "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "batches": self.published + with_data, "finished": finished}))
            elif self.waiting_for != remote["run_id"]:
                self.waiting_for = remote["run_id"]
                print(f"live feed: run {remote['run_id']} by {remote.get('by', 'a teammate')} is streaming now; "
                      "this run uploads too, and the dashboard shows it once that run ends")
            self.api.upload_folder(repo_id=self.repo_id, repo_type="dataset", folder_path=staging,
                                   commit_message=f"live run {self.run_id}: {with_data} micro-batches"
                                                  + (" (finished)" if finished else ""))
            self.published += with_data
            if with_data:
                print(f"live feed: published {with_data} micro-batches ({self.published} in this run)")
        except Exception as e:                   # keep streaming; try these batches again next time
            print(f"live feed: upload failed ({e}); will retry")
            with self.lock:
                self.pending = batches + self.pending
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def close(self):
        """Publish what is left and mark the run finished (the job stopped normally)."""
        self.stop.set()
        self.thread.join()
        self.push(force=True, finished=True)


def mirror(cache_dir: Path, repo_id: str = LIVE_REPO) -> dict | None:
    """Copy the current live run into cache_dir/runs/<run_id>/, downloading only new files.

    Returns current.json plus "folder" (the local copy, laid out like data/stream/), or None
    when no run has been published yet. New files are fetched in parallel, and every _SUCCESS
    marker only after all the data, so a reader never sees half a batch.
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
             if getattr(f, "size", None) is not None and not (Path(cache_dir) / f.path).exists()]

    def fetch(path):
        hf_hub_download(repo_id, path, repo_type="dataset", local_dir=cache_dir)

    # Many small files: fetch them in parallel, all the data first and the _SUCCESS markers after.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for done_marker in (False, True):
            list(pool.map(fetch, [f for f in files if f.endswith("_SUCCESS") == done_marker]))
    return {**current, "folder": Path(cache_dir) / run}


def age_seconds(current: dict) -> float:
    """Seconds since the live run last published."""
    return time.time() - datetime.fromisoformat(current["updated_utc"]).timestamp()


# A running job publishes every --publish-every seconds (15 by default), so a run that has
# been quiet for this long without saying it finished has most likely been stopped.
QUIET_SECONDS = 120


def run_state(current: dict | None) -> str:
    """'live' (streaming now), 'finished' (stopped normally), 'stopped' (went quiet without
    finishing: stopped with Ctrl-C, crashed, or the laptop went offline), or 'none'."""
    if not current or "run_id" not in current:
        return "none"
    if current.get("finished"):
        return "finished"
    return "live" if age_seconds(current) < QUIET_SECONDS else "stopped"
