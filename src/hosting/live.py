"""The live feed (decision 0013): the streaming job publishes each micro-batch online, so the
hosted dashboard shows a real run within seconds.

A hosted dashboard cannot read a laptop's data/stream/ folder. So with --publish-live the streaming
job uploads the micro-batches it has finished, every few seconds, to a public Hugging Face dataset
(common.settings.LIVE_REPO), in the same batch_id=N folders it writes locally, and without the
columns that would give back prices (NOT_PUBLISHED). The dashboard copies that folder into
data/live/ and reads it exactly like a local run.

It is opt-in (--publish-live), with safeguards (publish_check, takes_over):
  - it only starts when this computer is signed in to Hugging Face with write access to the
    team's organisation and the job writes to a local folder (not HDFS on the cluster); if not,
    the job stops at once and says why;
  - every run uploads into its own folder, and only becomes the run the dashboard shows when
    no other run is streaming, so a teammate's run cannot take over a demo;
  - each run says who ran it.

    hf auth login                                       (once, with a write token)
    scripts/demo_live.sh                                (replay + job --publish-live, Java 21 set)

or, with no computer at all, the cloud streamer: the "Live stream" workflow of the team's
live-stream repository on github.com runs the same script on GitHub's servers (decision 0015).

A job restarted without --fresh carries on the same run: the run's identity is kept in the job's
output folder (STATE), and batches finished but not yet uploaded when it stopped go up first.
`python src/hosting/live.py finish OUT` marks a run finished when its job was killed before it
could (the cloud streamer's clean-up step).

The repository holds:
    current.json        {"run_id", "by", "started_utc", "updated_utc", "batches", "reached", "finished",
                        "complete"}: the run the dashboard shows; "batches" counts micro-batches with
                        data, "reached" is the last trading day published, "finished" is true once the
                        job stops, "complete" once it stopped because the replay ran out
    runs/<run_id>/<signals|event_car|predictions>/batch_id=N/part-0.parquet and _SUCCESS

--publish-live needs --out to be a local folder (not HDFS), and a Hugging Face login.
"""
from __future__ import annotations

import json
import os
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
STATE = "_live_run.json"     # in the job's output folder: the run it publishes, so a restart carries it on
RUNNER_ENV = "CALENDAR_EFFECTS_RUNNER"   # who started the run, when git does not know (the cloud streamer)


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
    """Who is streaming, for the dashboard: $CALENDAR_EFFECTS_RUNNER, or the first name in this
    computer's git config."""
    if os.environ.get(RUNNER_ENV):
        return os.environ[RUNNER_ENV]
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


def finished_batches(out_dir: Path) -> list[int]:
    """The micro-batches in out_dir that Spark has finished writing (any output has _SUCCESS)."""
    return sorted({int(p.parent.name.split("=")[1]) for sub in STREAM_SUBDIRS.values()
                   for p in (Path(out_dir) / sub).glob("batch_id=*/_SUCCESS")})


def load_state(out_dir: Path) -> dict | None:
    path = Path(out_dir) / STATE
    return json.loads(path.read_text()) if path.exists() else None


class LivePublisher:
    """Uploads finished micro-batches every `every` seconds, in one commit each time, so a
    fast replay does not hit Hugging Face's limits. Upload errors are printed, not raised:
    the streaming job keeps running, and the batches are tried again next time.

    resume=True carries on the run saved in out_dir (a job restarted without --fresh), uploading
    first what it had finished but not published; otherwise it starts a new run."""

    def __init__(self, out_dir, repo_id: str = LIVE_REPO, every: float = 15.0, resume: bool = False,
                 background: bool = True):
        from huggingface_hub import HfApi

        self.api, self.repo_id, self.out_dir, self.every = HfApi(), repo_id, Path(out_dir), every
        self.api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        saved = load_state(self.out_dir) if resume else None
        if saved:
            self.run_id, self.started, self.by = saved["run_id"], saved["started_utc"], saved["by"]
            self.published, self.last_batch, self.reached = saved["published"], saved["last_batch"], saved["reached"]
            self.complete = saved.get("complete", False)
        else:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.by = runner_name()
            self.published, self.last_batch, self.reached, self.complete = 0, -1, None, False
        self.waiting_for = None             # another run that is streaming, which this one waits behind
        self.pending = [b for b in finished_batches(self.out_dir) if b > self.last_batch] if saved else []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        if background:
            self.push(force=True)                # current.json first, so the dashboard switches to this run
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
            print(f"live feed: {'carrying on' if saved else 'new'} run {self.run_id} by {self.by} "
                  f"({self.published} micro-batches so far) -> https://huggingface.co/datasets/{repo_id}")

    def save(self, finished: bool):
        (self.out_dir / STATE).write_text(json.dumps({
            "run_id": self.run_id, "by": self.by, "started_utc": self.started, "published": self.published,
            "last_batch": self.last_batch, "reached": self.reached, "finished": finished,
            "complete": self.complete}))

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
            staged = staging / RUNS / self.run_id / STREAM_SUBDIRS["stream_signals"]
            dates = pd.read_parquet(staged, columns=["date"])["date"].dropna() if staged.exists() else []
            reached = max([str(pd.Timestamp(d).date()) for d in dates] + [self.reached or ""]) or None
            remote = self.current()
            if takes_over(remote, self.run_id):
                if self.waiting_for:
                    print(f"live feed: run {self.waiting_for} has ended; the dashboard now shows this run")
                self.waiting_for = None
                (staging / CURRENT).write_text(json.dumps({
                    "run_id": self.run_id, "by": self.by, "started_utc": self.started,
                    "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "batches": self.published + with_data, "reached": reached, "finished": finished,
                    "complete": self.complete and finished}))
            elif self.waiting_for != remote["run_id"]:
                self.waiting_for = remote["run_id"]
                print(f"live feed: run {remote['run_id']} by {remote.get('by', 'a teammate')} is streaming now; "
                      "this run uploads too, and the dashboard shows it once that run ends")
            self.api.upload_folder(repo_id=self.repo_id, repo_type="dataset", folder_path=staging,
                                   commit_message=f"live run {self.run_id}: {with_data} micro-batches"
                                                  + (" (finished)" if finished else ""))
            self.published += with_data
            self.last_batch, self.reached = max(batches + [self.last_batch]), reached
            self.save(finished)
            if with_data:
                print(f"live feed: published {with_data} micro-batches ({self.published} in this run)")
        except Exception as e:                   # keep streaming; try these batches again next time
            print(f"live feed: upload failed ({e}); will retry")
            with self.lock:
                self.pending = batches + self.pending
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def close(self, complete: bool = False):
        """Publish what is left and mark the run finished: complete when the replay ran out,
        otherwise stopped (it can carry on later)."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.complete = complete
        self.push(force=True, finished=True)


def finish(out_dir: Path, repo_id: str = LIVE_REPO) -> str:
    """Mark the run in out_dir finished, after uploading what it had left, when its job was killed
    before it could do so itself (the cloud streamer's clean-up step). Does nothing otherwise."""
    saved = load_state(out_dir)
    if saved is None or saved.get("finished"):
        return "live feed: nothing to finish"
    LivePublisher(out_dir, repo_id, resume=True, background=False).close()
    return f"live feed: run {saved['run_id']} marked finished"


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="the live feed's command-line helpers")
    parser.add_argument("command", choices=["finish"])
    parser.add_argument("out", help="the streaming job's output folder")
    print(finish(Path(parser.parse_args().out)))
