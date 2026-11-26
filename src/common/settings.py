"""Where the project's files live, and how to reach Kafka.

Scripts take every path from here, so they work from any folder and a file name is
typed once. What is studied lives in config/ (events, retailers, Wikipedia articles,
retail series) and how it is studied (years, windows, market index) in common.study.

Usage:
    from common.settings import KAFKA_BOOTSTRAP, RAW_DIR, clean_path

    prices = pd.read_parquet(clean_path("prices"))
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Hand-edited registries, committed. Their columns are contracts (docs/data-contracts.md).
CONFIG_DIR = REPO_ROOT / "config"
EVENT_TYPES_CSV = CONFIG_DIR / "event_types.csv"
EVENT_DATES_CSV = CONFIG_DIR / "event_dates.csv"
RETAILERS_CSV = CONFIG_DIR / "retailers.csv"
WIKI_ARTICLES_CSV = CONFIG_DIR / "wiki_articles.csv"
RETAIL_SERIES_CSV = CONFIG_DIR / "retail_series.csv"

RAW_DIR = REPO_ROOT / "raw"                  # downloads; gitignored, never edited by hand
CLEAN_DIR = REPO_ROOT / "clean"              # cleaned tables; small, committed for the team
DATA_DIR = REPO_ROOT / "data"                # made by runs; gitignored
REPLAY_DIR = DATA_DIR / "replay"             # JSON files for running the streaming job without Kafka
SRC_DIR = REPO_ROOT / "src"                  # the project's Python packages
DEMO_STREAM_DIR = DATA_DIR / "demo" / "stream"   # where scripts/demo_live.sh writes its run
STREAM_DIR = DATA_DIR / "stream"             # streaming job output, one subfolder per dataset:
STREAM_SUBDIRS = {"stream_signals": "signals", "stream_event_car": "event_car",
                  "stream_predictions": "predictions"}
RESULTS_DIR = DATA_DIR / "results"           # batch event study and model output
MODEL_DIR = DATA_DIR / "models"              # trained models (src/ml/train.py); remade by a run
BIG_MOVE_MODEL = MODEL_DIR / "big_move"      # the Spark MLlib PipelineModel the streaming job loads
BIG_MOVE_MODEL_INFO = MODEL_DIR / "big_move.json"  # its features, alert cut-off and test scores
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"   # Spark streaming checkpoints, one folder per query; gitignored

# Documentation. docs/ is written by hand; docs-site/ is the Mintlify site that
# scripts/build_docs.py builds from the code and docs/ (decision 0011).
DOCS_DIR = REPO_ROOT / "docs"
DOCS_SITE_DIR = REPO_ROOT / "docs-site"

# What is published (src/hosting/, decisions 0010 and 0013).
HOSTING_DIR = REPO_ROOT / "hosting"
SNAPSHOT_DIR = HOSTING_DIR / "snapshot"      # the recorded replay and event history the dashboard shows;
                                             # made by src/hosting/snapshot.py, committed (derived data only)
SITE_DIR = HOSTING_DIR / "space"             # the Hugging Face Space: a page that embeds the hosted dashboard
CARDS_DIR = HOSTING_DIR / "cards"            # dataset, model and Space card templates
APP_BUNDLE_DIR = HOSTING_DIR / "streamlit"   # what the hosted dashboard's repository adds to the code
PUBLISH_DIR = DATA_DIR / "publish"           # the dataset, model and Space folders, staged before upload

# The team's public homes: the Hugging Face organisation (dataset, model, Space, live feed) and the
# github.com organisation (the docs site's and the hosted dashboard's repositories).
HF_OWNER = "calendar-effects"
HF_REPOS = {"dataset": "calendar-effects", "model": "big-move-model", "space": "dashboard", "live": "live-feed"}
GITHUB_ORG = "Calendar-Effects"
DOCS_URL = "https://calendar-effects.mintlify.app"

# The hosted dashboard (Streamlit Community Cloud) and its live feed: during a demo the streaming
# job publishes each micro-batch to a public Hugging Face dataset (src/hosting/live.py).
DASHBOARD_URL = os.environ.get("CALENDAR_EFFECTS_DASHBOARD_URL", "https://calendar-effects.streamlit.app")
LIVE_REPO = os.environ.get("CALENDAR_EFFECTS_LIVE_REPO", f"{HF_OWNER}/{HF_REPOS['live']}")
LIVE_CACHE_DIR = DATA_DIR / "live"           # the dashboard's local copy of the live feed
# The cloud streamer (decision 0015): a private github.com repository whose "Live stream" workflow
# runs scripts/demo_live.sh on GitHub's servers, so a live run needs no team member's computer.
STREAMER_REPO = f"{GITHUB_ORG}/live-stream"
STREAMER_URL = f"https://github.com/{STREAMER_REPO}/actions/workflows/live-stream.yml"
LINKS = {                                    # where everything is, for pages and cards
    "docs": DOCS_URL,
    "dashboard": DASHBOARD_URL,
    "dataset": f"https://huggingface.co/datasets/{HF_OWNER}/{HF_REPOS['dataset']}",
    "model": f"https://huggingface.co/{HF_OWNER}/{HF_REPOS['model']}",
    "space": f"https://huggingface.co/spaces/{HF_OWNER}/{HF_REPOS['space']}",
    "huggingface": f"https://huggingface.co/{HF_OWNER}",
    "github": f"https://github.com/{GITHUB_ORG}",
    "streamer": STREAMER_URL,
}


def clean_path(dataset: str) -> Path:
    """The committed Parquet file of a clean dataset, e.g. clean_path("prices")."""
    return CLEAN_DIR / f"{dataset}.parquet"


# The course cluster's Kafka servers. Set KAFKA_BOOTSTRAP=localhost:9092 for a local Kafka.
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "node1.local:9092,node2.local:9092")
