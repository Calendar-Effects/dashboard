# Calendar Effects dashboard (hosted copy)

This repository is generated: `scripts/sync_dashboard_app.sh` in the main Calendar Effects repository
copies the dashboard and the code it needs here after every commit, and Streamlit Community Cloud
redeploys it. Do not edit it here.

- App: `src/dashboard/app.py` (the same file teammates run with `streamlit run src/dashboard/app.py`)
- Recorded stream and event history: `hosting/snapshot/` (derived statistics only, no prices)
- Live runs: the streaming job publishes them with `--publish-live` (`scripts/demo_live.sh`, or the cloud streamer on GitHub Actions); the app reads them from Hugging Face
- Docs: https://calendar-effects.mintlify.app
