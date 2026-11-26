"""Calendar Effects dashboard (decisions 0008 and 0013): one Streamlit app, on a laptop and hosted.

On a laptop, next to the streaming job:    streamlit run src/dashboard/app.py
Hosted (free, Streamlit Community Cloud):   common.settings.DASHBOARD_URL, also embedded in the
                                            Hugging Face Space. scripts/sync_dashboard_app.sh deploys it.

Four views:
  Live stream     the streaming job's output as it is written: today's alerts, tomorrow's watchlist
                  from the big-move model, event CARs as windows close, alerts per day, one retailer.
                  Shows a published run (the online live feed a job writes with --publish-live,
                  src/hosting/live.py: the cloud streamer or a laptop) or, on a computer that can run the
                  job, that job's output folder. The hosted app shows published runs only.
  Replay          the same view over the recorded stream, any range of years (by default the big-move
                  model's test years, which it never saw in training), played day by day from
                  hosting/snapshot/ (replay.json and replay_rows.parquet, src/hosting/snapshot.py).
  Event history   each event type's CAR path in every year, the average, a straight-line trend to the
                  next year with its prediction interval, and the model's flags on each event day.
  About           what the project is, and links to the docs, the dataset and the model.

Only micro-batch folders Spark has finished (they hold a _SUCCESS file) are read, so a half-written
batch never shows up, and each finished folder is read once and cached.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import reloader

reloader.refresh()   # a redeploy keeps old modules in memory; reload the project's own if they changed

from common.contracts import expand_events, load_retailers
from common.settings import (DASHBOARD_URL, DEMO_STREAM_DIR, LINKS, LIVE_CACHE_DIR, LIVE_REPO, REPO_ROOT,
                             SNAPSHOT_DIR, SRC_DIR, STREAM_DIR, STREAM_SUBDIRS)
from common.study import (ABNORMAL_RETURN_Z, ATTENTION_SPIKE_Z, ESTIMATION_DAYS, ML_TEST_START_YEAR,
                          STUDY_END_YEAR, STUDY_START_YEAR)

MIN_HISTORY = ESTIMATION_DAYS

# Chart colours: a validated colour-blind-safe set (blue, orange, aqua), plus greys for context.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8f8d86"
LINK_NAMES = {"docs": "Docs: a tour of the whole codebase", "dataset": "Dataset", "model": "Model",
              "space": "Hugging Face Space", "huggingface": "Hugging Face organisation", "github": "GitHub organisation"}
REPLAY = "Replay"
VIEWS = ["Live stream", REPLAY, "Event history & forecast", "About"]


# ---------------------------------------------------------------- reading the stream

@st.cache_data(show_spinner=False)
def read_batch(folder: str) -> pd.DataFrame:
    """One finished micro-batch folder (cached: a finished batch never changes)."""
    return pd.read_parquet(folder)


def read_stream(dataset: str, stream_dir: Path) -> pd.DataFrame:
    """Every finished micro-batch of one streaming output, in batch order."""
    folder = Path(stream_dir) / STREAM_SUBDIRS[dataset]
    done = sorted((p for p in folder.glob("batch_id=*") if (p / "_SUCCESS").exists()),
                  key=lambda p: int(p.name.split("=")[1]))
    if not done:
        return pd.DataFrame()
    df = pd.concat([read_batch(str(p)) for p in done], ignore_index=True)
    for col in ("date", "day0_date", "event_date"):
        if col in df:
            df[col] = pd.to_datetime(df[col])
    df.attrs["batches"] = len(done)
    return df


# Where the Live stream view can read a run from. The hosted copy holds only the dashboard's own
# code, so no streaming job can run next to it, and it offers published runs only.
PUBLISHED, LOCAL = "Published runs (online)", "A job on this computer"
CAN_STREAM_HERE = ((SRC_DIR / "streaming" / "job.py").exists()
                   and os.environ.get("CALENDAR_EFFECTS_HOSTED") != "1")    # the tests set it to act hosted


def local_runs() -> list[tuple[Path, float]]:
    """This computer's stream folders with finished batches (the demo's, then the full run's),
    newest first, each with the time its latest batch finished."""
    found = []
    for folder in (DEMO_STREAM_DIR, STREAM_DIR):
        marks = list((folder / STREAM_SUBDIRS["stream_signals"]).glob("batch_id=*/_SUCCESS"))
        if marks:
            found.append((folder, max(m.stat().st_mtime for m in marks)))
    return sorted(found, key=lambda run: -run[1])


def streaming_here() -> bool:
    """Is a job writing on this computer right now (a batch finished in the last two minutes)?"""
    from hosting.live import QUIET_SECONDS
    runs = local_runs() if CAN_STREAM_HERE else []
    return bool(runs) and time.time() - runs[0][1] < QUIET_SECONDS


def shown(path: Path) -> str:
    """A folder as people type it: relative to the repository when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@st.cache_data(ttl=4, show_spinner=False)
def live_feed() -> dict | None:
    """The online live feed, copied to data/live/ (shared by every viewer, refreshed every 4 s).
    CALENDAR_EFFECTS_OFFLINE=1 turns it off (the tests use it)."""
    if os.environ.get("CALENDAR_EFFECTS_OFFLINE"):
        return None
    from hosting.live import mirror
    try:
        return mirror(LIVE_CACHE_DIR)
    except Exception as e:                       # offline, or Hugging Face is busy
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def retailers() -> pd.DataFrame:
    return pd.DataFrame(load_retailers())[["ticker", "company", "retailer_type"]]


@st.cache_data(show_spinner=False)
def calendar() -> pd.DataFrame:
    events = pd.DataFrame(expand_events(STUDY_START_YEAR, STUDY_END_YEAR))
    events["event_date"] = pd.to_datetime(events["event_date"])
    return events[["event_type", "event_date"]].drop_duplicates().sort_values("event_date")


# The snapshot is cached, keyed by its file's modification time: the hosted app's process outlives
# a redeploy, so a cache keyed by name alone would keep serving the old recording. (Streamlit leaves
# arguments starting with "_" out of the key, so these names must not start with one.)
def snapshot(name: str) -> dict | None:
    path = SNAPSHOT_DIR / f"{name}.json"
    return read_json(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_data(show_spinner=False)
def read_json(path: str, modified: float) -> dict:
    return json.loads(Path(path).read_text())


def replay_frames() -> dict | None:
    """The recorded stream (hosting/snapshot/) as the tables the live view reads."""
    path = SNAPSHOT_DIR / "replay.json"
    return replay_tables(path.stat().st_mtime) if path.exists() else None


@st.cache_data(show_spinner=False)
def replay_tables(modified: float) -> dict:         # modified: read again when the recording changes
    r = snapshot("replay")
    days = pd.to_datetime(pd.Series(r["days"]))
    tickers = pd.DataFrame(r["tickers"])
    rows = pd.read_parquet(SNAPSHOT_DIR / r["rows"])
    rows["date"] = days.iloc[rows["d"]].to_numpy()
    rows["ticker"] = tickers["ticker"].iloc[rows["t"]].to_numpy()
    signals = rows.rename(columns={"ab": "abnormal_return", "sp": "attention_spike"})[
        ["d", "ticker", "date", "ar", "z_ar", "abnormal_return", "views", "z_views", "attention_spike"]]
    preds = rows.dropna(subset=["p"]).rename(columns={"p": "p_big_move_next", "alert": "big_move_alert"})[
        ["d", "ticker", "date", "p_big_move_next", "big_move_alert"]]
    cars = pd.DataFrame(r["windows"]).rename(columns={"event": "event_type", "n_ab": "n_abnormal_days",
                                                      "n_sp": "n_spike_days"})
    cars["ticker"] = tickers["ticker"].iloc[cars["t"]].to_numpy()
    cars["day0_date"] = pd.to_datetime(cars["day0"])
    return {"days": days, "start": r["start"], "signals": signals, "preds": preds, "cars": cars}


def pct(x, digits: int = 2) -> str:
    return "" if pd.isna(x) else f"{x:+.{digits}%}"


# ---------------------------------------------------------------- the stream view (live and replay)

def render_stream(signals: pd.DataFrame, cars: pd.DataFrame, preds: pd.DataFrame, lookback: int,
                  kinds: list[str], focus: str | None, batches: int | None = None):
    """Today's picture of the stream: everything up to the latest day in `signals`."""
    names = retailers()
    signals = signals.merge(names, on="ticker", how="left")
    signals = signals[signals["retailer_type"].isin(kinds)]
    if signals.empty:
        st.info("No rows for this retailer type yet.")
        return
    typed = set(names.loc[names["retailer_type"].isin(kinds), "ticker"])
    cars = cars[cars["ticker"].isin(typed)] if not cars.empty else cars
    today = signals["date"].max()
    on_today = signals[signals["date"] == today]
    alerts_today = on_today[(on_today["abnormal_return"] == 1) | (on_today["attention_spike"] == 1)]

    c = st.columns(3) + st.columns(3)
    c[0].metric("Stream has reached", today.strftime("%d %b %Y"), today.strftime("%A"), delta_color="off")
    c[1].metric("Micro-batches" if batches is not None else "Trading days",
                batches if batches is not None else signals["date"].nunique())
    c[2].metric("Stock-days processed", f"{len(signals):,}")
    c[3].metric("Abnormal returns today", int((on_today["abnormal_return"] == 1).sum()))
    c[4].metric("Attention spikes today", int((on_today["attention_spike"] == 1).sum()))
    c[5].metric("Event windows closed", 0 if cars.empty else len(cars))

    left, right = st.columns(2)
    with left:
        st.subheader(f"Today's alerts · {today:%a %d %b %Y}")
        st.caption(f"Abnormal return: |move the market model did not expect| ≥ {ABNORMAL_RETURN_Z}× its usual "
                   f"size. Attention spike: Wikipedia views ≥ {ATTENTION_SPIKE_Z} sd above their recent level.")
        if alerts_today.empty:
            st.caption("No abnormal return or attention spike today.")
        else:
            show = alerts_today.assign(
                alert=alerts_today.apply(lambda r: "both" if r["abnormal_return"] == 1 and r["attention_spike"] == 1
                                         else "abnormal return" if r["abnormal_return"] == 1 else "attention spike",
                                         axis=1),
                ar=alerts_today["ar"].map(pct))
            st.dataframe(show[["ticker", "company", "alert", "ar", "z_ar", "views", "z_views"]]
                         .sort_values("z_ar", key=abs, ascending=False).round(2), hide_index=True,
                         column_config={"ar": "abnormal return", "z_views": "views z"})
    with right:
        st.subheader("Tomorrow's watchlist · big-move model")
        if preds.empty and signals["z_ar"].notna().any():
            st.caption("No model scores in this run: the job started without a trained model. Train it "
                       "(src/ml/train.py) or download it (see About), then restart the job.")
        elif preds.empty:
            st.caption(f"No scores yet: the model scores a retailer once its market model has "
                       f"{MIN_HISTORY} trading days of history to compare with, so the first scores come "
                       "about six months into a run.")
        else:
            preds = preds.merge(names, on="ticker", how="left")
            preds = preds[preds["retailer_type"].isin(kinds)]
            last_day = preds["date"].max()
            latest = preds[preds["date"] == last_day].sort_values("p_big_move_next", ascending=False)
            st.caption(f"Scored on {last_day:%d %b %Y} for the next trading day: {int(latest['big_move_alert'].sum())} "
                       f"of {len(latest)} retailers flagged (top 5% of scores). The five most likely big movers:")
            st.dataframe(latest.head(5).assign(flagged=latest["big_move_alert"].map({1: "⚑ yes", 0: ""}))
                         [["ticker", "company", "retailer_type", "p_big_move_next", "flagged"]].round(3),
                         hide_index=True, column_config={"p_big_move_next": "score"})
            outcome = (signals.sort_values(["ticker", "date"])
                       .assign(next_big=lambda d: d.groupby("ticker")["abnormal_return"].shift(-1))
                       [["ticker", "date", "next_big"]])
            known = preds[preds["date"] < today].merge(outcome, on=["ticker", "date"]).dropna(subset=["next_big"])
            flagged, quiet = known[known["big_move_alert"] == 1], known[known["big_move_alert"] == 0]
            if len(flagged) and len(quiet) and quiet["next_big"].mean() > 0:
                st.caption(f"So far: {flagged['next_big'].mean():.0%} of {len(flagged):,} flagged stock-days were big "
                           f"moves the next day, against {quiet['next_big'].mean():.1%} of the {len(quiet):,} not flagged "
                           f"({flagged['next_big'].mean() / quiet['next_big'].mean():.1f}× as often).")

    st.subheader("Event windows as they close")
    if cars.empty:
        st.caption("No event window has closed yet.")
    else:
        recent = cars.merge(names, on="ticker", how="left")
        recent = recent[recent["retailer_type"].isin(kinds)].sort_values(["day0_date", "ticker"],
                                                                         ascending=[False, True]).head(20)
        recent = recent.assign(CAR=recent["car"].map(pct), day0_date=recent["day0_date"].dt.date,
                               significant=(recent["car_z"].abs() >= ABNORMAL_RETURN_Z).map({True: "yes", False: ""}))
        st.dataframe(recent[["day0_date", "event_type", "ticker", "company", "CAR", "car_z", "significant",
                             "n_abnormal_days", "n_spike_days"]].round(2), hide_index=True)
        upcoming = calendar()[calendar()["event_date"] > today].head(3)
        if not upcoming.empty:
            st.caption("Next events: " + ", ".join(f"{r.event_type} {r.event_date:%Y-%m-%d}"
                                                   for r in upcoming.itertuples()))

    st.subheader(f"Alerts per trading day · last {lookback} trading days")
    days = sorted(signals["date"].unique())[-lookback:]
    window = signals[signals["date"].isin(days)]
    per_day = window.groupby("date")[["abnormal_return", "attention_spike"]].sum()
    fixed_lines(per_day.rename(columns={"abnormal_return": "abnormal returns",
                                        "attention_spike": "attention spikes"}), [BLUE, ORANGE], "alerts")

    if focus:
        st.subheader(f"{focus} · cumulative abnormal return and attention")
        one = window[window["ticker"] == focus].set_index("date").sort_index()
        if one.empty:
            st.caption("No rows for this retailer in the window yet.")
        else:
            a, b = st.columns(2)
            with a:
                fixed_lines(one["ar"].fillna(0).cumsum().rename("cumulative abnormal return").to_frame(), [BLUE])
            with b:
                fixed_lines(one["z_views"].rename("attention z-score").to_frame(), [BLUE])


def fixed_lines(data: pd.DataFrame, colors: list[str], y_title: str | None = None, height: int = 260):
    """One line per column of `data` (indexed by date), in a chart that does not zoom or pan.
    Streamlit's own line charts zoom when the page is scrolled over them, and the zoom then stays
    through every refresh of the live view, so the dates no longer match the other charts."""
    x = data.index.name or "date"
    long = data.reset_index(names=x).melt(id_vars=x, var_name="series", value_name="value")
    single = len(data.columns) == 1
    chart = alt.Chart(long).mark_line().encode(
        x=alt.X(f"{x}:T", title=None),
        y=alt.Y("value:Q", title=data.columns[0] if single else y_title),
        color=alt.value(colors[0]) if single else legend_color(list(data.columns), colors),
        tooltip=[alt.Tooltip(f"{x}:T", title="date"), alt.Tooltip("series:N", title=""),
                 alt.Tooltip("value:Q", format=",.3~f")])
    st.altair_chart(chart.properties(height=height), use_container_width=True)


def stream_controls():
    with st.sidebar:
        st.header("Stream view")
        lookback = st.slider("Trading days in the charts", 20, 250, 60)
        kinds = st.multiselect("Retailer type", ["online", "store"], default=["online", "store"]) or ["online", "store"]
        focus = st.selectbox("Retailer to follow", [""] + sorted(retailers()["ticker"])) or None
    return lookback, kinds, focus


def live_view():
    lookback, kinds, focus = stream_controls()
    with st.sidebar:
        stream_dir = None
        if CAN_STREAM_HERE:
            runs = local_runs()
            source = st.radio("Show", [PUBLISHED, LOCAL], index=1 if streaming_here() else 0, help=(
                f"**{PUBLISHED}:** the run a streaming job publishes, from the cloud streamer on GitHub or "
                "from `scripts/demo_live.sh` on any laptop. The hosted dashboard shows these. About 15 s behind.\n\n"
                f"**{LOCAL}:** the output folder of a streaming job running on this computer: no delay, and "
                "nothing published, as with `scripts/demo_live.sh --private`. It is picked for you while such "
                "a job is running."))
            if source == LOCAL:
                typed = Path(st.text_input("Output folder", shown(runs[0][0] if runs else DEMO_STREAM_DIR),
                                           help=f"`{shown(DEMO_STREAM_DIR)}`: the live demo (`scripts/demo_live.sh`). "
                                                f"`{shown(STREAM_DIR)}`: the full run (`run_everything.sh`)."))
                stream_dir = typed if typed.is_absolute() else REPO_ROOT / typed
        else:
            st.caption("Shows the run published by the cloud streamer or a laptop (see About).")
        refresh = st.slider("Refresh every (seconds)", 2, 30, 5)

    @st.fragment(run_every=refresh)
    def tick():
        if stream_dir is not None:
            folder = stream_dir
            st.caption(f"Reading `{shown(folder)}` on this computer")
        else:
            current = live_feed()
            if current is None or "error" in current:
                st.info("No live run is being published right now. To show one here, start the cloud streamer "
                        "or `scripts/demo_live.sh` (see the About view), or watch the recorded stream in "
                        f"**{REPLAY}**." + (f"\n\n({current['error']})" if current else ""))
                return
            from hosting.live import age_seconds, run_state
            age, state, n = age_seconds(current), run_state(current), current["batches"]
            ago = f"{age:.0f} s" if age < 120 else f"{age / 60:.0f} min" if age < 7200 else f"{age / 3600:.0f} h"
            feed = f"[{LIVE_REPO}](https://huggingface.co/datasets/{LIVE_REPO})"
            reached = f", up to {pd.Timestamp(current['reached']):%d %b %Y}" if current.get("reached") else ""
            if state == "live":
                st.success(f"**● Live now:** run {current['run_id']} by {current.get('by', 'a teammate')} is "
                           f"streaming; last update {ago} ago, "
                           f"{n} micro-batches so far{reached}. This page refreshes every {refresh} s. ({feed})")
            else:
                why = ("it reached the end of the data" if current.get("complete") else
                       "it was stopped, and can be resumed" if state == "finished" else
                       "it went quiet without finishing (stopped by force, or its computer went offline)")
                st.warning(f"**No run is streaming right now.** Below is where the last run ended: run "
                           f"{current['run_id']}, {n} micro-batches{reached}; {why} {ago} ago. To show a run live, "
                           f"start the cloud streamer or the streaming job (see About); to watch the stream move "
                           f"now, open **{REPLAY}**. ({feed})")
            folder = current["folder"]
        signals = read_stream("stream_signals", folder)
        if signals.empty:
            st.info(f"No streaming output in `{shown(folder)}` yet. To stream on this computer without "
                    "publishing, run `scripts/demo_live.sh --private` in a terminal: it writes to "
                    f"`{shown(DEMO_STREAM_DIR)}`.")
            return
        render_stream(signals, read_stream("stream_event_car", folder), read_stream("stream_predictions", folder),
                      lookback, kinds, focus, batches=signals.attrs["batches"])

    tick()


def replay_view():
    frames = replay_frames()
    if frames is None:
        st.info("No recording yet: build it with `python3 src/hosting/snapshot.py` after a full run.")
        return
    lookback, kinds, focus = stream_controls()
    days = frames["days"]
    years = sorted(days.dt.year.unique().tolist())
    default = (ML_TEST_START_YEAR if ML_TEST_START_YEAR in years else years[0], years[-1])
    ss = st.session_state

    top = st.columns([3, 2])
    y0, y1 = top[0].select_slider("Years to replay", options=years, value=default,
                                  help=f"By default {default[0]}–{default[1]}: the years the big-move model never "
                                       "saw in training.")
    speed = top[1].select_slider("Trading days per second", [1, 2, 5, 10, 20], value=5)
    first = int(days.dt.year.searchsorted(y0, side="left"))          # first and last trading day of the range
    last = int(days.dt.year.searchsorted(y1, side="right")) - 1
    if ss.get("replay_range") != (y0, y1):                           # a new range plays from its start
        ss.replay_range, ss.replay_day, ss.playing = (y0, y1), first, True
    jump_key = f"jump_{y0}_{y1}"

    def jump():
        ss.replay_day, ss.playing = ss[jump_key], False

    c = st.columns([1.4, 6.6])
    if c[0].button("Pause" if ss.playing else "Play", use_container_width=True):
        ss.playing = not ss.playing
        if ss.playing and ss.replay_day >= last:
            ss.replay_day = first
        st.rerun()
    c[1].select_slider("Jump to", options=list(range(first, last + 1)), value=ss.replay_day, key=jump_key,
                       format_func=lambda i: f"{days.iloc[i]:%d %b %Y}", on_change=jump)
    st.caption(f"Recording: what the streaming job wrote for {days.iloc[first]:%d %b %Y} to {days.iloc[last]:%d %b %Y}, "
               "played back day by day. Not live market data; for a live run see **Live stream**.")
    if y0 < ML_TEST_START_YEAR:
        st.warning(f"The big-move model was trained and tuned on the years before {ML_TEST_START_YEAR}, so its "
                   f"scores for those years are not a fair test: it has already seen what happened. Judge the "
                   f"model on {ML_TEST_START_YEAR} onwards.")

    @st.fragment(run_every=1.0)
    def tick():
        if ss.playing:
            ss.replay_day = min(last, ss.replay_day + speed)
            if ss.replay_day >= last:
                ss.playing = False
        d = ss.replay_day
        sig, cars, preds = frames["signals"], frames["cars"], frames["preds"]
        render_stream(sig[sig["d"].between(first, d)], cars[cars["close"].between(first, d)],
                      preds[preds["d"].between(first, d)], lookback, kinds, focus)

    tick()


# ---------------------------------------------------------------- event history and forecast

def legend_color(domain: list[str], colors: list[str]):
    return alt.Color("series:N", scale=alt.Scale(domain=domain, range=colors),
                     legend=alt.Legend(orient="top", title=None, symbolType="stroke", symbolStrokeWidth=3,
                                       symbolOpacity=1))


def event_view():
    ev_data = snapshot("events")
    if ev_data is None:
        st.info("No event history yet: build it with `python3 src/hosting/snapshot.py` after a full run.")
        return
    events = {e["name"]: e for e in ev_data["event_types"]}
    days, F, conf = ev_data["days"], ev_data["forecast_year"], round(ev_data["confidence"] * 100)
    c = st.columns([3, 2, 2])
    name = c[0].selectbox("Event", list(events))
    group = c[1].radio("Retailers", ev_data["groups"], horizontal=True, format_func=str.capitalize)
    years_all = list(range(ev_data["first_year"], ev_data["last_year"] + 1))
    highlight = c[2].selectbox("Highlight year", ["none"] + years_all[::-1], index=1)
    e, g = events[name], events[name]["groups"][group]
    s, tr = g["summary"], g["trend"]
    fc_mid, fc_lo, fc_hi = g["forecast"]["mid"][-1], g["forecast"]["lo"][-1], g["forecast"]["hi"][-1]

    k = st.columns(4)
    if s:
        k[0].metric(f"Average CAR, {ev_data['first_year']}–{ev_data['last_year']}", pct(s["mean_car"], 1),
                    f"{conf}% CI {pct(s['ci_low'], 1)} to {pct(s['ci_high'], 1)} · p = {s['p_clustered']:.3f}",
                    delta_color="off")
    if tr:
        k[1].metric("Change per year", pct(tr["slope"]), f"p = {tr['p']:.3f}", delta_color="off")
    k[2].metric(f"{F} forecast", pct(fc_mid, 1), f"{conf}% interval {pct(fc_lo, 1)} to {pct(fc_hi, 1)}",
                delta_color="off")
    k[3].metric(f"Next {name}", e["next_date"] or "not listed yet")
    if e["also"]:
        st.caption(f"Markets close on {' and '.join(e['also'])}, so it shares this event's window.")

    st.subheader(f"{name} · cumulative abnormal return, day {days[0]} to +{days[-1]}")
    rows = [{"series": "Each year", "year": str(y), "day": d, "car": v}
            for y, path in zip(g["years"], g["paths"]) for d, v in zip(days, path) if v is not None]
    df = pd.DataFrame(rows)
    hl = df[df["year"] == str(highlight)].assign(series=str(highlight))
    mean = pd.DataFrame({"series": "Average", "day": days, "car": g["mean"]})
    fc = pd.DataFrame({"series": f"{F} forecast", "day": days, "car": g["forecast"]["mid"],
                       "lo": g["forecast"]["lo"], "hi": g["forecast"]["hi"]})
    domain = ["Each year"] + ([str(highlight)] if not hl.empty else []) + ["Average", f"{F} forecast"]
    colors = [GREY] + ([AQUA] if not hl.empty else []) + [BLUE, ORANGE]
    color = legend_color(domain, colors)
    x = alt.X("day:Q", title="Trading days from the event", axis=alt.Axis(values=days, format="+d"))
    y = alt.Y("car:Q", title="Cumulative abnormal return", axis=alt.Axis(format="%"))
    tip = [alt.Tooltip("series:N"), alt.Tooltip("day:Q", format="+d"), alt.Tooltip("car:Q", format="+.2%")]
    chart = alt.layer(
        alt.Chart(df).mark_line(strokeWidth=1.25, opacity=0.55).encode(x=x, y=y, detail="year:N", color=color,
                                                                       tooltip=tip + [alt.Tooltip("year:N")]),
        alt.Chart(fc).mark_area(opacity=0.12).encode(x=x, y="lo:Q", y2="hi:Q", color=color),
        alt.Chart(hl).mark_line(strokeWidth=2.5).encode(x=x, y=y, color=color, tooltip=tip),
        alt.Chart(mean).mark_line(strokeWidth=2.5, point=True).encode(x=x, y=y, color=color, tooltip=tip),
        alt.Chart(fc).mark_line(strokeWidth=2.5, strokeDash=[6, 4]).encode(
            x=x, y=y, color=color, tooltip=tip + [alt.Tooltip("lo:Q", format="+.2%", title=f"{conf}% low"),
                                                  alt.Tooltip("hi:Q", format="+.2%", title=f"{conf}% high")]),
        alt.Chart(pd.DataFrame({"day": [0]})).mark_rule(color=GREY, strokeWidth=1).encode(x=x),
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Grey: one line per year, averaged over the {group} retailers. Blue: the average of those years. "
               f"Orange, dashed: a straight-line trend through the years, extended to {F}, with its {conf}% "
               "prediction interval (where one new year's value should fall).")

    left, right = st.columns(2)
    with left:
        st.subheader("Each year's CAR and the trend")
        last_i = len(days) - 1
        yearly = pd.DataFrame({"year": g["years"], "car": [p[last_i] for p in g["paths"]], "series": "CAR that year"})
        parts = [alt.Chart(yearly).mark_point(filled=True, size=70).encode(
            x=alt.X("year:O", title=None), y=alt.Y("car:Q", title="CAR", axis=alt.Axis(format="%")),
            color=legend_color(["CAR that year", "Trend", f"{F} forecast"], [BLUE, ORANGE, ORANGE]),
            tooltip=[alt.Tooltip("year:O"), alt.Tooltip("car:Q", format="+.2%")])]
        if tr:
            line = pd.DataFrame({"year": g["years"] + [F]})
            line["car"] = tr["intercept"] + tr["slope"] * line["year"]
            line["series"] = "Trend"
            point = pd.DataFrame({"year": [F], "car": [fc_mid], "lo": [fc_lo], "hi": [fc_hi], "series": [f"{F} forecast"]})
            parts += [alt.Chart(line).mark_line(strokeWidth=1.5, strokeDash=[6, 4]).encode(
                          x="year:O", y="car:Q", color="series:N"),
                      alt.Chart(point).mark_rule(strokeWidth=2).encode(x="year:O", y="lo:Q", y2="hi:Q", color="series:N"),
                      alt.Chart(point).mark_point(filled=True, size=90).encode(
                          x="year:O", y="car:Q", color="series:N",
                          tooltip=[alt.Tooltip("car:Q", format="+.2%", title=f"{F} forecast"),
                                   alt.Tooltip("lo:Q", format="+.2%", title=f"{conf}% low"),
                                   alt.Tooltip("hi:Q", format="+.2%", title=f"{conf}% high")])]
        st.altair_chart(alt.layer(*parts).properties(height=300), use_container_width=True)
        if tr:
            st.caption(f"Least-squares trend: {pct(tr['slope'])} a year (p = {tr['p']:.3f}). The bar is the {conf}% "
                       f"prediction interval for {F}.")
    with right:
        st.subheader("Predicted vs actual big moves")
        m = g["model"]
        predicted, actual = "Model predicted a big move", "Retailer actually had a big move"
        mdf = pd.DataFrame({"day": days * 2, "share": m["flagged"] + m["big_move"],
                            "series": [predicted] * len(days) + [actual] * len(days)})
        ref = pd.DataFrame({"share": [m["base_rate"]]})
        st.altair_chart(alt.layer(
            alt.Chart(mdf).mark_line(point=True).encode(
                x=alt.X("day:Q", title="Trading days from the event", axis=alt.Axis(values=days, format="+d")),
                y=alt.Y("share:Q", title="Share of retailers", axis=alt.Axis(format="%")),
                color=legend_color([predicted, actual], [BLUE, ORANGE]),
                tooltip=[alt.Tooltip("series:N"), alt.Tooltip("day:Q", format="+d"),
                         alt.Tooltip("share:Q", format=".1%")]),
            alt.Chart(ref).mark_rule(color=GREY, strokeDash=[2, 2]).encode(y="share:Q"),
        ).properties(height=300), use_container_width=True)
        st.caption(f"A **big move** is a day whose abnormal return is at least {ABNORMAL_RETURN_Z}× its usual "
                   f"size. Blue: the share of retailers the big-move model flagged for that day (the day before). "
                   f"Orange: the share that really had one. Test years {ev_data['test_years'][0]}–"
                   f"{ev_data['test_years'][1]}, which the model never saw ({m['n'][0]} stock-days per event day). "
                   f"Dotted line: big moves on an ordinary day ({m['base_rate']:.1%}).")

    st.subheader("How to read this")
    significant = [x["name"] for x in ev_data["event_types"]
                   if x["groups"][group]["trend"] and x["groups"][group]["trend"]["p"] < 1 - ev_data["confidence"]]
    st.markdown(
        f"- The forecast is a straight line through {len(g['years'])} yearly values: a trend, not a prediction of "
        f"the market. {('Only ' + ', '.join(significant) + ' has a trend with p < 0.05') if significant else 'No event has a trend with p < 0.05'} "
        f"for {group} retailers, and the interval shows how wide one year can be.\n"
        f"- Every number uses {ev_data['first_year']}–{ev_data['last_year']} data, so {F}'s outcomes are not in it.\n"
        "- The big-move model predicts tomorrow, not next year. Big moves cluster, and around events its flags often "
        "come a day after a burst of big moves starts.\n"
        "- Intervals and p-values for the average count each event date once (clustered), because the retailers "
        "share each event and move together.")


# ---------------------------------------------------------------- about

def about_view():
    st.markdown(f"""
**Calendar Effects** is the CMPT 732 (Big Data, SFU) project of Mahad, Sanchi, Anushiya and Areej. It asks
whether retail stocks move more than the market explains around shopping events (Black Friday, Cyber Monday,
Prime Day, Halloween, Chinese New Year), and whether Wikipedia attention moves with them.

- **Streaming:** ten years of daily prices and pageviews for {len(retailers())} retailers are replayed through
  Kafka into Spark Structured Streaming. The job fits a market model for every stock every day, flags abnormal
  returns and attention spikes, closes each event's cumulative abnormal return (CAR) on its day +5, and scores
  tomorrow's big moves with a Spark MLlib model.
- **Checked twice:** a batch event study computes the same CARs independently; all of them match.
- **This dashboard** is the same app on a laptop and here. *Live stream* shows a real run as it happens;
  *{REPLAY}* plays the recorded stream; *Event history & forecast* shows each event over the years.

**Show a live run here.** Either way replays the stored data from mid-{ML_TEST_START_YEAR - 1} through the Spark job and publishes
each micro-batch here:

- **The cloud streamer, no computer needed** (team members): on GitHub, the [*Live stream* workflow]({LINKS["streamer"]})
  → *Run workflow*, to start from a date or resume the last run. *Cancel run* stops it; it stops by itself when
  the data runs out.
- **From a laptop:** the command below sets Java 21 by itself, and needs `hf auth login` with write access to the
  team's organisation. Ctrl-C stops it; `--resume` carries it on.

```bash
scripts/demo_live.sh
```

This page opens on *Live stream* while a run is streaming, and shows each micro-batch within about 15 seconds.
A second run never takes over: it appears only once no other run is streaming.

*"Live" means the processing is live (Kafka, Spark, the model and this page, as each micro-batch arrives); the
data is a replay of {STUDY_START_YEAR}–{STUDY_END_YEAR}, the study period.*

**Links:** {" · ".join(f"[{name}]({LINKS[k]})" for k, name in LINK_NAMES.items())} · hosted dashboard: {DASHBOARD_URL}

Only statistics we computed are shown or published: no prices, volumes or raw returns (Yahoo's terms).
""")


def main():
    st.set_page_config(page_title="Calendar Effects", page_icon="📈", layout="wide")
    st.title("Calendar Effects")
    st.caption("Retail stocks and Wikipedia attention around shopping events, streamed through Kafka and Spark "
               "Structured Streaming, with a Spark MLlib model of tomorrow's big moves.")
    # Open on Live stream when a run is streaming: a job writing on this computer right now, or a
    # run streaming to the live feed (the hosted app during a demo). Otherwise open on Replay.
    from hosting.live import run_state
    feed = live_feed()
    streaming = streaming_here() or run_state(feed if feed and "error" not in feed else None) == "live"
    view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", index=0 if streaming else 1)
    {VIEWS[0]: live_view, VIEWS[1]: replay_view, VIEWS[2]: event_view, VIEWS[3]: about_view}[view]()


if __name__ == "__main__":   # streamlit runs the file as __main__; tests import it
    main()
