"""Single source of truth for dataset schemas, in code, plus the loaders for the
config/ registries that are themselves contracts.

This mirrors docs/data-contracts.md. tests/test_contracts.py fails if the two
disagree, so always change both in the same commit.

Usage:
    from common.contracts import load_retailers, pandas_columns, require_contract, spark_schema

    tickers = [r["ticker"] for r in load_retailers(in_scope_only=True)]
    require_contract("prices", pandas_columns(df))      # before writing a clean file
    stream = spark.readStream.schema(spark_schema("master"))...
"""
from __future__ import annotations

import calendar
import csv
import re
from datetime import date, timedelta
from typing import NamedTuple

from common.settings import (
    EVENT_DATES_CSV,
    EVENT_TYPES_CSV,
    RETAIL_SERIES_CSV,
    RETAILERS_CSV,
    WIKI_ARTICLES_CSV,
)
from common.study import MARKET_TICKER

# The only type names allowed in contracts (see docs/data-contracts.md).
ALLOWED_TYPES = {"string", "long", "int", "double", "date", "timestamp"}

# dataset -> {column: type}. Column order here is the order used for Spark schemas
# and the column order the config/ CSV files must have.
CONTRACTS: dict[str, dict[str, str]] = {
    # config/: hand-edited registries of what is studied
    "event_types": {
        "event_type": "string",
        "name": "string",
        "category": "string",
        "region": "string",
        "date_rule": "string",
        "in_scope": "string",
        "notes": "string",
    },
    "event_dates": {
        "event_type": "string",
        "event_date": "date",
        "event_end_date": "date",
        "source": "string",
    },
    "retailers": {
        "ticker": "string",
        "company": "string",
        "country": "string",
        "retailer_type": "string",
        "in_scope": "string",
        "notes": "string",
    },
    "wiki_articles": {
        "article": "string",
        "ticker": "string",
        "notes": "string",
    },
    "retail_series": {
        "series_id": "string",
        "category": "string",
        "notes": "string",
    },
    # Built from config/ whenever it is needed (expand_events)
    "events": {
        "event_type": "string",
        "event_date": "date",
        "event_end_date": "date",
        "ticker": "string",
    },
    # clean/: ETL output, committed
    "prices": {
        "ticker": "string",
        "date": "date",
        "adj_close": "double",
        "volume": "long",
        "ret": "double",
        "vol_ratio": "double",
    },
    "pageviews": {
        "article": "string",
        "ticker": "string",
        "date": "date",
        "views": "int",
    },
    "retail_sales": {
        "category": "string",
        "month": "date",
        "sales": "long",
        "mom_change": "double",
        "release_date": "date",
    },
    "master": {
        "ticker": "string",
        "date": "date",
        "adj_close": "double",
        "volume": "long",
        "ret": "double",
        "vol_ratio": "double",
        "views": "double",
        "event_type": "string",
        "country": "string",
        "retailer_type": "string",
    },
    # data/stream/: streaming job output
    "stream_signals": {
        "ticker": "string",
        "date": "date",
        "timestamp_utc": "timestamp",
        "ret": "double",
        "mkt_ret": "double",
        "alpha": "double",
        "beta": "double",
        "ar": "double",
        "ar_sd": "double",
        "z_ar": "double",
        "abnormal_return": "int",
        "views": "double",
        "z_views": "double",
        "attention_spike": "int",
        "vol_ratio": "double",
        "event_type": "string",
        "estimation_days": "int",
    },
    "stream_event_car": {
        "ticker": "string",
        "event_type": "string",
        "event_date": "date",
        "event_end_date": "date",
        "day0_date": "date",
        "alpha": "double",
        "beta": "double",
        "ar_sd": "double",
        "estimation_days": "int",
        "car": "double",
        "car_z": "double",
        "n_days": "int",
        "n_abnormal_days": "int",
        "max_z_views": "double",
        "n_spike_days": "int",
    },
    # Areej's batch event study (src/eventstudy/car.py): the answer key the stream is
    # compared against, so it has exactly the columns of stream_event_car.
    "batch_event_car": {
        "ticker": "string",
        "event_type": "string",
        "event_date": "date",
        "event_end_date": "date",
        "day0_date": "date",
        "alpha": "double",
        "beta": "double",
        "ar_sd": "double",
        "estimation_days": "int",
        "car": "double",
        "car_z": "double",
        "n_days": "int",
        "n_abnormal_days": "int",
        "max_z_views": "double",
        "n_spike_days": "int",
    },
    # Significance tables from src/eventstudy/stats.py, for the report.
    "batch_car_summary": {
        "event_type": "string",
        "group": "string",
        "n_windows": "int",
        "n_events": "int",
        "n_tickers": "int",
        "mean_car": "double",
        "median_car": "double",
        "ci_low": "double",
        "ci_high": "double",
        "pct_positive": "double",
        "p_t": "double",
        "p_wilcoxon": "double",
        "p_bmp": "double",
        "p_clustered": "double",
    },
    "batch_car_trend": {
        "event_type": "string",
        "n_events": "int",
        "n_tickers": "int",
        "first_year": "int",
        "last_year": "int",
        "slope_per_year": "double",
        "ci_low": "double",
        "ci_high": "double",
        "p_value": "double",
    },
    # data/stream/predictions/: the big-move model scored live by the streaming job
    "stream_predictions": {
        "ticker": "string",
        "date": "date",
        "timestamp_utc": "timestamp",
        "p_big_move_next": "double",
        "big_move_alert": "int",
    },
    # data/results/: the big-move model's evaluation (src/ml/train.py)
    "ml_metrics": {
        "model": "string",
        "params": "string",
        "features": "string",
        "split": "string",
        "n_rows": "long",
        "n_big_moves": "long",
        "base_rate": "double",
        "auc_roc": "double",
        "auc_pr": "double",
        "precision_top": "double",
        "lift_top": "double",
    },
    "ml_test_predictions": {
        "ticker": "string",
        "date": "date",
        "p_big_move_next": "double",
        "big_move_next": "int",
    },
    "ml_feature_importance": {
        "model": "string",
        "feature": "string",
        "importance": "double",
    },
}

# Kafka topic per streamed dataset. One master topic replaced the planned prices and
# attention topics (decision 0004); the cluster is shared by the class, so the name
# carries the project prefix.
KAFKA_TOPIC_PREFIX = "calendar-effects"
KAFKA_TOPICS = {"master": f"{KAFKA_TOPIC_PREFIX}-master"}

# Ticker value for events that apply to the whole market.
MARKET_WIDE_TICKER = "ALL"

# Grammar for event_types.date_rule.
DATE_RULE = re.compile(
    r"^(fixed:(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
    r"|nth_weekday:(0[1-9]|1[0-2]):(mon|tue|wed|thu|fri|sat|sun):(-1|[1-5])"
    r"|offset:[a-z][a-z0-9_]*:[+-][0-9]+"
    r"|listed)$"
)


def read_registry(dataset: str, path) -> list[dict[str, str]]:
    """Rows of a config/ CSV as dicts, after checking its header against the contract,
    so a renamed or missing column fails here instead of as a KeyError somewhere later.
    utf-8-sig also reads files saved by Excel as "CSV UTF-8", which start with a byte-order mark."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(CONTRACTS[dataset]):
            raise ValueError(
                f"{path}: columns are {reader.fieldnames}, "
                f"the {dataset!r} contract says {list(CONTRACTS[dataset])}"
            )
        return list(reader)


def load_event_types(path=EVENT_TYPES_CSV, in_scope_only: bool = False) -> list[dict[str, str]]:
    """Read the event registry as a list of rows (dicts keyed by column name).

    Pass in_scope_only=True to get only the events the pipeline uses now.
    Pass a different path when running from a cluster package where config/ lives elsewhere.
    """
    rows = read_registry("event_types", path)
    return [row for row in rows if not in_scope_only or row["in_scope"] == "yes"]


def load_retailers(path=RETAILERS_CSV, in_scope_only: bool = False) -> list[dict[str, str]]:
    """Read config/retailers.csv, the stocks the project studies (the market index,
    common.study.MARKET_TICKER, is not in it). in_scope_only=True drops candidates."""
    rows = read_registry("retailers", path)
    return [row for row in rows if not in_scope_only or row["in_scope"] == "yes"]


def price_tickers(retailers_path=RETAILERS_CSV) -> list[str]:
    """Every ticker that needs prices: the in-scope retailers, then the market index."""
    return [r["ticker"] for r in load_retailers(retailers_path, in_scope_only=True)] + [MARKET_TICKER]


def load_wiki_articles(
    path=WIKI_ARTICLES_CSV, in_scope_only: bool = False, retailers_path=RETAILERS_CSV
) -> list[dict[str, str]]:
    """Read config/wiki_articles.csv: Wikipedia titles per ticker, several per ticker when
    an article was renamed. in_scope_only=True keeps titles of in-scope retailers only."""
    rows = read_registry("wiki_articles", path)
    if not in_scope_only:
        return rows
    tickers = {r["ticker"] for r in load_retailers(retailers_path, in_scope_only=True)}
    return [row for row in rows if row["ticker"] in tickers]


def load_retail_series(path=RETAIL_SERIES_CSV) -> list[dict[str, str]]:
    """Read config/retail_series.csv: FRED series code -> retail category."""
    return read_registry("retail_series", path)


def check_columns(dataset: str, actual: dict[str, str]) -> list[str]:
    """Compare actual {column: type} pairs with the contract.

    Returns a list of human-readable problems; an empty list means it matches.
    """
    expected = CONTRACTS[dataset]
    problems = []
    for column, expected_type in expected.items():
        if column not in actual:
            problems.append(f"{dataset}: missing column {column!r}")
        elif actual[column] != expected_type:
            problems.append(
                f"{dataset}: column {column!r} is {actual[column]}, contract says {expected_type}"
            )
    for column in actual:
        if column not in expected:
            problems.append(f"{dataset}: unexpected column {column!r}")
    return problems


def require_contract(dataset: str, actual: dict[str, str]) -> None:
    """Raise ValueError listing every mismatch. Call it before writing a dataset."""
    problems = check_columns(dataset, actual)
    if problems:
        raise ValueError("does not match docs/data-contracts.md:\n  " + "\n  ".join(problems))


def _arrow_type_name(arrow_type) -> str:
    import pyarrow as pa

    if pa.types.is_timestamp(arrow_type):
        return "timestamp"
    names = {"string": "string", "large_string": "string", "int64": "long", "int32": "int",
             "double": "double", "date32[day]": "date"}
    return names.get(str(arrow_type), str(arrow_type))


def pandas_columns(df) -> dict[str, str]:
    """{column: contract type} of a pandas DataFrame, as it will be stored in Parquet.

    Uses pyarrow's own conversion, so a column of Python dates reads as `date` and a
    pandas datetime column reads as `timestamp`, exactly as the written file would.
    """
    import pyarrow as pa

    return {f.name: _arrow_type_name(f.type) for f in pa.Schema.from_pandas(df, preserve_index=False)}


def parquet_columns(path) -> dict[str, str]:
    """{column: contract type} of a Parquet file or a folder of them (such as a Spark
    output), read from the schema only. Subfolder names like batch_id=3 are not columns.

    Prefer this to pandas_columns for files on disk: pandas reads an int column that
    has nulls as double, while the file itself stores int.
    """
    import pyarrow.dataset as ds

    return {f.name: _arrow_type_name(f.type) for f in ds.dataset(str(path), format="parquet").schema}


def spark_schema(dataset: str):
    """Return a pyspark StructType for a dataset (pyspark is imported only here)."""
    from pyspark.sql import types as T

    spark_types = {
        "string": T.StringType(),
        "long": T.LongType(),
        "int": T.IntegerType(),
        "double": T.DoubleType(),
        "date": T.DateType(),
        "timestamp": T.TimestampType(),
    }
    return T.StructType(
        [T.StructField(column, spark_types[kind], True) for column, kind in CONTRACTS[dataset].items()]
    )


_WEEKDAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _nth_weekday_of_month(year: int, month: int, weekday_name: str, n: int) -> date:
    """The n-th occurrence of weekday_name in (year, month); n=-1 means the last one."""
    weekday = _WEEKDAY_INDEX[weekday_name]
    if n == -1:
        last_day = calendar.monthrange(year, month)[1]
        d = date(year, month, last_day)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        return d
    first_occurrence = date(year, month, 1)
    first_occurrence += timedelta(days=(weekday - first_occurrence.weekday()) % 7)
    occurrences_in_month = (calendar.monthrange(year, month)[1] - first_occurrence.day) // 7 + 1

    if n > occurrences_in_month:
        raise ValueError(
            f"{year}-{month:02d} has only {occurrences_in_month} {weekday_name}(s), "
            f"so there is no {n}th one (date_rule asked for n={n})"
        )
    return first_occurrence + timedelta(weeks=n - 1)


class _Fixed(NamedTuple):
    month: int
    day: int


class _NthWeekday(NamedTuple):
    month: int
    weekday_name: str
    n: int


class _Offset(NamedTuple):
    anchor: str
    days: int


def _parse_date_rule(rule: str) -> _Fixed | _NthWeekday | _Offset | None:
    """The parsed date_rule; None for 'listed', whose dates are in event_dates.csv."""
    if not DATE_RULE.match(rule):
        raise ValueError(f"date_rule {rule!r} does not match the DATE_RULE grammar in docs/data-contracts.md")
    if rule == "listed":
        return None
    if rule.startswith("fixed:"):
        mm, dd = rule.split(":")[1].split("-")
        return _Fixed(int(mm), int(dd))
    if rule.startswith("nth_weekday:"):
        _, mm, weekday_name, n = rule.split(":")
        return _NthWeekday(int(mm), weekday_name, int(n))
    if rule.startswith("offset:"):
        _, anchor, delta = rule.split(":")
        return _Offset(anchor, int(delta))
    raise ValueError(f"unrecognized date_rule: {rule!r}")


def _resolve_date_for_year(event_type: str, year: int, types_by_id: dict, cache: dict, resolving: set) -> date:
    key = (event_type, year)
    if key in cache:
        return cache[key]
    if key in resolving:
        raise ValueError(f"circular offset reference involving {event_type!r}")
    resolving.add(key)

    if event_type not in types_by_id:
        raise ValueError(
            f"date_rule references event_type {event_type!r}, which is not in event_types.csv "
            f"or is not in_scope"
        )

    rule = _parse_date_rule(types_by_id[event_type]["date_rule"])
    if isinstance(rule, _Fixed):
        result = date(year, rule.month, rule.day)
    elif isinstance(rule, _NthWeekday):
        result = _nth_weekday_of_month(year, rule.month, rule.weekday_name, rule.n)
    elif isinstance(rule, _Offset):
        anchor_date = _resolve_date_for_year(rule.anchor, year, types_by_id, cache, resolving)
        result = anchor_date + timedelta(days=rule.days)
    else:
        raise ValueError(
            f"{event_type!r} has date_rule 'listed'; its dates come from event_dates.csv, not a formula"
        )

    resolving.discard(key)
    cache[key] = result
    return result


def expand_events(
    start_year: int,
    end_year: int,
    event_types_path=EVENT_TYPES_CSV,
    event_dates_path=EVENT_DATES_CSV,
    ticker: str = MARKET_WIDE_TICKER,
) -> list[dict]:
    """Expand config/event_types.csv rules and config/event_dates.csv listed dates into
    one row per event occurrence per year, matching the `events` contract
    (event_type, event_date, event_end_date, ticker).

    Only in-scope event types are expanded. fixed/nth_weekday/offset rules are computed
    for every year in [start_year, end_year]; listed rules are read straight from
    event_dates_path and kept if their occurrence overlaps that range at all.

    An offset event anchored in one year can land in the next (e.g. an event a few days
    after Christmas), so offset/fixed/nth_weekday dates are computed over a padded year
    range and then filtered by where they actually land, not by the anchor's year.
    """
    types = load_event_types(path=event_types_path, in_scope_only=True)
    types_by_id = {row["event_type"]: row for row in types}

    rules = [_parse_date_rule(row["date_rule"]) for row in types]
    max_offset_days = max((abs(rule.days) for rule in rules if isinstance(rule, _Offset)), default=0)
    pad_years = max_offset_days // 365 + 1

    rows: list[dict] = []
    cache: dict = {}
    for row in types:
        if row["date_rule"] == "listed":
            continue
        for year in range(start_year - pad_years, end_year + pad_years + 1):
            d = _resolve_date_for_year(row["event_type"], year, types_by_id, cache, set())
            if start_year <= d.year <= end_year:
                rows.append({"event_type": row["event_type"], "event_date": d, "event_end_date": d, "ticker": ticker})

    listed_ids = {row["event_type"] for row in types if row["date_rule"] == "listed"}
    for listed_row in read_registry("event_dates", event_dates_path):
        if listed_row["event_type"] not in listed_ids:
            continue
        event_date = date.fromisoformat(listed_row["event_date"])
        event_end_date = date.fromisoformat(listed_row["event_end_date"])
        if event_date.year > end_year or event_end_date.year < start_year:
            continue
        rows.append({
            "event_type": listed_row["event_type"],
            "event_date": event_date,
            "event_end_date": event_end_date,
            "ticker": ticker,
        })

    rows.sort(key=lambda r: (r["event_date"], r["event_type"]))
    return rows


def spark_columns(df) -> dict[str, str]:
    """Turn a Spark DataFrame's schema into {column: contract type} for check_columns."""
    names = {
        "StringType": "string",
        "LongType": "long",
        "IntegerType": "int",
        "DoubleType": "double",
        "DateType": "date",
        "TimestampType": "timestamp",
    }
    return {f.name: names.get(type(f.dataType).__name__, type(f.dataType).__name__) for f in df.schema.fields}
