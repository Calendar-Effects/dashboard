"""Study settings shared by the streaming job (src/streaming/) and the batch event
study (src/eventstudy/).

The streaming vs batch comparison only lines up if both sides use the same return
definition, market index and window lengths, so they are defined here once
(decision 0004). Import them; never copy the numbers into another file.

Usage:
    from common.study import EVENT_WINDOW, ESTIMATION_DAYS, MARKET_TICKER
"""
from datetime import date

# The study period. Every download, cleaning step and event list uses these. To change
# the period, change these two lines, delete raw/prices/ and raw/wiki/ (the downloads
# skip files that already exist), and re-run the ETL.
STUDY_START_YEAR = 2016
STUDY_END_YEAR = 2025
STUDY_START = date(STUDY_START_YEAR, 1, 1)
STUDY_END = date(STUDY_END_YEAR, 12, 31)

# Market model: ret_stock = alpha + beta * ret_market + error, fitted by OLS.
# The market index is downloaded alongside the retailers but is not one of them,
# so it is not in config/retailers.csv.
MARKET_TICKER = "SPY"
# Labels the market index gets in master. Retailers get theirs from config/retailers.csv.
MARKET_COUNTRY = "US"
MARKET_RETAILER_TYPE = "market"
# Simple returns, ret = adj_close / previous adj_close - 1, as stored in clean/prices.parquet.
RETURN_TYPE = "simple"

# Event window in trading days around day 0, both ends included.
# Day 0 is the first trading day on or after the calendar date, so an event on a
# weekend or market holiday (US Thanksgiving, some Halloweens) starts on the next open day.
EVENT_WINDOW = (-5, 5)

# Estimation window: this many trading days, ending right before the event window.
ESTIMATION_DAYS = 120
# Fewer usable days than this (start of the data, missing returns) means no estimate.
MIN_ESTIMATION_DAYS = 100

# The streaming job re-estimates the model every day. Skipping the most recent
# ESTIMATION_GAP_DAYS trading days keeps every day of an event window out of its own
# baseline: for day +5 the estimation window then ends on day -6, exactly where the
# batch study's fixed window ends.
ESTIMATION_GAP_DAYS = EVENT_WINDOW[1] - EVENT_WINDOW[0]

# Confidence level of the intervals in the batch study's tables (src/eventstudy/stats.py).
CONFIDENCE = 0.95

# Streaming alerts. |abnormal return| / its standard deviation at or above this is "abnormal".
ABNORMAL_RETURN_Z = 1.96
# Attention spike: views at least this many standard deviations above a baseline of
# ATTENTION_BASELINE_DAYS trading days (with the same gap as the market model).
# Anushiya owns the spike rule, settled 2026-11-17. Confirmed against 10 years of
# real data: ~6% of ticker-days flag at z>=3, and Black Friday/Cyber Monday windows
# flag about 3x as often as other days; z=2.5 or z=3.5 give the same lift, just a
# different count, so 3.0 is kept as a reasonable middle choice (see docs/decisions/0004).
ATTENTION_BASELINE_DAYS = 28
MIN_ATTENTION_DAYS = 20
ATTENTION_SPIKE_Z = 3.0

# Machine learning (decision 0007): from what the stream knows about a stock today, will
# tomorrow be a big move, i.e. tomorrow's daily alert (|z_ar| >= ABNORMAL_RETURN_Z)?
# The split follows time, never random rows, so the model is always judged on days after
# the ones it learned from: fit on the training years, pick the model on the validation
# year, then refit on both and report on the test years.
ML_TRAIN_END_YEAR = 2022
ML_VALIDATION_YEAR = ML_TRAIN_END_YEAR + 1
ML_TEST_START_YEAR = ML_VALIDATION_YEAR + 1
# The live job flags this share of stock-days, the highest predicted probabilities, as
# likely big movers. The cut-off comes from the training years' predictions.
ML_ALERT_TOP_SHARE = 0.05
# Days to and since the nearest in-scope event are capped here (a quiet stretch is quiet).
ML_EVENT_DAYS_CAP = 60
