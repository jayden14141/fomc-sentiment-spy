# fomc-sentiment-spy

A financial-ML pipeline testing whether FinBERT sentiment over FOMC statements predicts SPY's next-day return. n = 125 events from 2011–2026. The signal does not survive a permutation test.

---

## What This Is

Five years of academic and trading-desk literature treats FOMC statements as one of the most market-moving texts published in any given quarter. This project tests the simplest possible form of that hypothesis:

> Does the polarity of an FOMC statement, scored by a pretrained financial-domain language model, contain information about SPY's next trading-day return?

The pipeline scrapes the statements, scores them with FinBERT, joins to SPY daily returns, runs linear and tree-based models against a mean baseline, and runs three sizing rules through a backtest with a permutation-based significance test.

Result: across correlation, out-of-sample R², and permutation p-values, no rejection of the null. Three independent measurements agree.

---

## Pipeline

Each notebook reads the previous one's CSV and writes a new one. Re-runnable end-to-end.

| Notebook | Purpose | Output |
|---|---|---|
| [01_scraping.ipynb](notebooks/01_scraping.ipynb) | Scrape FOMC statement URLs from federalreserve.gov | `fomc_urls.csv` (129 rows, 2011→2026) |
| [02_diagnose.ipynb](notebooks/02_diagnose.ipynb) | Probe selectors, extract + clean statement text | `fomc_statements.csv` |
| [03_spy_returns.ipynb](notebooks/03_spy_returns.ipynb) | Pull SPY daily prices, compute `ret` and `ret_next` | `spy_daily.csv` (4,121 rows) |
| [04_sentiment.ipynb](notebooks/04_sentiment.ipynb) | Filter to policy statements, score with FinBERT | `fomc_sentiment.csv` (126 rows) |
| [05_merge.ipynb](notebooks/05_merge.ipynb) | Join sentiment with SPY on FOMC date | `fomc_spy.csv` (125 rows) |
| [06_eda.ipynb](notebooks/06_eda.ipynb) | Distribution, correlation, regime check | — |
| [07_model.ipynb](notebooks/07_model.ipynb) | Linear regression, walk-forward CV vs `DummyRegressor` | — |
| [08_random_forest.ipynb](notebooks/08_random_forest.ipynb) | Random Forest, single- and multi-feature | — |
| [09_backtest.ipynb](notebooks/09_backtest.ipynb) | Long/Short, Long/Cash, Proportional; risk metrics; permutation test | — |

---

## Assumptions

The choices below are baked into the code. Each one shifts the result; none are defaults.

1. **Coverage starts in 2011.** The Fed's historical URL pattern `monetary\d{8}a?\.htm` only matches 2011 onward, so the 2008 GFC is excluded.
2. **Statements release at 14:00 ET; positions are taken at the FOMC-day close (16:00 ET).** The strategy assumes statement → FinBERT → execution in a two-hour window.
3. **Target is `ret_next` = close-to-close from FOMC day to the next trading day.** Some of the announcement reaction is already in the FOMC-day close, so the predictable component left in `ret_next` is small by construction.
4. **FinBERT is trained on Reuters financial news, not Fed-speak.** Fed language is deliberately hedged (*appropriate*, *patient*, *data-dependent*). The regime check in `06_eda` shows dovish-period mean sentiment = 0.297 vs hawkish = 0.106 — a real but small gap, suggesting partial alignment only.
5. **Truncation at 512 tokens, tail-cut.** FOMC puts the policy paragraph at the top, so the voting members and risk-balance sections are sometimes dropped.
6. **Sentiment score = `prob_pos − prob_neg`.** A 0-centered scalar that discards the neutral mass (~52% on average).
7. **Direction convention: `sign(sentiment) > 0 → long SPY`.** This is one of two choices. The permutation test on the Proportional strategy (p = 0.846) suggests the inverted sign would have produced higher returns in this sample.
8. **No transaction costs, no slippage.** Long/Short flips position every event (≥2 trades/meeting). 5 bps round-trip would shave ~1.5–2% off the headline return.
9. **The emergency Sunday meeting on 2020-03-15 is dropped**, not reassigned to the next open. The market reaction that week reflects COVID, not the statement.
10. **Three non-policy statements** (framework updates, FIMA repo facility) are dropped via the keyword filter `federal funds rate | target range`, and manually audited.
11. **The permutation test shuffles `pos` and keeps `ret_next` fixed.** This breaks the pairing while preserving each strategy's marginal exposure — the relevant null.

---

## Techniques

### NLP & Text
- HTTP scraping with rate limiting, selector probing, NFKC unicode normalization, encoding overrides
- Pretrained transformer inference: FinBERT via HuggingFace, tokenization, 512-token truncation, softmax over logits, polarity aggregation

### Data Engineering
- Stage-isolated CSV pipeline (one notebook → one output), explicit datetime joins, left-join with audit (not silent `dropna`)

### Statistical EDA
- Variance ratio (event vs baseline), Pearson + Spearman (rank-based, outlier-robust), outlier-stripped re-estimation, external-knowledge regime overlay

### Modelling
- Time-series cross-validation (expanding-window `TimeSeriesSplit`)
- `DummyRegressor(mean)` baseline; out-of-sample R² and RMSE
- Multicollinearity control: dropped compositional probability columns, `drop_first=True` on dummies
- Feature engineering: first difference (`sent_delta`), document length, regime one-hot
- Random Forest (`max_depth=3`, 200 trees), feature importance inspection

### Backtesting
- Three sizing schemes: sign-based long/short, sign-based long/cash, leakage-free proportional (expanding max)
- Dual benchmark: FOMC-day-only hold vs true daily-compounded buy-and-hold (log-scale overlay)
- Risk metrics: Sharpe (annualized by √8 ≈ FOMC meetings/year), max drawdown, hit rate, average win / average loss
- Permutation significance test (1,000 shuffles, seeded), one-sided p-value, null-distribution histograms

---

## Results

### Signal Strength (06_eda)

| Test | Value |
|---|---|
| `sentiment ~ ret_next` Pearson r | **−0.082** (p = 0.36) |
| Same, top-8 outliers removed | −0.065 (p = 0.49) |
| FOMC-day volatility ratio vs baseline | **1.42×** |
| Mean sentiment, dovish regimes | 0.297 (n=74) |
| Mean sentiment, hawkish regimes | 0.106 (n=43) |

FOMC days move SPY more than average. FinBERT's polarity score does not predict the direction.

### Predictive Models (07, 08)

Every out-of-sample R² is negative — each model is worse than predicting the train-set mean.

| Model | Target | OOS R² |
|---|---|---|
| Linear regression, sentiment only | `ret_next` | **−0.15** |
| Linear regression, sentiment only | `ret` | −0.02 |
| Random Forest, sentiment only | `ret_next` | −0.36 |
| Random Forest, multi-feature | `ret_next` | −0.29 |

RF's additional capacity hurts because there is no signal to fit. Multi-feature top importances: `sent_delta` (0.53), `doc_len` (0.29), `sentiment` (0.17).

### Backtest (09)

125 events, $10,000 seed.

| Strategy | Total return | Sharpe (×√8) | Max DD | Hit rate | Perm p |
|---|---|---|---|---|---|
| Long/Short | **−6.1%** | −0.07 | −14.0% | 48.0% | 0.442 |
| Long/Cash | −8.1% | −0.12 | −11.4% | 48.1% | 0.474 |
| Proportional | −9.3% | −0.31 | −12.0% | 48.0% | **0.846** |
| FOMC-day-only hold | −10.7% | −0.15 | −16.2% | 48.8% | — |
| **True buy-and-hold (daily SPY)** | **+622%** | — | — | — | — |

---

## Key Findings

- **No signal survives the permutation test.** All p-values cluster around 0.5 — the realised returns are well inside the null distribution from random sign assignment.
- **Proportional p = 0.846 implies the FinBERT direction is anti-correlated** with subsequent SPY returns in this sample. Inverting the sign would have produced modestly higher returns. Consistent with the negative (but non-significant) Pearson r in EDA.
- **FOMC days do move SPY more than average (vol ratio 1.42×)**, but FinBERT polarity cannot time the direction. "Important event" and "tradable signal" are not the same thing.
- **The gap between any FOMC-only strategy (−6 to −11%) and true buy-and-hold (+622%)** is mostly the cost of being out of the market 97% of the time during a 15-year bull run, not a property of the signal itself.
- **Three orthogonal measurements agree on no signal**: R² < 0 across linear and tree models, |r| < 0.1 in EDA with and without outliers, and permutation p ≈ 0.5 across three independent sizing rules.

### What Would Change the Conclusion

- A sentiment model fine-tuned on Fed-speak (e.g. RoBERTa on FOMC + Fed speeches) instead of FinBERT
- A larger event set — extending the scraper to pre-2011 (~20 more meetings including the GFC)
- A finer target — intraday return from 14:00 to 16:00 on the FOMC day itself
- Multi-modal features — dot-plot revisions, SEP changes, surprise vs Fed-funds-futures-implied path

---

## Build & Run

**Prerequisites**: Python 3.10+

```bash
pip install -r requirements.txt
jupyter lab
# run notebooks 01 → 09 in order
```
