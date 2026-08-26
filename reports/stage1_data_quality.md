# TESSCH — Simulated Data Quality Report

## 1. Volume

| Metric           | Value                              |
|------------------|------------------------------------|
| Products         | 300                                |
| Users            | 80                                 |
| Sessions         | 328                                |
| Events           | 11,285                             |
| Orders           | 27                                 |
| Revenue          | INR 93,358                         |
| Window           | 2026-07-26 to 2026-08-24 (29 days) |
| Events / user    | 141.1                              |
| Events / session | 34.4                               |
| Sessions / user  | 4.1                                |

## 2. Event mix

| Event type         |   Count | Share   |
|--------------------|---------|---------|
| product_impression |   7,291 | 64.61%  |
| product_view       |   1,286 | 11.40%  |
| product_click      |   1,035 | 9.17%   |
| product_hover      |     778 | 6.89%   |
| page_view          |     270 | 2.39%   |
| home_view          |     254 | 2.25%   |
| search             |     137 | 1.21%   |
| wishlist_add       |     114 | 1.01%   |
| add_to_cart        |      66 | 0.58%   |
| purchase           |      27 | 0.24%   |
| remove_from_cart   |      19 | 0.17%   |
| product_share      |       7 | 0.06%   |
| wishlist_remove    |       1 | 0.01%   |

## 3. Conversion funnel

| Stage              |   Count | vs previous   | vs impression   |
|--------------------|---------|---------------|-----------------|
| product_impression |   7,291 | —             | 100.000%        |
| product_click      |   1,035 | 14.20%        | 14.196%         |
| product_view       |   1,286 | 124.25%       | 17.638%         |
| wishlist_add       |     114 | 8.86%         | 1.564%          |
| add_to_cart        |      66 | 57.89%        | 0.905%          |
| purchase           |      27 | 40.91%        | 0.370%          |

A real footwear funnel loses roughly an order of magnitude at each
of click, cart and purchase. Impression-to-purchase in the low tenths
of a percent is the shape to expect.

## 4. Position bias

|   Grid slot |   Impressions |   Clicks | CTR    |
|-------------|---------------|----------|--------|
|           1 |         1,093 |      302 | 27.63% |
|           2 |         1,082 |      215 | 19.87% |
|           3 |         1,078 |      147 | 13.64% |
|           4 |         1,044 |      127 | 12.16% |
|           5 |         1,023 |      106 | 10.36% |
|           6 |         1,013 |       85 | 8.39%  |
|           7 |           479 |       29 | 6.05%  |
|           8 |           479 |       24 | 5.01%  |

Slot 1 CTR / slot 8 CTR = 5.51x.
This is the exposure bias any honest offline evaluation has to
contend with: a product in slot 8 is not unpopular, it is unseen.

## 5. Surface performance

| Surface             |   Impressions |   Clicks | CTR    |
|---------------------|---------------|----------|--------|
| home_grid           |         2,024 |      216 | 10.67% |
| related_products    |         1,944 |      324 | 16.67% |
| category_page       |         1,880 |      242 | 12.87% |
| search_results      |           739 |      118 | 15.97% |
| recommended_for_you |           643 |      108 | 16.80% |
| wishlist_page       |            61 |       27 | 44.26% |

`recommended_for_you` here is the *legacy* popularity module the
site already had. Its CTR is the number the ML system has to beat
in Stage 7, and it is logged in recommendation_logs.

## 6. User activity (heavy tail)

| Percentile   |   Product events |
|--------------|------------------|
| min          |           0.0000 |
| p10          |           0.0000 |
| p25          |          41.0000 |
| p50          |          94.0000 |
| p75          |         177.0000 |
| p90          |         272.0000 |
| p99          |         701.0000 |
| max          |         838.0000 |
| mean         |         132.8000 |

Gini = 0.517. Top 10% of users generate 35.5% of all product events.
A flat distribution here would be the clearest sign of a fake dataset.

## 7. Product popularity concentration

|   # | Product   | Name           |   Views |
|-----|-----------|----------------|---------|
|   1 | T285      | Dawn Black     |      35 |
|   2 | T105      | Quarry Red     |      25 |
|   3 | T154      | Ash Rust       |      24 |
|   4 | T100      | Stone Charcoal |      23 |
|   5 | T150      | Oxide Sand     |      22 |
|   6 | T157      | Plume Ice Blue |      20 |
|   7 | T135      | Trace Beige    |      19 |
|   8 | T064      | Bevel Sky Blue |      18 |

Gini = 0.542. Top 10 of 300 products take 17.0% of views. 50 products have zero views (cold items).

## 8. User-item matrix sparsity

| Metric                                           | Value   |
|--------------------------------------------------|---------|
| Observed (user, product) pairs — any interaction | 4,384   |
| Observed (user, product) pairs — views only      | 878     |
| Matrix cells                                     | 24,000  |
| Density (any interaction)                        | 18.267% |
| Density (views only)                             | 3.658%  |
| Sparsity (views only)                            | 96.342% |
| Distinct products viewed / user (median)         | 10      |

Production recommenders typically work at 99%+ sparsity. This is why
the catalogue is extended past the 50 curated products: at 50 items the
matrix would be ~50% dense and none of the hard problems would appear.

## 9. Repeat viewing

| Times viewed   |   Pairs | Share   |
|----------------|---------|---------|
| 1 view(s)      |     597 | 68.00%  |
| 2 view(s)      |     199 | 22.67%  |
| 3 view(s)      |      55 | 6.26%   |
| 4 view(s)      |      19 | 2.16%   |
| 5 view(s)      |       2 | 0.23%   |
| 6 view(s)      |       3 | 0.34%   |
| 7 view(s)      |       2 | 0.23%   |
| 8 view(s)      |       1 | 0.11%   |

281 pairs (32.0%) were viewed more than once.
Repeat viewing is the behavioural signal Phase 4 singles out, and it only
exists because the simulator models comparison shopping inside a session
and deep-linking back across sessions.

## 10. Dwell time distribution

| Statistic          | Value   |
|--------------------|---------|
| count              | 1,286   |
| mean               | 62.9s   |
| median             | 50.1s   |
| p90                | 121.2s  |
| p99                | 244.2s  |
| max                | 408.0s  |
| < 5s (bounces)     | 4.0%    |
| > 60s (deep reads) | 40.4%   |

## 11. Session shape

| Metric                           | Value   |
|----------------------------------|---------|
| Median duration                  | 164s    |
| Mean duration                    | 244s    |
| Median products viewed           | 3       |
| Sessions with a wishlist add     | 24.7%   |
| Sessions with an add-to-cart     | 13.1%   |
| Sessions with a purchase         | 6.1%    |
| Single-product sessions (bounce) | 27.1%   |

Session intent mix: casual_browse 46.6%, focused_research 24.1%, quick_bounce 19.2%, purchase_intent 10.1%
Device mix: desktop 42.7%, mobile 42.4%, tablet 14.9%

## 12. Temporal structure and seasonality

Weekly event volume: min 228 / median 2,435 / max 3,162 (peak week of 2026-08-16).
Days more than 2 sd above the mean: 1 (2026-08-16).

Category share of product views by month (seasonality):

| Month   | Athletic   | Boot   | Bundle   | Loafer   | Modular Sole   | Modular Upper   | Sandal   | Sneaker   |
|---------|------------|--------|----------|----------|----------------|-----------------|----------|-----------|
| 2026-07 | 27.9%      | 4.4%   | 0.0%     | 2.5%     | 0.0%           | 0.0%            | 7.4%     | 57.8%     |
| 2026-08 | 21.3%      | 7.7%   | 0.1%     | 6.9%     | 0.1%           | 0.1%            | 12.8%    | 50.9%     |

Sandal share should climb into summer and boot share should fall.
If those two lines are flat, the seasonality code is not firing.

## 13. Cold-start population

| Cohort               |   Users | Share   |
|----------------------|---------|---------|
| <= 0 product events  |       9 | 11.2%   |
| <= 1 product events  |       9 | 11.2%   |
| <= 3 product events  |       9 | 11.2%   |
| <= 5 product events  |       9 | 11.2%   |
| <= 10 product events |      10 | 12.5%   |

19 users signed up in the final 30 days — genuine cold-start cases.

## 14. Legacy recommender baseline

The pre-existing on-site module served 643 impressions and earned 108 clicks — CTR 16.80%.
That is the online number Stage 8 has to improve on, and it comes from logged data rather than an assumption.

## 15. GROUND TRUTH — is behaviour predictive of preference?

|   Affinity decile |   Mean affinity | Any interaction   |   Mean score |
|-------------------|-----------------|-------------------|--------------|
|                 1 |          0.0260 | 1.00%             |       0.0100 |
|                 2 |          0.0680 | 0.58%             |       0.0100 |
|                 3 |          0.1200 | 1.25%             |       0.0100 |
|                 4 |          0.1970 | 1.38%             |       0.0200 |
|                 5 |          0.3140 | 1.88%             |       0.0200 |
|                 6 |          0.4860 | 3.04%             |       0.0500 |
|                 7 |          0.6980 | 4.75%             |       0.1100 |
|                 8 |          0.8700 | 7.96%             |       0.2500 |
|                 9 |          0.9550 | 10.46%            |       0.4300 |
|                10 |          0.9890 | 15.96%            |       0.7300 |

Spearman(hidden affinity, observed engagement) = 0.206.

Mean dwell time by hidden-affinity quintile: Q1 28.5s, Q2 62.3s, Q3 69.3s, Q4 69.5s, Q5 73.4s

Both relationships must be strongly monotonic: that is the proof that
behaviour encodes preference and therefore that a model can recover it.
The relationship must also be well short of perfect — the gap is the
noise, exposure bias and exploration that make the task non-trivial.

## 16. DIFFICULTY — how much headroom is there?

Spearman(observed popularity rank, mean hidden appeal rank) = 0.464.

This is the dataset's difficulty dial. Near 1.0 and popularity is
already optimal, so beating it would prove nothing. Near 0.0 and
popularity is noise, which is equally unrealistic. A strong but
imperfect correlation is what leaves genuine room for
personalisation to add measurable value.

## 17. GROUND TRUTH — archetype separability

| Archetype      | Athletic   | Boot   | Bundle   | Loafer   | Modular Sole   | Modular Upper   | Sandal   | Sneaker   |
|----------------|------------|--------|----------|----------|----------------|-----------------|----------|-----------|
| athlete        | 88%        | 1%     | 0%       | 1%       | 0%             | 0%              | 3%       | 8%        |
| heritage       | 5%         | 55%    | 0%       | 6%       | 0%             | 0%              | 6%       | 27%       |
| minimalist     | 0%         | 1%     | 0%       | 3%       | 0%             | 0%              | 6%       | 89%       |
| outdoors       | 50%        | 39%    | 0%       | 0%       | 0%             | 0%              | 0%       | 11%       |
| refined        | 0%         | 3%     | 0%       | 71%      | 0%             | 0%              | 0%       | 27%       |
| streetwear     | 15%        | 11%    | 0%       | 0%       | 1%             | 0%              | 2%       | 71%       |
| summer         | 0%         | 0%     | 0%       | 5%       | 0%             | 0%              | 58%      | 37%       |
| value          | 1%         | 1%     | 0%       | 0%       | 0%             | 0%              | 43%      | 55%       |
| window_shopper | 3%         | 3%     | 0%       | 0%       | 0%             | 0%              | 10%      | 83%       |

Each archetype should concentrate on a different category. This is the
cluster structure collaborative filtering exploits — if these rows were
identical, user-based CF could not work at all.
