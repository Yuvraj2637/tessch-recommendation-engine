# Phase 3 — item-item similarity results

Generated 2026-08-26T02:07:27+00:00 · cosine backend: `sklearn`

## Data

The model was trained on 1,286 product views across a catalogue of 300 products. Of those views, 1,286 are simulated and 0 came from real browsing (0.0% of the total).

The resulting user-item matrix is 69 users by 300 products with 877 filled cells — a density of 4.24%, which is normal for behavioural data and is the reason cosine similarity is computed over columns rather than anything requiring dense overlap. Row normalisation: `l2`.

250 of 300 products have at least one learned neighbour. The remaining 50 were never viewed, so they have no column to compare and fall back to popularity — the cold-start limitation, stated as a number rather than a caveat.

## Does the model produce sensible neighbours?

The model never sees a product's category, style or colour — it is trained purely on who viewed what, for how long. That makes those attributes an independent check: if behaviourally-similar products share a category far more often than chance predicts, the co-view signal carries real information.

Chance is the important part. With 8 categories and an uneven catalogue, two random products already share a category surprisingly often, so each rate is reported against the random-pair baseline for the same attribute.

| attribute | top-1 match | top-k match | random baseline | lift |
| --- | --- | --- | --- | --- |
| category | 76.4% | 72.6% | 26.1% | 2.78x |
| style | 42.8% | 44.1% | 18.7% | 2.36x |
| color_family | 21.2% | 25.0% | 20.0% | 1.25x |

The colour row is the one that validates the rest. There is no reason co-viewing should track colour, and it doesn't — colour agreement sits essentially at chance. A model scoring highly on *every* attribute would be evidence of a bug, most likely neighbour lists dominated by a few globally popular products rather than genuine per-item structure.

Concentration confirms that: 248 distinct products appear across all neighbour lists, and the ten most-recommended occupy only 7.3% of the slots. Recommendations are spread across the catalogue rather than collapsing onto bestsellers.

The similarity distribution is the expected shape for a healthy matrix: 85% of product pairs score zero, the median is 0.000, and there is a thin strong tail (p99 = 0.617, max = 1.000). A high mean would indicate the geometry had collapsed — the signature of a missing normalisation step.

## What normalisation actually buys

Attention is very unevenly distributed. Across 70 users the activity Gini is 0.51: the heaviest browser (`U00079`) accounts for 8.9% of all recorded dwell time, 10x the median user. Without per-user row scaling that one person's taste is a large fraction of every similarity score in the model.

The obvious way to justify normalisation would be to show it improves category coherence. On this dataset it does not — and reporting that is more useful than quietly picking the metric that agrees. The unnormalised matrix scores *higher* on category coherence, because the heaviest user browses consistently within a few categories, so letting them dominate reinforces category agreement while making the model mostly about one person.

The right question is whether the recommendations belong to the population or to an individual. Removing the single heaviest user and refitting answers it directly:

| row scaling | category top-k | lift | top-k stability | top-1 unchanged |
| --- | --- | --- | --- | --- |
| `none` | 77.0% | 2.95x | 0.819 | 84.8% |
| `l2` | 72.6% | 2.78x | 0.940 | 92.2% |
| `user_share` | 69.3% | 2.66x | 0.976 | 96.3% |

Stability is Jaccard overlap of each product's neighbour list before versus after that user is removed, so 1.000 would mean their departure changed nothing.

L2 was chosen on that trade: it gives up 4.4 points of category coherence to gain 12.1 points of stability, cutting the share of neighbour slots that move when one user leaves from 18.1% to 6.0%. `user_share` is more stable still but costs noticeably more coherence, which reads as over-flattening genuine differences in engagement.

Cosine already normalises each product column to unit length, so the column side needs nothing further. Row scaling is the part that has to be a decision.

## Worked examples

The most-viewed products with neighbours, so these lists rest on the most evidence available rather than being chosen for how good they look.

**T014 · Tangent Rasputeen** — category Sneaker

- `0.542` T001 Arc Tangelo · style Casual · same category
- `0.542` T293 Summit Black · style Smart Casual · same category
- `0.542` T207 Rift White · style Casual · same category
- `0.542` T234 Beacon Navy · style Smart Casual · same category

**T015 · Tangent Biskoff** — category Sneaker

- `0.702` T087 Sable White · style Casual · category Sandal — differs
- `0.358` T238 Plume Charcoal · style Casual · same category
- `0.346` T078 Gale Rust · style Streetwear · same category
- `0.275` T249 Basalt Beige · style Casual · same category

**T002 · Arc Sakura** — category Sneaker

- `0.716` T250 Meridian Teal Sneaker · style Casual · same category
- `0.633` T196 Trellis Sand · style Smart Casual · same category
- `0.478` T165 Alder Off-White · style Casual · same category
- `0.453` T159 Slate Cream Sneaker · style Streetwear · same category

**T007 · Carve Brenjam** — category Sneaker

- `0.845` T212 Delta White Sneaker · style Smart Casual · same category
- `0.623` T001 Arc Tangelo · style Casual · same category
- `0.623` T207 Rift White · style Casual · same category
- `0.623` T234 Beacon Navy · style Smart Casual · same category

**T009 · Carve Wasabi** — category Sneaker

- `0.750` T127 Vantage Grey · style Streetwear · same category
- `0.675` T260 Crag Sky Blue · style Smart Casual · same category
- `0.630` T027 Warden Sand · style Casual · same category
- `0.619` T003 Arc Omnitrix · style Streetwear · same category

**T010 · Carve Volcano** — category Sneaker

- `0.976` T300 Vessel Black · style Heritage · category Boot — differs
- `0.917` T147 Stark Black · style Heritage · category Boot — differs
- `0.869` T107 Nimbus Rust · style Casual · same category
- `0.842` T181 Flare Beige · style Smart Casual · category Loafer — differs

## Reproducing this

```
python -m scripts.build_similarity --report --compare-normalisation
```
