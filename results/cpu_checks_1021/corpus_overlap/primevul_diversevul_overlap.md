# PrimeVul ∩ DiverseVul on our server copies (descriptive, A3)

PrimeVul 235,768 functions (test 25,911); DiverseVul 330,491. raw = exact text; norm = `normalize_tokens` stream; commit = (commit_id, function name).

| key | PrimeVul matched | % of PrimeVul | DiverseVul matched | % of DiverseVul | same label (PrimeVul matched) |
|---|---|---|---|---|---|
| raw | 136,716 | 57.99 | 136,716 | 41.37 | 99.85 |
| norm | 188,641 | 80.01 | 186,877 | 56.55 | 99.65 |
| commit | 138,912 | 58.92 | 140,130 | 42.4 | 99.69 |

PrimeVul test functions found in DiverseVul train (count, % of PrimeVul test):

| key | published | random_control | family_disjoint | project_disjoint | commit_disjoint |
|---|---|---|---|---|---|
| raw | 18,109 (69.89%) | 18,112 (69.9%) | 18,290 (70.59%) | 17,461 (67.39%) | 18,034 (69.6%) |
| norm | 19,280 (74.41%) | 19,329 (74.6%) | 18,583 (71.72%) | 18,029 (69.58%) | 18,960 (73.17%) |
| commit | 13,912 (53.69%) | 13,899 (53.64%) | 13,927 (53.75%) | 13,469 (51.98%) | 12,967 (50.04%) |
