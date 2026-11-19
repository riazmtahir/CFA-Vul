# Direct high-recall clone edges, test -> train (no transitive closure)

Criterion: blind identifiers, 3-token shingles, exact Jaccard >= 0.7; LSH candidates (16 bands x 8 rows; P(candidate | J = 0.7) = 0.613), so the rates are lower bounds.

| split | test n | direct leak % | positives: direct leak % | train functions hit (% of train) | positives hit / train positives |
|---|---|---|---|---|---|
| published | 33,050 | 56.45 | 84.46 | 60,535 (22.9%) | 3,724 / 15,145 |
| random_control | 33,172 | 56.23 | 83.53 | 60,938 (23.02%) | 3,820 / 15,178 |
| family_disjoint | 33,172 | 44.87 | 59.7 | 38,630 (14.59%) | 2,359 / 15,145 |
| project_disjoint | 33,165 | 16.53 | 6.87 | 19,199 (7.25%) | 359 / 15,145 |
| commit_disjoint | 33,172 | 37.15 | 40.4 | 35,192 (13.29%) | 1,646 / 15,145 |
