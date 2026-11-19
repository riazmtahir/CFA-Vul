# Identical vulnerable/patched pairs on DiverseVul (descriptive, A4)

Pair = same commit + function name, opposite labels, clean 1:1 group. `corpus` = all such pairs in the corpus; arm rows = test pairs of that split (paired_eval.py). raw = same string; ws = same after removing whitespace (Li et al. 2026); norm = same `normalize_tokens` stream; any = any of the three.

| scope | pairs | raw | ws | norm | any |
|---|---|---|---|---|---|
| corpus | 14,324 | 0 (0.0%) | 388 (2.71%) | 1,125 (7.85%) | 1,125 (7.85%) |
| random_control | 144 | 0 (0.0%) | 2 (1.39%) | 8 (5.56%) | 8 (5.56%) |
| family_disjoint | 688 | 0 (0.0%) | 51 (7.41%) | 138 (20.06%) | 138 (20.06%) |
| project_disjoint | 1,237 | 0 (0.0%) | 201 (16.25%) | 248 (20.05%) | 248 (20.05%) |
| commit_disjoint | 1,515 | 0 (0.0%) | 32 (2.11%) | 121 (7.99%) | 121 (7.99%) |
