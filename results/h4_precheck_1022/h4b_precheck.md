# H4b pre-check (commit_decon / commit_sizectrl vs commit_disjoint)

commit_disjoint train 264,775; removed 35,192 (13.29%), of which 1,646 positives; positives left 13,499; borrowed 0 (0.00% of the removal)

train positive rate: commit_disjoint 0.0572, decon 0.0588, sizectrl 0.0588

| set | n | mean chars | median chars | pos rate | top-10 projects (% of set) | top-10 CWEs (% of set) |
|---|---|---|---|---|---|---|
| removed_by_decon | 35,192 | 1256 | 182 | 0.0468 | linux 27.49, gpac 6.75, linux-2.6 5.52, openssl 3.72, server 3.44, qemu 2.56, ImageMagick 1.91, php-src 1.85, net 1.41, vim 1.4 | (none) 16.29, CWE-787 11.03, CWE-125 7.72, CWE-703 7.65, CWE-416 7.1, CWE-119 7.02, CWE-20 6.71, CWE-476 5.69, CWE-400 4.4, CWE-401 3.9 |
| removed_by_control | 35,192 | 1038 | 186 | 0.0468 | linux 17.32, server 3.85, envoy 3.38, linux-2.6 2.87, php-src 2.6, mongo 2.17, vim 2.16, cimg 2.09, qemu 2.03, ceph 1.83 | (none) 14.21, CWE-787 10.62, CWE-125 8.41, CWE-703 7.91, CWE-119 7.59, CWE-416 7.13, CWE-20 7.0, CWE-476 5.11, CWE-200 3.59, CWE-400 3.41 |
| kept | 229,583 | 1336 | 574 | 0.0588 | linux 20.69, php-src 3.05, linux-2.6 2.73, envoy 2.42, server 2.22, qemu 2.19, vim 1.94, mongo 1.53, openssl 1.53, gpac 1.44 | (none) 14.76, CWE-787 11.14, CWE-125 8.18, CWE-703 7.86, CWE-119 7.78, CWE-416 6.96, CWE-20 6.93, CWE-476 5.74, CWE-200 3.98, CWE-190 3.28 |

| split | direct leak % (test) | positives: direct leak % |
|---|---|---|
| published | 56.45 | 84.46 |
| random_control | 56.23 | 83.53 |
| family_disjoint | 44.87 | 59.7 |
| project_disjoint | 16.53 | 6.87 |
| commit_disjoint | 37.15 | 40.4 |

**PASS**
