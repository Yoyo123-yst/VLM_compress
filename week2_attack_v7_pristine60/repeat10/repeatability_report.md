# repeat10 endpoint reproduction

- Strict label agreement: 8/10
- Diagnostic: REPEATABILITY_DIAGNOSTIC_PASS
- Full text agreement: 10/10
- Selected step agreement: 7/10
- FastV text agreement: 2/10
- PNG hash agreement: 0/10
- Keep hash agreement: 0/10
- Both strict 7; first only 0; second only 2; neither 1

A different selected step is not by itself a failure. EXACT means the selected image, Full text, FastV text, keep hash, and K match. ENDPOINT means both runs are strict, Full-exact, and replay-valid even if the image differs.

- 56808: ENDPOINT_REPRODUCTION; main strict 1 step 200; repeat strict 1 step 200; FastV `2` vs `1`
- 45827: ENDPOINT_REPRODUCTION; main strict 1 step 150; repeat strict 1 step 200; FastV `Endometrial cancers` vs `endometriosis`
- 5586: ENDPOINT_REPRODUCTION; main strict 1 step 200; repeat strict 1 step 200; FastV `Daniel K. Jahanan, Jahanan` vs `Daniel K. Jahan, Johan, Alaka`
- 51660: ENDPOINT_REPRODUCTION; main strict 1 step 200; repeat strict 1 step 200; FastV `11/75` vs `1975, 1975, 1975`
- 8072: ENDPOINT_REPRODUCTION; main strict 1 step 200; repeat strict 1 step 200; FastV `2` vs `5099.B1G1F`
- 55463: ENDPOINT_MISMATCH; main strict 0 step 200; repeat strict 1 step 200; FastV `Cigarettes cause cancer, heart disease, monoxide` vs `Smoking causes cancer, heart disease, lung disease, and emphysema`
- 1170: ENDPOINT_REPRODUCTION; main strict 1 step 200; repeat strict 1 step 200; FastV `confidential` vs `confidential`
- 49530: ENDPOINT_MISMATCH; main strict 0 step 100; repeat strict 1 step 150; FastV `Figure 12` vs `26,00.0`
- 64689: ENDPOINT_REPRODUCTION; main strict 1 step 200; repeat strict 1 step 200; FastV `122` vs `102`
- 57221: ENDPOINT_MISMATCH; main strict 0 step 150; repeat strict 0 step 200; FastV `2021-03` vs `2021-03`
