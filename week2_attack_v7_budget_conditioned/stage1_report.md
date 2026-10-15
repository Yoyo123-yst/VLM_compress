# Stage I diagnosis

Decision: NO_GO_NO_COMPRESSION_CONDITIONAL_EFFECT

This is a mechanism diagnosis on the existing pristine60 Variant B images. It is not a generalization result.

- r=0.50: paired mean 2.439758, bootstrap [1.937394, 2.988352], strict 11/60, positive-among-strict 1.0, median strict 4.257863998413086, median nonstrict 1.6673564910888672
- r=0.35: paired mean 10.458477, bootstrap [8.818732, 12.183783], strict 48/60, positive-among-strict 1.0, median strict 9.32343578338623, median nonstrict 5.917478695511818
- r=0.25: paired mean 17.939648, bootstrap [15.763110, 20.242203], strict 52/60, positive-among-strict 1.0, median strict 17.729982495307922, median nonstrict 10.888028621673584
- r=0.15: paired mean 6.251852, bootstrap [5.319282, 7.253732], strict 31/60, positive-among-strict 1.0, median strict 5.511856891214848, median nonstrict 4.044065535068512

Gates: {'A_paired_delta_025': True, 'B_sixty_percent_positive': True, 'C_median_order': True, 'D_margin_text_consistent': False}
Margin/text conflicts: ['5615', '1023', '22317', '51140', '56084', '60744']
Samples whose Delta grows as the keep ratio falls: 0/60
