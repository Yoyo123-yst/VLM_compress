# Change log

- 2026-10-08: Froze `DEV_PROTOCOL_ADDENDUM.json` before any B/C training. Parent protocol file was not overwritten.
- Variant A audit was corrected to use uint8 pixel L∞ and to replay only after selection.
- 2026-10-08: Stage I passed. Stage II stopped with `BLOCKED_BC_IMPLEMENTATION_PREREQUISITE` because the Full hinge gradient is 0 at delta 0 and after the one +1/255 fallback. No B/C training and no pristine60.
