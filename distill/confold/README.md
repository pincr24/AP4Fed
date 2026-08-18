# Vendored CON-FOLD runtime

This directory is a frozen third-party dependency from
[CON-FOLD](https://github.com/lachlanmcg123/CONFOLD), pinned in
`VENDORED_AT.txt`. It contains only the three modules needed to create and use
`foldrm.Classifier` on AP4Fed-generated CSVs. The upstream MIT licence is in
`LICENSE`.

Install the small runtime dependency set from `../requirements-confold.txt`.
Keep AP4Fed integration code outside this directory; use
`../confold_adapter.py` so upgrades remain reviewable as a vendor replacement.
