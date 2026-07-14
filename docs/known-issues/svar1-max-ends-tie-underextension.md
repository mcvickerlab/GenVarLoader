# SVAR1 `_write_from_svar` max_ends under-extends at same-POS ties

**Status:** open (latent bug in the SVAR1 write path; not yet fixed)
**Found:** 2026-07-05, during SVAR2 read-bound dataset wiring.

## Symptom
`_write_from_svar` (`python/genvarloader/_dataset/_write.py`) computes each region's
`max_ends` as the end of the **highest-store-order variant** (`max v_idx`) overlapping
the region, then `chromEnd = max(max_ends, chromEnd)`. When two variant records share a
POS but have different ends — e.g. a SNP (`end = POS`) and a deletion (`end = POS - ILEN`)
at the same position — and the SNP has the higher store-order index, SVAR1 selects the
SNP's (shorter) end and **under-extends** the region. The deletion's coverage past the
region boundary is then truncated.

Example: region `[11,13)`, `chr1:12 G>A` (SNP) and `chr1:12 GTA>G` (DEL, ILEN -2, end 14),
DEL indexed before SNP. SVAR1 → `chromEnd=13` (under-extended); the correct value is `14`.

The code comment at the `max_ends` computation already hedges: "this is fine if there
aren't any overlapping variants that could make a v_idx < -1 have a further end."

## SVAR2 behavior (correct)
The SVAR2 read-bound write path (`_write_from_svar2`) computes `max_ends` as the true end
of the max-position variant (`pos - min(ilen,0)`), yielding the **correct** `chromEnd=14`.

## Parity policy
SVAR1↔SVAR2 `chromEnd` parity is byte-identical **except** same-POS multi-record tie
regions, where SVAR1 is the buggy oracle. Such regions are excluded from strict parity
(see the SVAR2 wiring plan's Task 7). Fixing SVAR1's `max_ends` is deferred (it would
change SVAR1 output for tie cases, i.e. it is not an additive change).
