# Reject symbolic & breakend variants in `gvl.write`

**Date:** 2026-06-07
**Status:** Approved (design)

## Problem

`gvl.write` reconstructs haplotypes by expanding each variant's ALT allele into
literal nucleotide sequence. Two classes of VCF 4.x ALT alleles cannot be
expanded this way:

- **Symbolic alleles** — `<DEL>`, `<INS>`, `<DUP>`, `<INV>`, `<CNV>`, `<*>`, … (ALT starts with `<`).
- **Breakends** — mate-pair / single-breakend notation, e.g. `G[chr2:321[`, `]chr2:321]G`, `.TGCA`, `TGCA.`.

Today `gvl.write` applies **no** variant filter when it constructs a reader from
a path, and trusts whatever filter the user set on a pre-built genoray reader.
If symbolic or breakend records reach the writer, haplotype reconstruction is
undefined. The only existing input guard is the **multi-allelic** check in
`_write_from_vcf` / `_write_from_pgen`, which raises and tells the user to split.

Unlike multi-allelics (which can be split to preserve information), symbolic and
breakend alleles carry no expandable sequence — there is nothing to recover. The
correct behavior is to **reject the input**, not silently drop records.

## Decision summary

- **Reject, do not filter.** Inspect the variant index that will actually drive
  writing (post any user-supplied filter) and raise a `ValueError` if it
  contains any symbolic or breakend ALT. We do **not** apply an on-the-fly
  filter — that is a future QoL improvement, explicitly out of scope here.
- **Uniform across VCF, PGEN, and SVAR.** All three expose an index with an
  `ALT` column, so one validator covers all three. SVAR (already materialized)
  can only be validated, which is why a check — not a filter — is the right
  primitive for every source.
- **Consolidate** the symbolic/breakend check together with the existing
  multi-allelic check into a single "unsupported variant" validator, used by all
  three write paths. This also adds the missing multi-allelic guard to the SVAR
  path.
- **Full-index check.** The validator runs over the entire index (matching the
  existing multi-allelic guard and the "contain any" requirement), not only
  variants overlapping query regions.

## Dependency bump

genoray 2.9.1 introduces `genoray.exprs.is_symbolic` and
`genoray.exprs.is_breakend` (absent in the pinned 2.7.3). Both are polars
expressions over the `.gvi` index `ALT` column.

- `pyproject.toml`: `"genoray>=2.7.3,<3"` → `"genoray>=2.9.1,<3"`
- `pixi.toml`: `genoray = "==2.7.3"` → `genoray = "==2.9.1"`, then relock
  (`pixi run -e dev` / regenerate `pixi.lock`; respect the `merge=ours`
  lockfile convention).

## Design

### Consolidated validator

A single helper in `python/genvarloader/_dataset/_write.py`:

```python
def _reject_unsupported_variants(index: pl.DataFrame, source: str) -> None:
    """Raise if the variant index contains alleles gvl cannot reconstruct.

    Checks multi-allelic, symbolic, and breakend ALTs. Runs over the full
    (post-filter) index. ``source`` names the input for the error message
    (e.g. "VCF", "PGEN", "SVAR").
    """
    counts = index.select(
        n_multi=(pl.col("ALT").list.len() > 1).sum(),
        n_symbolic=genoray.exprs.is_symbolic.sum(),
        n_breakend=genoray.exprs.is_breakend.sum(),
    ).row(0)
    n_multi, n_sym, n_bnd = counts
    if n_multi or n_sym or n_bnd:
        raise ValueError(
            f"{source} contains unsupported variants: {n_multi} multi-allelic, "
            f"{n_sym} symbolic (<DEL>/<INS>/...), {n_bnd} breakend. gvl can only "
            f"reconstruct bi-allelic, non-symbolic, non-breakend variants. Remove "
            f"them upstream (bcftools/plink2 — split multi-allelics, drop SVs) or "
            f"construct the genoray reader with a filter such as "
            f"`filter=genoray.exprs.is_biallelic & ~genoray.exprs.is_symbolic & "
            f"~genoray.exprs.is_breakend`."
        )
```

(Exact column-expression form to be confirmed against genoray 2.9.1 during
implementation; `is_symbolic`/`is_breakend` only require the `ALT` column, which
VCF `_load_index`, PGEN `_index`, and `svar.index` all carry.)

### Call sites

- `_write_from_vcf`: replace the current multi-allelic `raise` with a call to
  `_reject_unsupported_variants(vcf._index, "VCF")`.
- `_write_from_pgen`: replace the current `_sei is None` multi-allelic `raise`
  with `_reject_unsupported_variants(pgen._index, "PGEN")`.
  **Verify** that the index-based multi-allelic check subsumes the `_sei is None`
  guard; if `_sei is None` signals a *distinct* genoray failure mode, keep it as
  a secondary assertion rather than dropping it.
- `_write_from_svar`: add `_reject_unsupported_variants(svar.index, "SVAR")` near
  the top (this path currently has no such guard).

### What we explicitly do NOT do

- No on-the-fly filtering / dropping of records.
- No mutation of a user-supplied genoray reader's filter.
- No VCF filter-callable plumbing (only needed when *applying* a filter; we only
  *read* the index).

## Testing

Mirror the existing rejection test `test_multiallelic_raw_is_rejected` in
`tests/integration/dataset/test_haps_property.py`:

- Extend the `_ALL_VIOLATIONS` set / `_spec_and_violating_doc` harness with
  `"symbolic"` and `"breakend"` labels **if** `vcfixture`'s `VcfBuilder` can emit
  such ALTs. The implementation plan must first verify vcfixture's capability.
- **Fallback (if vcfixture cannot emit symbolic/breakend ALTs):** hand-write a
  minimal VCF containing one `<DEL>` (symbolic) and one breakend ALT, and assert
  `gvl.write(..., variants=<vcf>)` raises `ValueError` matching the count
  message. Derive a PGEN (plink2) and a `.svar` (`SparseVar.from_vcf` with **no**
  filter) from the same source and assert both are rejected too.
- Assert the error message reports nonzero counts for the symbolic and breakend
  classes.

## Skill update

This changes `gvl.write`'s input contract (preprocessing requirements). Per
`CLAUDE.md`, update `skills/genvarloader/SKILL.md`:

- Add symbolic/breakend rejection to the preprocessing requirements and the
  "Common gotchas" section.
- Note the recommended genoray filter / bcftools/plink2 preprocessing to remove
  them.

## Out of scope (future QoL)

- Automatically filtering symbolic/breakend variants on-the-fly so users don't
  have to preprocess. Tracked separately; not part of this change.
```
