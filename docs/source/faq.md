# FAQ

## Why does a Dataset return "Ragged" objects and what are they?

For why, see ["What's a `gvl.Dataset`?"](dataset.md). [`Ragged`](api.md#genvarloader.Ragged) arrays are similar to NumPy arrays except that the final axis is a variable size. For example, a 2D ragged array might look like:

:::{image} _static/ragged.svg
:alt: A 2D ragged array with 3 rows.
:align: center
:width: 150
:::

To store this, a [`Ragged`](api.md#genvarloader.Ragged) array minimally consists of two NumPy arrays: a 1D array `data` with shape `(size)` containing the values, and another 1D array `offsets` with shape `(n_rows+1)` specifying the start and end position (exclusive) of every row's data in the `data` array. We could thus create the above example:

```python
data = np.array([1, 2, 3, 4, 5, 6])
offsets = np.array([0, 2, 3, 6])
shape = (3,)
ragged = gvl.Ragged.from_offsets(data, shape, offsets)
# [
#     [1, 2],
#     [3],
#     [4, 5, 6]
# ]
```

You can then work with the ragged data as-is or convert to them [to](api.md#genvarloader.Ragged.to_awkward) and [from](api.md#genvarloader.Ragged.from_awkward) Awkward Arrays. Depending on what you need to do, either representation may be more convenient. Within GVL, we use numba JIT'd functions to compute on the ragged objects directly since it's relatively straightforward.

## How can I get multiple tracks/stranded data?

If you provide multiple tracks to [`gvl.write()`](api.md#genvarloader.write), all of them can be returned simultaneously from the resulting [`Dataset`](api.md#genvarloader.Dataset) and placed along the track axis, sorted by name. By default, a Dataset sets all tracks to active when opened. i.e. tracks have shape `(batch, tracks, [ploidy], length)`.

## How can I get personalized protein/RNA sequences?

This is not yet supported but on GVL's roadmap for the near future. Keep an eye out in future releases!

<!-- Example of variable length regions

Example of spliced gvl.write() and enabling splicing

Example of SeqPro translate for RNA and AA -->

## Why aren't the methods `with_len()`, `with_seqs()`, etc. combined into `with_settings()`?

These methods modify the type of output returned by a `gvl.Dataset`. In order to allow type checkers like mypy and pyright to know how these settings modify state, they are given their own methods. As a result, if you use a type checker, you will have access to an improved developer workflow with compile-time errors for many common issues. For example, using an incompatible transform or unpacking return values into the wrong number of arguments.