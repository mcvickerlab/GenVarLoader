# API Reference

## Writing

```{eval-rst}
.. currentmodule:: genvarloader

.. autofunction:: write

.. autofunction:: update

.. autofunction:: get_splice_bed

.. autofunction:: read_bedlike

.. autofunction:: with_length

.. autoclass:: BigWigs
    :members: __init__, from_table
    :exclude-members: __new__

.. autoclass:: genvarloader.Table
    :members:
    :exclude-members: __new__
```

## Insertion fill

Strategies controlling how re-aligned track values are filled across inserted bases (indels). Pass an instance to [`gvl.Dataset.with_insertion_fill()`](#genvarloader.Dataset.with_insertion_fill). `InsertionFill` is the abstract base; instantiate one of the concrete strategies.

```{eval-rst}
.. currentmodule:: genvarloader

.. autoclass:: InsertionFill
    :members:

.. autoclass:: Constant
    :members:

.. autoclass:: FlankSample
    :members:

.. autoclass:: Interpolate
    :members:

.. autoclass:: Repeat5p
    :members:

.. autoclass:: Repeat5pNormalized
    :members:
```

## Dataset maintenance

Utilities for upgrading on-disk datasets written by older GVL versions.

```{eval-rst}
.. currentmodule:: genvarloader

.. autofunction:: migrate

.. autofunction:: migrate_svar_link
```

## Reading

### Personalized data

```{eval-rst}
.. currentmodule:: genvarloader

.. autoclass:: Dataset
    :members:
    :exclude-members: __init__

.. autofunction:: get_dummy_dataset

.. autoclass:: DummyVariant
    :members:

.. autoclass:: VarWindowOpt
    :members:

.. autoclass:: RaggedDataset
    :exclude-members: __new__, __init__

.. autoclass:: ArrayDataset
    :exclude-members: __new__, __init__
```

### Reference genome(s)

```{eval-rst}
.. currentmodule:: genvarloader

.. autoclass:: Reference
    :members:
    :exclude-members: __new__, __init__

.. autoclass:: RefDataset
    :members:
    :exclude-members: __new__
```

### Non-personal/site-only variants

```{eval-rst}
.. currentmodule:: genvarloader

.. autoclass:: DatasetWithSites
    :members:
    :exclude-members: __new__

.. autofunction:: sites_vcf_to_table

.. autodata:: SitesSchema
```

## Data registry

```{eval-rst}
.. autofunction:: genvarloader.data_registry.fetch
```

## Containers

Classes that GVL Datasets may return.

```{eval-rst}
.. currentmodule:: genvarloader

.. autoclass:: AnnotatedHaps
    :members:
    :exclude-members: __init__

.. autoclass:: Ragged
    :members:
    :exclude-members: __init__

.. autoclass:: RaggedAnnotatedHaps
    :members:
    :exclude-members: __init__

.. autoclass:: RaggedVariants
    :members:
    :exclude-members: __init__

.. autoclass:: RaggedIntervals
    :members:
    :exclude-members: __init__
```

### Flat containers

Returned in place of the ragged containers when a Dataset uses [`with_output_format("flat")`](#genvarloader.Dataset.with_output_format). Each carries flat `data`/`offsets` buffers and a `to_ragged()` escape hatch back to the ragged form.

```{eval-rst}
.. currentmodule:: genvarloader

.. autoclass:: FlatRagged
    :members:
    :exclude-members: __init__

.. autoclass:: FlatAnnotatedHaps
    :members:
    :exclude-members: __init__

.. autoclass:: FlatIntervals
    :members:
    :exclude-members: __init__

.. autoclass:: FlatVariants
    :members:
    :exclude-members: __init__

.. autoclass:: FlatAlleles
    :members:
    :exclude-members: __init__

.. autoclass:: FlatVariantWindows
    :members:
    :exclude-members: __init__
```

### PyTorch interop

```{eval-rst}
.. currentmodule:: genvarloader

.. autofunction:: to_nested_tensor
```