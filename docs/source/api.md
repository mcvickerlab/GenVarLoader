# API Reference

## Writing

```{eval-rst}
.. currentmodule:: genvarloader

.. autofunction:: write

.. autofunction:: read_bedlike

.. autofunction:: with_length

.. autoclass:: BigWigs
    :members:
    :exclude-members: rev_strand_fn, chunked
```

## Reading

```{eval-rst}
.. currentmodule:: genvarloader

.. autoclass:: Dataset
    :members:
    :exclude-members: __init__

.. autofunction:: get_dummy_dataset

.. autoclass:: Reference
    :members:
    :exclude-members: __new__, __init__

.. autoclass:: RaggedDataset
    :exclude-members: __new__, __init__

.. autoclass:: ArrayDataset
    :exclude-members: __new__, __init__

.. autofunction:: sites_vcf_to_table

.. autodata:: SitesSchema

.. autoclass:: DatasetWithSites
    :exclude-members: __new__, __init__
```

## Containers

Classes that GVL Datasets may return.

```{eval-rst}

.. autoclass:: genvarloader._types.AnnotatedHaps
    :members:
    :exclude-members: __init__

.. autoclass:: genvarloader.Ragged
    :members:
    :exclude-members: __init__

.. autoclass:: genvarloader._ragged.RaggedAnnotatedHaps
    :members:
    :exclude-members: __init__

.. autoclass:: genvarloader._dataset._rag_variants.RaggedVariants
    :members:
    :exclude-members: __init__
```