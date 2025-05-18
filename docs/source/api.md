# API Reference

## Writing

```{eval-rst}
.. currentmodule:: genvarloader

.. autofunction:: write

.. autofunction:: read_bedlike

.. autofunction:: with_length

.. autoclass:: BigWigs
    :members: __init__, from_table
    :exclude-members: __new__
```

## Reading

### Personalized data

```{eval-rst}
.. currentmodule:: genvarloader

.. autoclass:: Dataset
    :members:
    :exclude-members: __init__

.. autofunction:: get_dummy_dataset

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