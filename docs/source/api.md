# API Reference

## Writing

```{eval-rst}
.. currentmodule:: genvarloader

.. autofunction:: write

.. autofunction:: read_bedlike

.. autofunction:: with_length

.. autoclass:: Variants
    :members: from_file
    :exclude-members: __new__, __init__

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

.. autoclass:: Ragged
    :members:
    :exclude-members: __init__

.. autofunction:: get_dummy_dataset

.. autoclass:: RaggedDataset
    :exclude-members: __new__, __init__

.. autoclass:: ArrayDataset
    :exclude-members: __new__, __init__
```