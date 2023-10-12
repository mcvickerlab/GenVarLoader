# GenVarLoader
GenVarLoader aims to enable training sequence models on 10's to 100's of thousands of individuals' personalized genomes.

## Installation
`pip install genvarloader`

A PyTorch dependency is not included since it requires [special instructions](https://pytorch.org/get-started/locally/).

## Quick Start
```python
import genvarloader as gvl

reference = 'reference.fasta'
variants = 'variants.pgen' # highly recommended to convert VCFs to PGEN
regions_of_interest = 'regions.bed'
```
Create readers for each file providing sequence data:
```python
ref = gvl.Fasta(name='ref', path=reference, pad='N')
var = gvl.Pgen(variants)
varseq = gvl.FastaVariants(name='varseq', fasta=ref, variants=var)
```
Put them together and get a `torch.DataLoader`:
```python
gvloader = gvl.GVL(
    readers=varseq,
    bed=regions_of_interest,
    fixed_length=1000,
    batch_size=16,
    max_memory_gb=8,
    batch_dims=['sample', 'ploid'],
    shuffle=True,
    num_workers=2
)

dataloader = gvloader.torch_dataloader()
```
And now you're ready to use the `dataloader` however you need to:
```python
# implement your training loop
for batch in dataloader:
    ...
```