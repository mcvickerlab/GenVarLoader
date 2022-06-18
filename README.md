# genome-loader
Pipeline for efficient genomic data processing.

&nbsp;
## Requirements
- Python >= 3.7
- h5py
- pysam
- numpy
- pandas
- typer

&nbsp;
## Installation
Recommended installation through conda, and given environment
```shell script
conda env create -f environment.yml
```

Then add the package to your path i.e. in your `.bashrc`:
```bash
export PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}/path/to/genome-loader
```

&nbsp;
## Table of Contents
- [HDF5 Writers](#hdf5-writers)
    - [writefasta](#writefasta)
    - [writefrag](#writefrag)
    - [writecoverage](#writecoverage)
- [Python Functions](#python-functions)
    - [encode_data.py](#encode_datapy)
    - [get_encoded.py](#get_encodedpy)
    - [get_data.py](#get_datapy)
    - [load_data.py](#load_datapy)
    - [load_h5.py](#load_h5py)

&nbsp;

---

&nbsp;
## **HDF5 Writers**
Command line tools for writing genome data to HDF5 format

&nbsp;
## writefasta
Converts Fasta file into char-array(default) or one-hot encoded HDF5 file.

&nbsp;
**File Format**
- Group: `[chrom]`
- Dataset: `"sequence"` if char array, `"onehot"` if one-hot encoded
- Attributes: `"id"`- dataset name associated with file

&nbsp;
**Usage**
```shell script
gloader writefasta [FASTA] --output/--directory [OUT] {OPTIONS}
```

**Required Arguments**
- FASTA: Positional argument, fasta file to write to hdf5
- -o/--output: Full path and file name of output (NOTE: Cannot use both -o and -d flags)
- -d/--directory: Directory to write hdf5 output

**One-Hot Encoding Arguments**
- -e/--encode: Flag that denotes output in one-hot encoding
- -s/--spec: Ordered string of non-repeating chars. Denotes encoded bases and order ie: "ACGT" (Default: "ACGTN")

**Optional Arguments**
- -c/--chroms: Chromosomes to write (Default: ALL)
- -n/--name: Output file if --directory given, ignored if using --output flag. Defaults to input fasta name

&nbsp;
## writefrag
Writes BAM ATAC fragment depth into HDF5 file.

&nbsp;
**File Format**
- Group: `[chrom]`
- Dataset: `"depth"` - 0-based array with depth per position
- Attributes:
    - `"id"` - dataset name associated with file
    - `"count_method"` - method used to count fragments


&nbsp;
**Usage**
```shell script
gloader writefrag [BAM] --output/--directory [OUT] {OPTIONS}
```

**Required Arguments**
- BAM: Positional argument, BAM file to parse and write to H5
- -o/--output: Full path and file name of output (NOTE: Cannot use both -o and -d flags)
- -d/--directory: Directory to write hdf5 output

**Optional Arguments**
- -c/--chroms: Chromosomes to write (Default: ALL)
- -l/--lens: Lengths of provided chroms (Auto retrieved if not provided)
- -n/--name: Output file if --directory given, ignored if using --output flag. Defaults to input fasta name
- --ignore_offset: Don't offset Tn5 cut sites (+4 bp on + strand, -5 bp on - strand, 0-based)
- --method: Method used to count fragment. Choice of `"cutsite"`|`"midpoint"`|`"fragment"` (Default: `"cutsite"`)
    - `cutsite`: Count both Tn5 cut sites
    - `midpoint`: Count the midpoint between Tn5 cut sites
    - `fragment`: Count all positions between Tn5 cut sites

&nbsp;
## writecoverage
Writes BAM allelic coverage into HDF5 file.

&nbsp;
**File Format**
- Group: `[chrom]`
- Dataset: `"coverage"` - 4 x N Matrix ordered A, C, G, T showing per allele coverage per position (0-based)
- Attributes: `"id"`- dataset name associated with file

&nbsp;
**Usage**
```shell script
gloader writecoverage [BAM] --output/--directory [OUT] {OPTIONS}
```

**Required Arguments**
- BAM: Positional argument, BAM file to parse and write to H5
- -o/--output: Full path and file name of output (NOTE: Cannot use both -o and -d flags)
- -d/--directory: Directory to write hdf5 output

**Optional Arguments**
- -c/--chroms: Chromosomes to write (Default: ALL)
- -n/--name: Output file if --directory given, ignored if using --output flag. Defaults to input fasta name

---

&nbsp;
## **Python Functions**
Python functions for directly loading and parsing genome data.

Specific argument level usage can be found as docstrings within scripts (Located in `src/`).

&nbsp;
## encode_data.py
Contains functions for creating one-hot encoded data.
- **encode_sequence**: Encodes input data into one-hot encoded format
- **encode_from_fasta**: Create one-hot encoded data directly from FASTA
- **encode_from_h5**: Create one-hot encoded data from char-array encoded H5

&nbsp;
## get_encoded.py
Contains functions for loading, and transforming one-hot encoded data.
- **get_encoded_haps**: Creates one-hot encoded haplotypes from one-hot encoded data

&nbsp;
## get_data.py
Functions that retrieves non-encoded data from files.
- **get_read_depth**: Retrieve read depths from a BAM file
- **get_allele_coverage**: Retrieve per-allele coverage from BAM file

&nbsp;
## load_data.py
Functions that read non-encoded data from files.
- **load_vcf**: Read VCF and load SNP's/Genotypes into dataframe

&nbsp;
## load_h5.py
Functions that load H5 data to python objects.
- **load_onehot_h5**: Load onehot encoded genome from H5 to dictionary
- **load_depth_h5**: Load read depths from H5 to dictionary
- **load_coverage_h5**: Load allele coverage from H5 to dictionary
