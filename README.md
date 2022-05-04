# genome-loader
Pipeline for efficient genomic data processing.

&nbsp;
## Requirements
- Python >= 3.7
- h5py
- pysam
- numpy
- pandas

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
    - [writedepth](#writedepth)
    - [writecoverage](#writecoverage)
- [Python Functions](#python-functions)
    - [encode_data.py](#encodedatapy)
    - [get_encoded.py](#getencodedpy)
    - [get_data.py](#getdatapy)
    - [load_data.py](#loaddatapy)

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
## writedepth
Writes BAM read depth into HDF5 file.

&nbsp;
**File Format**
- Group: `[chrom]`
- Dataset: `"depth"` - 0-based array with depth per position
- Attributes: `"id"`- dataset name associated with file

&nbsp;
**Usage**
```shell script
gloader writedepth [BAM] --output/--directory [OUT] {OPTIONS}
```

**Required Arguments**
- BAM: Positional argument, BAM file to parse and write to H5
- -o/--output: Full path and file name of output (NOTE: Cannot use both -o and -d flags)
- -d/--directory: Directory to write hdf5 output

**Optional Arguments**
- -c/--chroms: Chromosomes to write (Default: ALL)
- -l/--lens: Lengths of provided chroms (Auto retrieved if not provided)
- -n/--name: Output file if --directory given, ignored if using --output flag. Defaults to input fasta name

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
Contains functions for creating one-hot encoded data
- **encode_sequence**: Encodes input data into one-hot encoded format
- **encode_from_fasta**: Create one-hot encoded data directly from FASTA
- **encode_from_h5**: Create one-hot encoded data from char-array encoded H5

&nbsp;
## get_encoded.py
Contains functions for loading, and transforming one-hot encoded data
- **get_encoded_haps**: Creates one-hot encoded haplotypes from one-hot encoded data

&nbsp;
## get_data.py
Functions that retrieves non-encoded data from files
- **get_read_depth**: Retrieve read depths from a BAM file
- **get_allele_coverage**: Retrieve per-allele coverage from BAM file

&nbsp;
## load_data.py
Functions that read non-encoded data from files
- **load_vcf**: Read VCF and load SNP's/Genotypes into dataframe

