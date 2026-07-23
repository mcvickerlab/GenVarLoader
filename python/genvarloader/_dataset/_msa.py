import time
import numpy as np
import numba as nb


@nb.njit(cache=True)
def stack_ploidy(arr):
    """Optimized ploidy stacking using numba.

    This function takes a 3D array of shape (n_samples, 2, seq_len) containing diploid sequences
    and stacks them into a 2D array of shape (n_samples*2, seq_len) where each sample's two
    haplotypes are placed in consecutive rows.

    Args:
        arr: 3D numpy array of shape (n_samples, 2, seq_len) containing diploid sequences
            where arr[i,0] is the first haplotype and arr[i,1] is the second haplotype
            for sample i.

    Returns:
        2D numpy array of shape (n_samples*2, seq_len) where rows 2*i and 2*i+1 contain
        the two haplotypes for sample i.
    """
    n_samples = arr.shape[0]
    seq_len = arr.shape[2]
    result = np.empty((n_samples * 2, seq_len), dtype=arr.dtype)
    for i in range(n_samples):
        result[i * 2] = arr[i, 0]
        result[i * 2 + 1] = arr[i, 1]
    return result


@nb.njit(parallel=True, cache=True)
def _create_msa_fast(ref_coords, seq_arr, min_coord, max_coord):
    """Numba-accelerated core MSA creation function.

    This internal function creates a gappy Multiple Sequence Alignment (MSA) by mapping
    sequence positions to their corresponding reference coordinates. It uses Numba's
    parallel processing capabilities for improved performance.

    Args:
        ref_coords: 2D array of shape (n_seqs, seq_len) containing reference coordinates
            for each position in each sequence
        seq_arr: 2D array of shape (n_seqs, seq_len) containing the sequence data as
            integer-encoded nucleotides
        min_coord: Integer indicating the minimum reference coordinate across all sequences
        max_coord: Integer indicating the maximum reference coordinate across all sequences

    Returns:
        2D array of shape (n_seqs, alignment_length) containing the gappy MSA where:
        - Each row represents one sequence
        - Each column represents one position in the reference
        - Gaps are represented by the ASCII value of '-'
        - Nucleotides are represented by their ASCII values
    """
    n_seqs = ref_coords.shape[0]
    alignment_length = max_coord - min_coord + 1
    msa = np.full((n_seqs, alignment_length), ord("-"), dtype=np.int8)

    # Pre-compute coordinate offsets to avoid repeated subtraction in the loop
    coord_offsets = ref_coords - min_coord

    for i in nb.prange(n_seqs):
        coords = coord_offsets[i]
        seq = seq_arr[i]
        seq_len = len(coords)
        for j in range(seq_len):
            msa[i, coords[j]] = seq[j]

    return msa


def create_msa(ref_coords, seq_arr):
    """
    Create a gappy Multiple Sequence Alignment from a ragged 2D array using reference coordinates.

    This will take a 2D array of the reference coordinates:
        ref_coords = [
        [100, 101, 103],  # First sequence has bases at positions 100, 101, 103
        [100, 102, 103]   # Second sequence has bases at positions 100, 102, 103
        ]
    And a 2D array of the sequences (left-aligned):
        seq_arr = [
            ['A', 'T', 'G'],  # First sequence
            ['A', 'C', 'G']   # Second sequence
        ]
    And return a gappy Multiple Sequence Alignment that is aligned at all positions.
        A T - G  # First sequence
        A - C G  # Second sequence
    Args:
        ref_coords: 2D array of reference coordinates (sequence x position) indicating where each position maps in the reference genome
        seq_arr: 2D array of sequences (sequence x position) with ragged right end

    Returns:
        2D array containing the gappy MSA
    """
    start_time = time.time()

    assert ref_coords.shape == seq_arr.shape, (
        f"Shapes do not match: ref_coords.shape ({ref_coords.shape}) != seq_arr.shape ({seq_arr.shape})"
    )

    # Convert input arrays to numpy arrays if they aren't already
    ref_coords = np.asarray(ref_coords)
    seq_arr = np.asarray(seq_arr)

    # Find the min and max reference coordinates across all sequences
    min_coord = np.min(ref_coords)
    max_coord = np.max(ref_coords)

    # Optimize byte conversion using vectorized operations
    if seq_arr.dtype == np.dtype("|S1"):
        seq_arr_int = np.frombuffer(seq_arr.tobytes(), dtype=np.int8).reshape(
            seq_arr.shape
        )
    else:
        seq_arr_int = np.array(
            [[ord(c) for c in row] for row in seq_arr], dtype=np.int8
        )

    # Create MSA using numba-accelerated function
    msa = _create_msa_fast(ref_coords, seq_arr_int, min_coord, max_coord)

    # Convert back to bytes array using view
    msa = msa.view("|S1")

    end_time = time.time()
    print(
        f"MSA for {seq_arr.shape[0]} sequences done {end_time - start_time:.2f} seconds"
    )

    return msa


def preview_msa(msa, max_len=None):
    """
    Preview the MSA by printing the sequences with differences.

    Args:
        msa: 2D array containing the gappy MSA
        max_len: Maximum number of columns to print

    Returns:
        None
    """
    # Convert byte arrays to strings and find columns with differences using numpy vectorization
    msa_str = np.array(
        [[x.decode("utf-8") for x in row] for row in msa], dtype="U1"
    )  # Convert bytes to unicode strings row by row

    # Find columns with differences using numpy operations
    col_unique = np.apply_along_axis(lambda x: len(np.unique(x)), 0, msa_str)
    diff_cols = np.where(col_unique > 1)[0]

    # Limit the number of columns if max_len is specified
    if max_len is not None and len(diff_cols) > max_len:
        diff_cols = diff_cols[:max_len]

    # Print sequences showing only columns with differences using numpy indexing
    for seq in msa_str:
        print("".join(seq[diff_cols]))


@nb.njit(nogil=True, cache=True)
def _get_consensus(msa_uint8):
    """
    Get the consensus sequence from a multiple sequence alignment (MSA) represented as uint8 array.

    For each column in the MSA, this function finds the most frequently occurring value.
    The function is accelerated using Numba for better performance.

    Args:
        msa_uint8 (np.ndarray): 2D array of shape (n_sequences, n_columns) containing the MSA
                               encoded as uint8 values.

    Returns:
        np.ndarray: 1D array of length n_columns containing the consensus sequence
                    encoded as uint8 values.
    """
    n_cols = msa_uint8.shape[1]
    consensus = np.zeros(n_cols, dtype=np.uint8)

    for col in range(n_cols):
        # Count occurrences of each value
        counts = np.zeros(256, dtype=np.int32)
        for row in range(msa_uint8.shape[0]):
            val = msa_uint8[row, col]
            counts[val] += 1

        # Find most common value
        max_count = -1
        max_val = 0
        for val in range(256):
            if counts[val] > max_count:
                max_count = counts[val]
                max_val = val

        consensus[col] = max_val

    return consensus


def create_consensus_seq(msa, as_str=False):
    """
    Create a consensus sequence from a gappy MSA.

    This function takes a multiple sequence alignment (MSA) and generates a consensus sequence
    by finding the most frequently occurring character at each position. The function uses
    Numba-accelerated operations for improved performance.

    Args:
        msa (np.ndarray): 2D array containing the gappy MSA created with create_msa.
                         Should be a structured array with string dtype.
        as_str (bool, optional): Whether to return the consensus sequence as a string.
                               If False, returns a structured array with string dtype.
                               Defaults to False.

    Returns:
        np.ndarray or str: The consensus sequence. If as_str is True, returns a string.
                          Otherwise returns a structured array with string dtype ('|S1').
    """
    start_time = time.time()
    # Convert to uint8 and reshape
    msa_uint8 = msa.view(np.uint8).reshape(msa.shape[0], -1)

    # Get consensus using numba function
    consensus_seq = _get_consensus(msa_uint8)

    end_time = time.time()
    print(
        f"Consensus sequence of {msa.shape[0]} sequences created in {end_time - start_time:.2f} seconds"
    )

    if as_str is True:
        consensus_seq = bytearray_to_string(consensus_seq)
    else:
        consensus_seq = consensus_seq.view("|S1")
    return consensus_seq


def bytearray_to_string(byte_arr):
    """
    Convert a bytearray to a string.

    Args:
        byte_arr: bytearray, the bytearray to convert.

    Returns:
        str, the string of the bytearray.

    Example:
        >>> bytearray_to_string(np.array([b'A', b'C', b'G', b'T']))
        'ACGT'
    """
    if isinstance(byte_arr, str):
        return byte_arr
    return byte_arr.tobytes().decode()
