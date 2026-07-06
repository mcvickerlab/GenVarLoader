# SVAR2 read-bound Python-layer baseline (2026-07-06)
2026-07-06 00:30:19.871 | INFO     | genvarloader._dataset._write:write:196 - Writing dataset to tmp/svar2_mvp/prof_out/readbound/germline_haplotypes.gvl
2026-07-06 00:30:19.889 | INFO     | genvarloader._dataset._write:write:288 - Using 3202 samples.
2026-07-06 00:30:19.889 | INFO     | genvarloader._dataset._write:write:293 - Writing genotypes.
  0%|          | 0/3 [00:00<?, ? region/s]Processing svar2 ranges for 3 regions on chr21:   0%|          | 0/3 [00:00<?, ? region/s]Processing svar2 ranges for 3 regions on chr21: 100%|██████████| 3/3 [00:00<00:00,  4.91 region/s]Processing svar2 ranges for 3 regions on chr21: 100%|██████████| 3/3 [00:00<00:00,  4.91 region/s]
2026-07-06 00:30:20.683 | INFO     | genvarloader._dataset._write:write:388 - Finished writing.
2026-07-06 00:30:20.822 | INFO     | genvarloader._dataset._open:resolve:95 - Opened dataset:
Unspliced GVL dataset at tmp/svar2_mvp/prof_out/readbound/germline_haplotypes.gvl
Is subset: False
# of regions: 3
# of samples: 3202
Output length: ragged
Jitter: 0 (max: 0)
Deterministic: True
Sequence type: reference [haplotypes] annotated variants variant-windows
Active tracks: None
Tracks available: None

### cProfile haplotypes germline (K=200), sort=cumulative

```
         77201 function calls in 43.451 seconds

   Ordered by: cumulative time
   List reduced from 136 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      200    0.738    0.004   43.452    0.217 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/prof_getitem.py:48(call)
      200    0.001    0.000   42.714    0.214 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_impl.py:2218(__getitem__)
      200    0.004    0.000   42.713    0.214 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_impl.py:1751(__getitem__)
      200    0.002    0.000   42.706    0.214 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:66(getitem)
      200    0.044    0.000   42.681    0.213 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:154(_getitem_unspliced)
      200    0.006    0.000   42.607    0.213 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:256(__call__)
      200    0.146    0.001   42.599    0.213 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:319(get_haps_and_shifts)
      200    4.692    0.023   38.910    0.195 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:714(_assemble_haps)
      200    8.965    0.045   34.167    0.171 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:80(_ragged_arange_gather)
     1200    0.003    0.000   16.360    0.014 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
      600    0.002    0.000   16.352    0.027 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:423(repeat)
      600   16.346    0.027   16.346    0.027 {method 'repeat' of 'numpy.ndarray' objects}
      600    8.851    0.015    8.851    0.015 {built-in method numpy.arange}
      200    1.972    0.010    1.972    0.010 {built-in method genvarloader.genvarloader.reconstruct_haplotypes_from_svar2_readbound}
      200    1.271    0.006    1.271    0.006 {built-in method genvarloader.genvarloader.hap_diffs_from_svar2_readbound}
      400    0.008    0.000    0.253    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:632(_gather_inputs)
     1600    0.209    0.000    0.214    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/memmap.py:334(__getitem__)
      200    0.034    0.000    0.037    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:695(_inverse_row_perm)
     3200    0.030    0.000    0.030    0.000 {built-in method numpy.ascontiguousarray}
      200    0.013    0.000    0.023    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_indexing.py:208(parse_idx)
      200    0.004    0.000    0.022    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:619(_contig_groups)
      400    0.000    0.000    0.017    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:119(<genexpr>)
      200    0.001    0.000    0.016    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:131(_reshape_outer)
      200    0.001    0.000    0.015    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:1428(reshape)
      200    0.004    0.000    0.015    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:1434(_reshape_impl)
      200    0.001    0.000    0.014    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/lib/arraysetops.py:138(unique)
      200    0.004    0.000    0.013    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/lib/arraysetops.py:323(_unique1d)
      400    0.002    0.000    0.012    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_utils.py:56(lengths_to_offsets)
      600    0.001    0.000    0.011    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:2979(prod)
      400    0.001    0.000    0.009    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:2512(cumsum)


```

### pyinstrument haplotypes germline (K=200)

```

  _     ._   __/__   _ _  _  _ _/_   Recorded: 00:31:04  Samples:  2811
 /_//_/// /_\ / //_// / //_'/ //     Duration: 43.543    CPU time: 44.033
/   _/                      v5.1.2

Profile at /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/prof_python.py:32

43.5427 main  prof_python.py:16
`- 43.5427 call  prof_getitem.py:48
   |- 42.7615 RaggedDataset.__getitem__  genvarloader/_dataset/_impl.py:2218
   |  `- 42.7615 RaggedDataset.__getitem__  genvarloader/_dataset/_impl.py:1751
   |     `- 42.7615 getitem  genvarloader/_dataset/_query.py:66
   |        `- 42.7615 _getitem_unspliced  genvarloader/_dataset/_query.py:154
   |           `- 42.7594 Svar2Haps.__call__  genvarloader/_dataset/_svar2_haps.py:256
   |              `- 42.7594 Svar2Haps.get_haps_and_shifts  genvarloader/_dataset/_svar2_haps.py:319
   |                 |- 39.0037 Svar2Haps._assemble_haps  genvarloader/_dataset/_svar2_haps.py:714
   |                 |  |- 34.2766 _ragged_arange_gather  genvarloader/_dataset/_svar2_haps.py:80
   |                 |  |  |- 16.3452 repeat  numpy/core/fromnumeric.py:423
   |                 |  |  |  `- 16.3452 _wrapfunc  numpy/core/fromnumeric.py:53
   |                 |  |  |     `- 16.3452 ndarray.repeat  <built-in>
   |                 |  |  |- 8.9989 [self]  genvarloader/_dataset/_svar2_haps.py
   |                 |  |  `- 8.9325 arange  <built-in>
   |                 |  `- 4.7255 [self]  genvarloader/_dataset/_svar2_haps.py
   |                 |- 1.9794 reconstruct_haplotypes_from_svar2_readbound  <built-in>
   |                 `- 1.3297 hap_diffs_from_svar2_readbound  <built-in>
   `- 0.7812 [self]  prof_getitem.py

```

2026-07-06 00:31:54.920 | INFO     | genvarloader._dataset._write:write:196 - Writing dataset to tmp/svar2_mvp/prof_out/readbound/germline_variants.gvl
2026-07-06 00:31:54.941 | INFO     | genvarloader._dataset._write:write:288 - Using 3202 samples.
2026-07-06 00:31:54.941 | INFO     | genvarloader._dataset._write:write:293 - Writing genotypes.
  0%|          | 0/3 [00:00<?, ? region/s]Processing svar2 ranges for 3 regions on chr21:   0%|          | 0/3 [00:00<?, ? region/s]Processing svar2 ranges for 3 regions on chr21: 100%|██████████| 3/3 [00:00<00:00,  4.87 region/s]Processing svar2 ranges for 3 regions on chr21: 100%|██████████| 3/3 [00:00<00:00,  4.87 region/s]
2026-07-06 00:31:55.733 | INFO     | genvarloader._dataset._write:write:388 - Finished writing.
2026-07-06 00:31:55.874 | INFO     | genvarloader._dataset._open:resolve:95 - Opened dataset:
Unspliced GVL dataset at tmp/svar2_mvp/prof_out/readbound/germline_variants.gvl
Is subset: False
# of regions: 3
# of samples: 3202
Output length: ragged
Jitter: 0 (max: 0)
Deterministic: True
Sequence type: reference [haplotypes] annotated variants variant-windows
Active tracks: None
Tracks available: None

### cProfile variants germline (K=200), sort=cumulative

```
         149801 function calls (148601 primitive calls) in 1.064 seconds

   Ordered by: cumulative time
   List reduced from 140 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      200    0.000    0.000    1.064    0.005 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/prof_getitem.py:48(call)
      200    0.000    0.000    1.064    0.005 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_impl.py:2218(__getitem__)
      200    0.001    0.000    1.063    0.005 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_impl.py:1751(__getitem__)
      200    0.001    0.000    1.061    0.005 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:66(getitem)
      200    0.037    0.000    1.031    0.005 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:154(_getitem_unspliced)
      200    0.001    0.000    0.975    0.005 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:256(__call__)
      200    0.041    0.000    0.973    0.005 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:527(_reconstruct_variants)
      200    0.542    0.003    0.542    0.003 {built-in method genvarloader.genvarloader.decode_variants_from_svar2_readbound}
     3000    0.002    0.000    0.134    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
     1600    0.001    0.000    0.116    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:423(repeat)
     1600    0.114    0.000    0.114    0.000 {method 'repeat' of 'numpy.ndarray' objects}
      400    0.030    0.000    0.105    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:80(_ragged_arange_gather)
      200    0.003    0.000    0.104    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:632(_gather_inputs)
      200    0.029    0.000    0.099    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:101(_ragged_arange_gather_2level)
      800    0.086    0.000    0.088    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/memmap.py:334(__getitem__)
      200    0.031    0.000    0.033    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:695(_inverse_row_perm)
      400    0.000    0.000    0.028    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:119(<genexpr>)
      200    0.000    0.000    0.028    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:131(_reshape_outer)
  800/200    0.001    0.000    0.027    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:1428(reshape)
  800/200    0.006    0.000    0.027    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:1434(_reshape_impl)
     1200    0.002    0.000    0.022    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_utils.py:56(lengths_to_offsets)
     1200    0.001    0.000    0.019    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:2512(cumsum)
      400    0.003    0.000    0.018    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:188(from_fields)
      200    0.001    0.000    0.018    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:1437(<dictcomp>)
     1200    0.017    0.000    0.017    0.000 {method 'cumsum' of 'numpy.ndarray' objects}
      200    0.002    0.000    0.015    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:619(_contig_groups)
      200    0.001    0.000    0.015    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_rag_variants.py:210(__init__)
      200    0.009    0.000    0.014    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_indexing.py:208(parse_idx)
     1200    0.013    0.000    0.013    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/lib/function_base.py:1324(diff)
     1400    0.012    0.000    0.012    0.000 {built-in method numpy.ascontiguousarray}


```

### pyinstrument variants germline (K=200)

```

  _     ._   __/__   _ _  _  _ _/_   Recorded: 00:31:56  Samples:  1200
 /_//_/// /_\ / //_// / //_'/ //     Duration: 1.070     CPU time: 1.064
/   _/                      v5.1.2

Profile at /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/prof_python.py:32

1.0697 main  prof_python.py:16
`- 1.0697 call  prof_getitem.py:48
   `- 1.0697 RaggedDataset.__getitem__  genvarloader/_dataset/_impl.py:2218
      `- 1.0697 RaggedDataset.__getitem__  genvarloader/_dataset/_impl.py:1751
         `- 1.0692 getitem  genvarloader/_dataset/_query.py:66
            |- 0.9705 _getitem_unspliced  genvarloader/_dataset/_query.py:154
            |  `- 0.9705 Svar2Haps.__call__  genvarloader/_dataset/_svar2_haps.py:256
            |     `- 0.9705 Svar2Haps._reconstruct_variants  genvarloader/_dataset/_svar2_haps.py:527
            |        |- 0.5452 decode_variants_from_svar2_readbound  <built-in>
            |        |- 0.1100 [self]  genvarloader/_dataset/_svar2_haps.py
            |        |- 0.1062 Svar2Haps._gather_inputs  genvarloader/_dataset/_svar2_haps.py:632
            |        |  |- 0.0954 ascontiguousarray  <built-in>
            |        |  `- 0.0108 memmap.__getitem__  numpy/core/memmap.py:334
            |        |- 0.1040 _ragged_arange_gather  genvarloader/_dataset/_svar2_haps.py:80
            |        |  `- 0.1040 repeat  numpy/core/fromnumeric.py:423
            |        |     `- 0.1040 _wrapfunc  numpy/core/fromnumeric.py:53
            |        |        `- 0.1040 ndarray.repeat  <built-in>
            |        `- 0.1020 _ragged_arange_gather_2level  genvarloader/_dataset/_svar2_haps.py:101
            |           `- 0.0974 repeat  numpy/core/fromnumeric.py:423
            |              `- 0.0974 _wrapfunc  numpy/core/fromnumeric.py:53
            |                 `- 0.0969 ndarray.repeat  <built-in>
            `- 0.0978 <genexpr>  genvarloader/_dataset/_query.py:119
               `- 0.0968 _reshape_outer  genvarloader/_dataset/_query.py:131
                  `- 0.0963 RaggedVariants.reshape  seqpro/rag/_core.py:1428
                     `- 0.0948 RaggedVariants._reshape_impl  seqpro/rag/_core.py:1434
                        `- 0.0903 from_fields  seqpro/rag/_core.py:188
                           |- 0.0617 <genexpr>  seqpro/rag/_core.py:206
                           |  `- 0.0577 array_equal  numpy/core/numeric.py:2378
                           |     |- 0.0417 [self]  numpy/core/numeric.py
                           |     `- 0.0125 _all  numpy/core/_methods.py:61
                           `- 0.0160 <dictcomp>  seqpro/rag/_core.py:213

```

2026-07-06 00:32:02.542 | INFO     | genvarloader._dataset._write:write:196 - Writing dataset to tmp/svar2_mvp/prof_out/readbound/somatic_haplotypes.gvl
2026-07-06 00:32:02.562 | INFO     | genvarloader._dataset._write:write:288 - Using 16007 samples.
2026-07-06 00:32:02.563 | INFO     | genvarloader._dataset._write:write:293 - Writing genotypes.
  0%|          | 0/3 [00:00<?, ? region/s]Processing svar2 ranges for 3 regions on chr21:   0%|          | 0/3 [00:00<?, ? region/s]Processing svar2 ranges for 3 regions on chr21: 100%|██████████| 3/3 [00:03<00:00,  1.31s/ region]Processing svar2 ranges for 3 regions on chr21: 100%|██████████| 3/3 [00:03<00:00,  1.31s/ region]
2026-07-06 00:32:08.014 | INFO     | genvarloader._dataset._write:write:388 - Finished writing.
2026-07-06 00:32:08.145 | INFO     | genvarloader._dataset._open:resolve:95 - Opened dataset:
Unspliced GVL dataset at tmp/svar2_mvp/prof_out/readbound/somatic_haplotypes.gvl
Is subset: False
# of regions: 3
# of samples: 16007
Output length: ragged
Jitter: 0 (max: 0)
Deterministic: True
Sequence type: reference [haplotypes] annotated variants variant-windows
Active tracks: None
Tracks available: None

### cProfile haplotypes somatic (K=50), sort=cumulative

```
         19301 function calls in 53.447 seconds

   Ordered by: cumulative time
   List reduced from 136 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       50    0.338    0.007   53.447    1.069 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/prof_getitem.py:48(call)
       50    0.000    0.000   53.109    1.062 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_impl.py:2218(__getitem__)
       50    0.001    0.000   53.108    1.062 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_impl.py:1751(__getitem__)
       50    0.001    0.000   53.106    1.062 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:66(getitem)
       50    0.047    0.001   53.099    1.062 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:154(_getitem_unspliced)
       50    0.360    0.007   53.036    1.061 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:256(__call__)
       50    0.554    0.011   52.675    1.053 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:319(get_haps_and_shifts)
       50    6.248    0.125   49.781    0.996 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:714(_assemble_haps)
       50   11.425    0.228   43.483    0.870 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:80(_ragged_arange_gather)
      300    0.001    0.000   20.497    0.068 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
      150    0.001    0.000   20.489    0.137 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:423(repeat)
      150   20.487    0.137   20.487    0.137 {method 'repeat' of 'numpy.ndarray' objects}
      150   11.570    0.077   11.570    0.077 {built-in method numpy.arange}
       50    1.400    0.028    1.400    0.028 {built-in method genvarloader.genvarloader.reconstruct_haplotypes_from_svar2_readbound}
       50    0.551    0.011    0.551    0.011 {built-in method genvarloader.genvarloader.hap_diffs_from_svar2_readbound}
      100    0.013    0.000    0.343    0.003 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:632(_gather_inputs)
      400    0.293    0.001    0.294    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/memmap.py:334(__getitem__)
       50    0.041    0.001    0.044    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:695(_inverse_row_perm)
      800    0.036    0.000    0.036    0.000 {built-in method numpy.ascontiguousarray}
       50    0.002    0.000    0.026    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:619(_contig_groups)
       50    0.000    0.000    0.015    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/lib/arraysetops.py:138(unique)
       50    0.002    0.000    0.015    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/lib/arraysetops.py:323(_unique1d)
      200    0.012    0.000    0.012    0.000 {method 'astype' of 'numpy.ndarray' objects}
       50    0.009    0.000    0.012    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_indexing.py:208(parse_idx)
       50    0.011    0.000    0.011    0.000 {method 'sort' of 'numpy.ndarray' objects}
      100    0.001    0.000    0.009    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_utils.py:56(lengths_to_offsets)
      100    0.000    0.000    0.008    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:2512(cumsum)
      100    0.007    0.000    0.007    0.000 {method 'cumsum' of 'numpy.ndarray' objects}
      100    0.007    0.000    0.007    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/lib/function_base.py:1324(diff)
      100    0.000    0.000    0.005    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:119(<genexpr>)


```

### pyinstrument haplotypes somatic (K=50)

```

  _     ._   __/__   _ _  _  _ _/_   Recorded: 00:33:03  Samples:  1451
 /_//_/// /_\ / //_// / //_'/ //     Duration: 53.412    CPU time: 53.899
/   _/                      v5.1.2

Profile at /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/prof_python.py:32

53.4118 main  prof_python.py:16
`- 53.4118 call  prof_getitem.py:48
   `- 53.0665 RaggedDataset.__getitem__  genvarloader/_dataset/_impl.py:2218
      `- 53.0665 RaggedDataset.__getitem__  genvarloader/_dataset/_impl.py:1751
         `- 53.0665 getitem  genvarloader/_dataset/_query.py:66
            `- 53.0665 _getitem_unspliced  genvarloader/_dataset/_query.py:154
               `- 53.0170 Svar2Haps.__call__  genvarloader/_dataset/_svar2_haps.py:256
                  `- 52.6492 Svar2Haps.get_haps_and_shifts  genvarloader/_dataset/_svar2_haps.py:319
                     |- 49.7006 Svar2Haps._assemble_haps  genvarloader/_dataset/_svar2_haps.py:714
                     |  |- 43.3816 _ragged_arange_gather  genvarloader/_dataset/_svar2_haps.py:80
                     |  |  |- 20.5025 repeat  numpy/core/fromnumeric.py:423
                     |  |  |  `- 20.5025 _wrapfunc  numpy/core/fromnumeric.py:53
                     |  |  |     `- 20.5025 ndarray.repeat  <built-in>
                     |  |  |- 11.5908 arange  <built-in>
                     |  |  `- 11.2883 [self]  genvarloader/_dataset/_svar2_haps.py
                     |  `- 6.2733 [self]  genvarloader/_dataset/_svar2_haps.py
                     |- 1.4053 reconstruct_haplotypes_from_svar2_readbound  <built-in>
                     |- 0.5750 [self]  genvarloader/_dataset/_svar2_haps.py
                     `- 0.5739 hap_diffs_from_svar2_readbound  <built-in>

```

2026-07-06 00:34:04.651 | INFO     | genvarloader._dataset._write:write:196 - Writing dataset to tmp/svar2_mvp/prof_out/readbound/somatic_variants.gvl
2026-07-06 00:34:04.732 | INFO     | genvarloader._dataset._write:write:288 - Using 16007 samples.
2026-07-06 00:34:04.732 | INFO     | genvarloader._dataset._write:write:293 - Writing genotypes.
  0%|          | 0/3 [00:00<?, ? region/s]Processing svar2 ranges for 3 regions on chr21:   0%|          | 0/3 [00:00<?, ? region/s]Processing svar2 ranges for 3 regions on chr21: 100%|██████████| 3/3 [00:03<00:00,  1.30s/ region]Processing svar2 ranges for 3 regions on chr21: 100%|██████████| 3/3 [00:03<00:00,  1.30s/ region]
2026-07-06 00:34:10.126 | INFO     | genvarloader._dataset._write:write:388 - Finished writing.
2026-07-06 00:34:10.270 | INFO     | genvarloader._dataset._open:resolve:95 - Opened dataset:
Unspliced GVL dataset at tmp/svar2_mvp/prof_out/readbound/somatic_variants.gvl
Is subset: False
# of regions: 3
# of samples: 16007
Output length: ragged
Jitter: 0 (max: 0)
Deterministic: True
Sequence type: reference [haplotypes] annotated variants variant-windows
Active tracks: None
Tracks available: None

### cProfile variants somatic (K=200), sort=cumulative

```
         149801 function calls (148601 primitive calls) in 4.216 seconds

   Ordered by: cumulative time
   List reduced from 140 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      200    0.175    0.001    4.216    0.021 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/prof_getitem.py:48(call)
      200    0.000    0.000    4.041    0.020 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_impl.py:2218(__getitem__)
      200    0.003    0.000    4.041    0.020 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_impl.py:1751(__getitem__)
      200    0.001    0.000    4.036    0.020 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:66(getitem)
      200    0.167    0.001    3.994    0.020 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:154(_getitem_unspliced)
      200    0.071    0.000    3.780    0.019 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:256(__call__)
      200    0.270    0.001    3.709    0.019 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:527(_reconstruct_variants)
      200    2.010    0.010    2.010    0.010 {built-in method genvarloader.genvarloader.decode_variants_from_svar2_readbound}
      200    0.033    0.000    0.809    0.004 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:632(_gather_inputs)
      800    0.693    0.001    0.696    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/memmap.py:334(__getitem__)
     3000    0.002    0.000    0.224    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
      400    0.068    0.000    0.223    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:80(_ragged_arange_gather)
     1600    0.001    0.000    0.172    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:423(repeat)
     1600    0.169    0.000    0.169    0.000 {method 'repeat' of 'numpy.ndarray' objects}
      200    0.152    0.001    0.160    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:695(_inverse_row_perm)
      200    0.035    0.000    0.113    0.001 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:101(_ragged_arange_gather_2level)
     1400    0.080    0.000    0.080    0.000 {built-in method numpy.ascontiguousarray}
      200    0.005    0.000    0.065    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_svar2_haps.py:619(_contig_groups)
     1200    0.003    0.000    0.053    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_utils.py:56(lengths_to_offsets)
      200    0.001    0.000    0.052    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/lib/arraysetops.py:138(unique)
      200    0.006    0.000    0.051    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/lib/arraysetops.py:323(_unique1d)
     1200    0.001    0.000    0.049    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/numpy/core/fromnumeric.py:2512(cumsum)
     1200    0.047    0.000    0.047    0.000 {method 'cumsum' of 'numpy.ndarray' objects}
      200    0.042    0.000    0.042    0.000 {method 'sort' of 'numpy.ndarray' objects}
      400    0.000    0.000    0.039    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:119(<genexpr>)
      200    0.000    0.000    0.039    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_query.py:131(_reshape_outer)
      200    0.031    0.000    0.038    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/python/genvarloader/_dataset/_indexing.py:208(parse_idx)
  800/200    0.001    0.000    0.038    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:1428(reshape)
  800/200    0.007    0.000    0.037    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:1434(_reshape_impl)
      400    0.004    0.000    0.036    0.000 /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/.pixi/envs/dev/lib/python3.10/site-packages/seqpro/rag/_core.py:188(from_fields)


```

### pyinstrument variants somatic (K=200)

```

  _     ._   __/__   _ _  _  _ _/_   Recorded: 00:34:14  Samples:  2807
 /_//_/// /_\ / //_// / //_'/ //     Duration: 4.232     CPU time: 4.210
/   _/                      v5.1.2

Profile at /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel/tmp/svar2_mvp/prof_python.py:32

4.2318 main  prof_python.py:16
`- 4.2318 call  prof_getitem.py:48
   |- 4.0152 RaggedDataset.__getitem__  genvarloader/_dataset/_impl.py:2218
   |  `- 4.0152 RaggedDataset.__getitem__  genvarloader/_dataset/_impl.py:1751
   |     `- 4.0152 getitem  genvarloader/_dataset/_query.py:66
   |        `- 4.0137 _getitem_unspliced  genvarloader/_dataset/_query.py:154
   |           |- 3.8348 Svar2Haps.__call__  genvarloader/_dataset/_svar2_haps.py:256
   |           |  |- 3.6962 Svar2Haps._reconstruct_variants  genvarloader/_dataset/_svar2_haps.py:527
   |           |  |  |- 2.0787 decode_variants_from_svar2_readbound  <built-in>
   |           |  |  |- 0.7419 Svar2Haps._gather_inputs  genvarloader/_dataset/_svar2_haps.py:632
   |           |  |  |  `- 0.7413 memmap.__getitem__  numpy/core/memmap.py:334
   |           |  |  |- 0.3715 [self]  genvarloader/_dataset/_svar2_haps.py
   |           |  |  |- 0.2237 _ragged_arange_gather  genvarloader/_dataset/_svar2_haps.py:80
   |           |  |  |  |- 0.1215 [self]  genvarloader/_dataset/_svar2_haps.py
   |           |  |  |  `- 0.1012 repeat  numpy/core/fromnumeric.py:423
   |           |  |  |     `- 0.1012 _wrapfunc  numpy/core/fromnumeric.py:53
   |           |  |  |        `- 0.1012 ndarray.repeat  <built-in>
   |           |  |  |- 0.1660 _inverse_row_perm  genvarloader/_dataset/_svar2_haps.py:695
   |           |  |  `- 0.1110 _ragged_arange_gather_2level  genvarloader/_dataset/_svar2_haps.py:101
   |           |  |     |- 0.0569 repeat  numpy/core/fromnumeric.py:423
   |           |  |     |  `- 0.0569 _wrapfunc  numpy/core/fromnumeric.py:53
   |           |  |     |     `- 0.0569 ndarray.repeat  <built-in>
   |           |  |     `- 0.0541 [self]  genvarloader/_dataset/_svar2_haps.py
   |           |  `- 0.1386 [self]  genvarloader/_dataset/_svar2_haps.py
   |           `- 0.1784 [self]  genvarloader/_dataset/_query.py
   `- 0.2166 [self]  prof_getitem.py

```


## Top Python functions (ranked)

Ranked by cProfile cumulative time (`cumtime`), cross-checked against the
pyinstrument call trees. "gvl Python" = defined in `python/genvarloader/`.
FFI wrappers = `{built-in method genvarloader.genvarloader.*}` frames — these
are single-call-boundary hops into the compiled Rust extension; their
`tottime` is Rust execution time, not Python interpreter overhead.

### haplotypes mode (germline K=200, somatic K=50)

| rank | function | cumtime (germline / somatic) | tottime (germline / somatic) | kind |
|---|---|---|---|---|
| 1 | `Svar2Haps.get_haps_and_shifts` (`_svar2_haps.py:319`) | 42.599s / 52.675s | 0.146s / 0.554s | pure-Python dispatcher (thin — nearly all time is in callees) |
| 2 | `Svar2Haps._assemble_haps` (`_svar2_haps.py:714`) | 38.910s / 49.781s | 4.692s / 6.248s | pure-Python, real work — non-trivial self-time plus calls into `_ragged_arange_gather` |
| 3 | `_ragged_arange_gather` (`_svar2_haps.py:80`) | 34.167s / 43.483s | 8.965s / 11.425s | pure-Python hot loop — dominates via the numpy calls it issues (`ndarray.repeat` 16.3s/20.5s, `numpy.arange` 8.9s/11.6s of tottime); this, not `_gather_inputs`, is the second-largest gvl-Python cost center |
| 4 | `reconstruct_haplotypes_from_svar2_readbound` (built-in) | 1.972s / 1.400s | 1.972s / 1.400s | **FFI wrapper** (Rust) — thin, self-time is Rust-side |
| 5 | `hap_diffs_from_svar2_readbound` (built-in) | 1.271s / 0.551s | 1.271s / 0.551s | **FFI wrapper** (Rust) — thin, self-time is Rust-side |
| — | `Svar2Haps._gather_inputs` (`_svar2_haps.py:632`) | 0.253s / 0.343s | 0.008s / 0.013s | pure-Python — **much smaller than expected**; see note below |

**Deviation from expectation:** the brief expected `_gather_inputs` to rank among the top 3
alongside `get_haps_and_shifts` and `_assemble_haps`. It does not — observed cumtime for
`_gather_inputs` is 0.25–0.34s vs. 38.9–49.8s for `_assemble_haps`, i.e. ~2 orders of magnitude
smaller in haplotypes mode. The actual #2/#3 hot spots are `_assemble_haps` itself (self-time)
and `_ragged_arange_gather`, which together account for essentially all non-FFI time; the
`numpy.ndarray.repeat` and `numpy.arange` calls issued *from inside* `_ragged_arange_gather`
are the single largest tottime consumers of the whole profile (16–20s and 9–12s respectively
at K=200/50). `_inverse_row_perm` and `_contig_groups` are present but negligible (<0.05s cum).

### variants mode (germline K=200, somatic K=200)

| rank | function | cumtime (germline / somatic) | tottime (germline / somatic) | kind |
|---|---|---|---|---|
| 1 | `Svar2Haps._reconstruct_variants` (`_svar2_haps.py:527`) | 0.973s / 3.709s | 0.041s / 0.270s | pure-Python dispatcher, thin |
| 2 | `decode_variants_from_svar2_readbound` (built-in) | 0.542s / 2.010s | 0.542s / 2.010s | **FFI wrapper** (Rust) — largest single tottime consumer in this mode |
| 3 | `Svar2Haps._gather_inputs` (`_svar2_haps.py:632`) | 0.104s / 0.809s | 0.003s / 0.033s | pure-Python — scales up sharply with cohort size (germline 3202 vs. somatic 16007 samples); dominated by `numpy.memmap.__getitem__` (0.696s tottime at somatic scale) |
| 4 | `_ragged_arange_gather` (`_svar2_haps.py:80`) | 0.105s / 0.223s | 0.030s / 0.068s | pure-Python hot loop (same as haplotypes mode, smaller absolute magnitude here) |
| 5 | `_ragged_arange_gather_2level` (`_svar2_haps.py:101`) | 0.099s / 0.113s | 0.029s / 0.035s | pure-Python hot loop, 2-level variant used for variants mode |
| — | `_contig_groups` (`_svar2_haps.py:619`) | 0.015s / 0.065s | 0.002s / 0.006s | pure-Python, small but present as expected |

**Matches expectation:** the ranking generally confirms the brief's predicted list
(`_reconstruct_variants`, `_gather_inputs`, `_ragged_arange_gather`/`_2level`, `_contig_groups`),
with one addition — `decode_variants_from_svar2_readbound` (Rust FFI) is actually the single
largest *tottime* contributor in variants mode, ahead of any pure-Python gvl function. Also
notable: `_gather_inputs` is proportionally much more expensive for variants (rank 3, scales
with cohort/memmap-read size) than for haplotypes (negligible), the inverse of the brief's
haplotypes-mode expectation.

### Cross-cutting observation

In **both** modes, the true CPU-bound hot path is the pure-Python `_ragged_arange_gather`
(and its `_2level` sibling) issuing repeated small `numpy.arange`/`ndarray.repeat` calls in a
Python loop — this is the clearest "vectorize or push to Rust" candidate surfaced by this
baseline. The Rust FFI calls (`reconstruct_haplotypes_from_svar2_readbound`,
`hap_diffs_from_svar2_readbound`, `decode_variants_from_svar2_readbound`) are thin
single-hop wrappers whose cost is compiled-code execution time, not Python overhead, and are
out of scope for Python-layer optimization.
