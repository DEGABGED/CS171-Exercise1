## Compiling

```
nvcc -o a <filename>.cu; .\a.exe
```

## Output

```
GeForce GTX 745
Major revision number:         5
Minor revision number:         0
Total global memory:           0 bytes
Number of multiprocessors:     3
Total shared memory per block: 49152
Total registers per block:     65536
Warp size:                     32
Maximum memory pitch:          2147483647
Total constant memory:         65536
Maximum threads per block:     1024
Maximum threads per dimension: 1024,1024,64

#0:     0.0086  0.0060  0.0060  0.0061  0.0060  0.0060  0.0062  0.0068  0.0059  0.0060  Ave: 0.0064
#1:     0.0179  0.0180  0.0182  0.0180  0.0181  0.0187  0.0180  0.0178  0.0179  0.0183  Ave: 0.0187
#2:     0.0251  0.0261  0.0249  0.0262  0.0251  0.0253  0.0287  0.0283  0.0270  0.0279  Ave: 0.0283

Done!
```