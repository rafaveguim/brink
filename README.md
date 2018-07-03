Brink
=====



## Utilities



### N x N image distance

You can leverage multiprocessing by passing `--parallel` and `--num_processes`:

```shell
./similarity_matrix.py --num_processes=4 --rotation_invariant --parallel data/pandey/*.png > output/pandey_ssim_similarity_rotation_invariant_.csv
```

For more options `./similarity_matrix.py --help`



### Download Pandey et al.'s scatterplots

```shell
cd data/pandey
make all
```

### Pairwise image comparisons

The following command calculates SSIM and NRMSE between imageA and the rest of the images.

```
./compare_images.py imageA imageB imageC ...
```

