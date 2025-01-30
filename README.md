# GNN-Grandmaster

## Installation

You need `conda` (or `mamba`) to be installed. Then to set up the `gnn-gm` environment :

```shell
conda env create -f environment.yml
conda activate gnn-gm
``` 

If `environment.yml` file is modified, just run the following command to update it:

```shell
conda activate gnn-gm
conda env update --file environment.yml --prune
```  

# Generating Dataset

To generate the dataset, download the Lichess database of **evaluation**
(See [HERE](https://database.lichess.org/#evals)). Next, extract the file in a directory.
Then generate the dataset with:

```shell
conda activate gnn-gm
python preprocess.py --size SIZE LICHESS_FILE OUTPUT_DIRECTORY
```

`SIZE` controls the size of the dataset. You can play with the `--chunk-size` option, it controls the size of the chunk
in which the dataset is split.