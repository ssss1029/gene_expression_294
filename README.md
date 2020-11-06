# DeepChrome Exploration

## Dataset
Download this from https://zenodo.org/record/26522781:
```
$ wget https://zenodo.org/record/2652278/files/data.tar.gz?download=1
```

## Dataloading
Sanity check the code with the following command:
```
$ python3 -m dataloading.DeepChrome
```

## Model
Sanity check the model with the following command:
```
$ python3 -m models.DeepChrome
```

## Running
Training and validation datasets are defined as glob strings.

The following are all valid ways to specify training datasets. You can do the same thing for `--globstr-val` as well.

Train on cell E123:
```
$ python3 main.py \
    --globstr-train="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E123/classification/train.csv" \
    --globstr-val="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E123/classification/valid.csv"
```

Train on {E118,E120,E123,E128}:
```
$ python3 main.py \
    --globstr-train="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E118/classification/train.csv" \
    --globstr-train="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E120/classification/train.csv" \
    --globstr-train="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E123/classification/train.csv" \
    --globstr-train="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E128/classification/train.csv" \
    --globstr-val="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E123/classification/valid.csv"
```

Train on {E118,E120E123,E128}:
**NOTE: This is broken but can be fixed easily lmk if it is useful.**
```
$ python3 main.py \
    --globstr-train="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/{E118,E120,E123,E128}/classification/train.csv" \
    --globstr-val="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E123/classification/valid.csv"
```

Train on everything:
```
$ python3 main.py \
    --globstr-train="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/*/classification/train.csv" \
    --globstr-val="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E123/classification/valid.csv"
```
