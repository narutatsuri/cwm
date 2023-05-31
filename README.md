# Contrastive Word Model

## Training
First, ensure the GCC compiler is installed (used for compiling training code). Download `text.txt` from the [Wikipedia Corpus Dump](https://drive.google.com/a/illinois.edu/uc?export=download&id=1fT1GxBMXEItf2NtNMjdXhA61HPecPO98) and place the file in `dataset/`. 

To train CWM, run the following in the home directory:
```
./train.sh [margin_size] [lambda_1] [lambda_2] [learning_rate] [iterations] [dimensions] [window_size] [negative_samples] [threads]
```
Additional parameter changes can be made directly in `train.sh`.

## Evaluation
Evaluation scripts are written in Python 3. Install necessary libraries using the `requirements.txt` file using:
```
pip install -r requirements.txt
```

Downloads the [BATS dataset](http://vecto.space/projects/BATS/) and place the folder `BATS_3.0` under `dataset/`. PCS and MSM scores can be evaluated interactively through the Jupyter Notebook `evaluation/evaluation.ipynb`.