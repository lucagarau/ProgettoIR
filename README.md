# Information Retrieval Project
## DHMAE: Reproducibility Environment Setup and Execution Guide

In this project, we reproduce the DHMAE model proposed by [Zhao et al. (SIGIR 2024)](https://doi.org/10.1145/3626772.3657699)  for group recommendation.
We implemented and validated the model using the original datasets provided by the authors, replicating the experimental pipeline described in the paper. The results obtained in terms of evaluation metrics are consistent with those reported, confirming the robustness of the proposed framework.
The code was executed in PyTorch with GPU support, strictly following the hyperparameter settings and evaluation strategy adopted by the authors.




## 1. Create the Python environment

```bash
conda create -n dhmae_replication python=3.9 -y
conda activate dhmae_replication
```

## 2. Install required dependencies

Make sure the file `requirements.txt` is in the project root. Then run:

```bash
pip install -r requirements.txt
```

## 3. Launching the experiments

### A. Run `main.py` to replicate authors’ experiments from scratch

This script trains the model on the selected dataset and evaluates it using the original evaluation metrics (HR, NDCG, MRR).

Example:
```bash
python -u main.py --dataset=Mafengwo --num_negatives=10 --num_enc_layers=2 --num_dec_layers=3 --sce_alpha=2 --drop_ratio=0.1 --epoch=200 --device=cuda:0
```




### B. Run `main_with_loaded_model.py` to load pre-trained model and compute extended metrics

This script loads saved model weights and computes standard and fine-grained metrics for both users and groups:

- **HR@K**, **NDCG@K**, **MRR@K**
- Breakdown for **popular** (≥5 interactions) and **non-popular** (<5)
- Separate metrics for users and groups
- Prints statistics on item popularity

```bash
python -u main_with_trained_model.py --dataset=Mafengwo --num_negatives=10 --num_enc_layers=2 --num_dec_layers=3 --sce_alpha=2 --drop_ratio=0.1 --epoch=200 --device=cuda:0
```

Ensure that the corresponding model weights are available under:

```
./saved_models/Mafengwo/model_last.pth
```

Replace `Mafengwo` as needed with the dataset of interest.

---

## 4. Notes

- Both scripts assume the dataset structure is located under `./data/<dataset_name>/`.
- Make sure GPU is available if using `--device cuda:0`; otherwise, set `--device cpu`.




