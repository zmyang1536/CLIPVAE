### 1. Environment Setup
```bash
git clone https://github.com/zmyang1536/CLIPVAE.git
cd CLIPVAE

conda create -n clipvae python=3.9
conda activate clipvae

pip install -r requirements.txt
pip install -e .
```

### 2. Data Preparation
This project supports standard 10x Visium spatial transcriptomics data format. You need to prepare:
- Gene expression matrix file (`filtered_feature_bc_matrix.h5`)
- Spatial coordinates file (`spatial/tissue_positions_list.csv`)
- H&E histology image (`spatial/tissue_hires_image.png`)

We provide a one-click preprocessing script:
```bash
python data/preproces.py \
    --adata_path data/raw/your_data.h5ad \
    --image_path data/raw/tissue_hires_image.png \
    --output_dir data/processed \
    --n_top_genes 2000
```

Preprocessed data will be saved as `processed_adata.h5ad` and `processed_image.png`.

### 3. Model Training
Edit the `configs/train_config.yaml` file to modify data paths and hyperparameters, then run:
```bash
# Single GPU training
python scripts/train.py --config configs/train_config.yaml

# Multi-GPU distributed training
torchrun --nproc_per_node=4 scripts/train.py --config configs/train_config.yaml
```

During training, TensorBoard logs will be saved in the `outputs/[experiment_name]/logs/` directory. You can view them with:
```bash
tensorboard --logdir outputs/[experiment_name]/logs/
```

### 4. Model Evaluation
After training completes, evaluate model performance using the best checkpoint:
```bash
python scripts/evaluate.py \
    --checkpoint outputs/clipvae_human_breast_cancer_v1/checkpoints/best_model.pth \
    --adata_path data/processed/processed_adata.h5ad \
    --image_path data/processed/processed_image.png \
    --output_dir outputs/evaluation
```

Evaluation results will include loss values and clustering metrics, and will be saved as `evaluation_results.txt`.
