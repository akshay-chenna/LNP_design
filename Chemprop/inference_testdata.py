import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from pathlib import Path
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from chemprop import data, featurizers, models, nn
import argparse

parser = argparse.ArgumentParser(description="Inference on test data using MPNN with regression model")
parser.add_argument('-f','--file', type=str)
parser.add_argument('-n','--name', type=str)

args = parser.parse_args()

input_path = Path.cwd().parent / 'pulmonary' / 'data' / 'all_data.csv'
df = pd.read_csv(input_path)

smiles_col = 'smiles'
target_col = ['quantified_delivery']

df = df.loc[df.groupby(smiles_col)['quantified_delivery'].idxmax()].reset_index(drop=True)

df = df[~df['Experiment_ID'].isin(['Liu_Phospholipids', 'Zhou_dendrimer', 'Akinc_Michael_addition'])].reset_index(drop=True)

smis = df.loc[:,smiles_col].values
Ys = df.loc[:, target_col].values

datapoints = [data.MoleculeDatapoint.from_smi(smi,y) for smi,y in zip(smis, Ys)]
mols = [d.mol for d in datapoints]

train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1), seed=42)
train_data, val_data, test_data = data.split_data_by_indices(datapoints, train_indices, val_indices, test_indices)

x_data = []
for smi in smis:
	try:
		dp = data.MoleculeDatapoint.from_smi(smi)
		if dp is not None:
			x_data.append(dp)
	except:
		print(f"Invalid SMILES skipped: {smi}")
print(len(x_data))

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
test_dset = data.MoleculeDataset(test_data[0], featurizer=featurizer)
test_loader = data.build_dataloader(test_dset, shuffle=False)

mpnn = models.MPNN.load_from_checkpoint(args.file)

with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )
    test_preds = trainer.predict(mpnn, test_loader)
    
test_preds = np.concatenate(test_preds,axis=0).flatten()
test_vals = np.concatenate(test_loader.dataset.Y,axis=0)

from torchmetrics.regression import R2Score
import torch
r2score = R2Score()
print("Test R2 Score:")
print(r2score(torch.tensor(test_preds), torch.tensor(test_vals)))

from scipy.stats import pearsonr, spearmanr
pearson_corr, _ = pearsonr(test_vals, test_preds)
spearman_corr, _ = spearmanr(test_vals, test_preds)
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(test_vals, test_preds, label=f'Rho = {spearman_corr:.2f})')
plt.xlabel('Test -- True Values')
plt.ylabel('Test -- Predictions')
plt.legend(loc="upper left")
plt.title(args.name)
plt.savefig('best_'+args.name+'_test_predictions.png')
