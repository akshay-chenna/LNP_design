import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from pathlib import Path
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import data, featurizers, models, nn
import argparse

parser = argparse.ArgumentParser(description="Run MPNN using regression with foundation model on data cut short")
parser.add_argument('-d','--depth', type=int)
parser.add_argument('-m','--hiddendim', type=int)
parser.add_argument('-n','--nlayers', type=int)
parser.add_argument('-p','--dropout', type=float)
parser.add_argument('-e','--epochs', type=int)
parser.add_argument('-r','--run', type=int)

args = parser.parse_args()

name = 'mean_regression_chem'+ '_d' + str(args.depth) +'_h'+ str(args.hiddendim)+ '_n' + str(args.nlayers) + '_p' + str(args.dropout) + '_e' + str(args.epochs) + '_r' + str(args.run)

input_path = Path.cwd().parent / 'pulmonary' / 'data' / 'all_data.csv'
df = pd.read_csv(input_path)

smiles_col = 'smiles'
target_col = ['quantified_delivery']

df = df.loc[df.groupby(smiles_col)['quantified_delivery'].idxmax()].reset_index(drop=True)

df = df[~df['Experiment_ID'].isin(['Liu_Phospholipids', 'Zhou_dendrimer', 'Akinc_Michael_addition'])].reset_index(drop=True)
smis = df.loc[:,smiles_col].values
Ys = df.loc[:, target_col].values

plt.figure()
plt.plot(np.sort(Ys,axis=0))
plt.title(name)
plt.savefig(name+'_target_distribution.png')

datapoints = [data.MoleculeDatapoint.from_smi(smi,y) for smi,y in zip(smis, Ys)]
mols = [d.mol for d in datapoints]

train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1), seed=42)
train_data, val_data, test_data = data.split_data_by_indices(datapoints, train_indices, val_indices, test_indices)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
train_dset = data.MoleculeDataset(train_data[0], featurizer=featurizer)
scaler = train_dset.normalize_targets()
val_dset = data.MoleculeDataset(val_data[0], featurizer=featurizer)
val_dset.normalize_targets(scaler)
test_dset = data.MoleculeDataset(test_data[0], featurizer=featurizer)

train_loader = data.build_dataloader(train_dset, batch_size=128, num_workers=0, shuffle=True)
val_loader = data.build_dataloader(val_dset, batch_size=128, num_workers=0, shuffle=False)
test_loader = data.build_dataloader(test_dset, batch_size=128, num_workers=0, shuffle=False)

chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
mp.load_state_dict(chemeleon_mp['state_dict'])
agg = nn.agg.MeanAggregation()

output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
predictor = nn.predictors.RegressionFFN(input_dim=mp.output_dim, hidden_dim=args.hiddendim, n_layers=args.nlayers, dropout=args.dropout, activation='relu',output_transform=output_transform)
batch_norm = True
metrics = [nn.MSE(), nn.MAE(), nn.RMSE(), nn.R2Score()]
mpnn = models.MPNN(mp, agg, predictor, batch_norm, metrics)

checkpointing = ModelCheckpoint(name, "best-{epoch}-{val_loss:.2f}", "val_loss", mode="min", save_last=True)
trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=[0],
    max_epochs=args.epochs,
    callbacks=[checkpointing]
)
trainer.fit(mpnn, train_loader, val_loader)

test_preds = np.concatenate(trainer.predict(mpnn,test_loader),axis=0).flatten()
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
plt.title(name)
plt.legend(loc="upper left")
plt.savefig(name+'_test_predictions.png')

