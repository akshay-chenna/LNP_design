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

parser = argparse.ArgumentParser(description="Run MPNN using classification with foundation model on data cut short")
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

Ys[Ys > 1.75] = 10
Ys[Ys <= 1.75] = 0
Ys[Ys > 1.75] = 1

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

mp = nn.BondMessagePassing(depth=args.depth, activation=nn.utils.Activation.RELU)
agg = nn.agg.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
predictor = nn.predictors.BinaryClassificationFFN(input_dim=mp.output_dim, hidden_dim=args.hiddendim, n_layers=args.nlayers, dropout=args.dropout, activation='relu',output_transform=output_transform)
batch_norm = True
metrics = None # AUCROC by default
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

from sklearn.metrics import precision_recall_curve, auc, roc_curve

fpr, tpr, _ = roc_curve(test_vals, test_preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal dashed line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(name + '_roc_curve.png')
