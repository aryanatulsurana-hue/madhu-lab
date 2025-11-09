import pandas as pd
import numpy as np
import os
from Bio.PDB import PDBParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product

# ========== Load Data ==========
df = pd.read_csv("binding_site_no_with_pdb_id.csv")  # Replace with actual path to your CSV

# ========== Split Binding Sites ==========
unique_sites = df["binding_site_number"].unique()
train_sites, test_sites = train_test_split(unique_sites, test_size=0.2, random_state=42)

# ========== Extract CA Coordinates ==========
def get_ca_coords(pdb_path, chain_id, residue_number):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    for model in structure:
        chain = model[chain_id]
        for residue in chain:
            if residue.get_id()[1] == residue_number:
                if "CA" in residue:
                    return residue["CA"].get_coord()
    return None

# ========== Feature Extraction ==========
def extract_features(df, binding_sites):
    features = []
    labels = []

    for site in binding_sites:
        subset = df[df["binding_site_number"] == site]
        if subset.empty:
            continue

        coords_apo = []
        coords_transformed_apo = []
        
        for _, row in subset.iterrows():
            # Get file paths for original Apo and transformed Apo files
            pdb_file_apo = os.path.join('arelecant_pdbs', f"{row['uniprot_id']}_Apo_{row['apo_pdb']}.pdb")
            pdb_file_transformed_apo = os.path.join('transformed_apo_coords', f"{row['uniprot_id']}_Apo_transformed_{row['Holo_Chain']}{row['lig_num']}.pdb")
            
            if not os.path.exists(pdb_file_apo) or not os.path.exists(pdb_file_transformed_apo):
                continue

            # Get coordinates from the original Apo and Apo Transformed structures
            coord_apo = get_ca_coords(pdb_file_apo, row['chain_id'], row['residue_number'])
            coord_transformed_apo = get_ca_coords(pdb_file_transformed_apo, row['chain_id'], row['residue_number'])

            if coord_apo is not None and coord_transformed_apo is not None:
                coords_apo.append(coord_apo)
                coords_transformed_apo.append(coord_transformed_apo)
            else:
                coords_apo.append(np.array([np.nan, np.nan, np.nan]))
                coords_transformed_apo.append(np.array([np.nan, np.nan, np.nan]))

        coords_apo = np.array(coords_apo)
        coords_transformed_apo = np.array(coords_transformed_apo)

        if np.isnan(coords_apo).any() or np.isnan(coords_transformed_apo).any():
            continue

        # Calculate center of original apo structure
        center_apo = coords_apo.mean(axis=0)
        # Calculate movement (difference between transformed and original apo)
        movement = coords_transformed_apo - coords_apo

        # Store features (secondary structure encoding) and labels (movement)
        for i, (_, row) in enumerate(subset.iterrows()):
            ss = row["secondary_structure"]
            features.append([1 if ss == "HELIX" else 0, 1 if ss == "SHEET" else 0, 1 if ss == "LOOP" else 0])
            labels.append(movement[i])

    return np.array(features), np.array(labels)

# ========== Neural Network ==========
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.model(x)

# ========== Main Loop Over Weights ==========
results = []
weights_range = np.arange(0, 1.05, 0.05)

X_train_raw, y_train = extract_features(df, train_sites)
X_test_raw, y_test = extract_features(df, test_sites)

for wh, we, wc in product(weights_range, repeat=3):
    weight_array = np.array([wh, we, wc])
    X_train = X_train_raw * weight_array
    X_test = X_test_raw * weight_array

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()
        mae = mean_absolute_error(y_test, preds)
        results.append((wh, we, wc, mae))

# ========== Heatmap for Best MAE with H vs C Weights ==========
best_mae_map = {}
for wh, we, wc, mae in results:
    key = (round(wh, 2), round(wc, 2))
    if key not in best_mae_map or mae < best_mae_map[key]:
        best_mae_map[key] = mae

wh_vals = sorted(set(k[0] for k in best_mae_map))
wc_vals = sorted(set(k[1] for k in best_mae_map))
mae_matrix = np.array([[best_mae_map.get((wh, wc), np.nan) for wc in wc_vals] for wh in wh_vals])

plt.figure(figsize=(10, 8))
sns.heatmap(mae_matrix, xticklabels=wc_vals, yticklabels=wh_vals, cmap="magma", annot=True, fmt=".2f")
plt.xlabel("Weight for Coil (C)")
plt.ylabel("Weight for Helix (H)")
plt.title("Best MAE Heatmap (Best Sheet Weight for Each H–C Pair)")
plt.tight_layout()
plt.show()
s
