import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
from Bio import PDB
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import itertools
import multiprocessing as mp
from tqdm import tqdm

# Define the neural network class
class ConformationNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=None):
        super(ConformationNN, self).__init__()
        if output_size is None:
            output_size = input_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Load secondary structure data
def load_secondary_structure(sec_struct_csv):
    sec_df = pd.read_csv(sec_struct_csv)
    return sec_df.set_index(["uniprot_id", "residue_number"], drop=False)

# Extract features from CSV and integrate secondary structure
def extract_features(data_csv, sec_struct_data, weights_dict):
    df = pd.read_csv(data_csv)
    features, labels, uniprot_ids, weights = [], [], [], []

    for _, row in df.iterrows():
        uniprot_id, residue_num = row["uniprot_id"], row["residue_number"]
        key = (uniprot_id, residue_num)
        if key in sec_struct_data.index:
            sec_struct = sec_struct_data.loc[key, "secondary_structure"]
            if isinstance(sec_struct, pd.Series):
                sec_struct = sec_struct.iloc[0]
        else:
            sec_struct = "?"

        if len(sec_struct) > 1:
            sec_struct = sec_struct[0]

        sec_struct_encoded = ord(sec_struct) if sec_struct != "?" else -1
        weight = weights_dict.get(sec_struct, 0.3)

        feature_vector = [row["rmsd"], sec_struct_encoded] + eval(row["coords_apo"])
        features.append(feature_vector)
        labels.append(eval(row["coords_transformed"]))
        uniprot_ids.append(uniprot_id)
        weights.append(weight)

    return np.array(features), np.array(labels), np.array(uniprot_ids), np.array(weights)

# Load data
sec_struct_csv = "CA_rmsd_less_than_6_cases_sec_struc.csv"
data_csv = "CA_rmsd_less_than_6_cases.csv"
sec_struct_data = load_secondary_structure(sec_struct_csv)

# Function to evaluate one weight combination
def evaluate_weights_with_progress(params):
    h_weight, e_weight, c_weight, idx, total = params

    # Skip if weights do not sum to 1
    if not np.isclose(h_weight + e_weight + c_weight, 1.0, atol=1e-6):
        return None

    weights_dict = {"H": h_weight, "E": e_weight, "C": c_weight, "?": 0.3}
    X, Y, uniprot_ids, weights = extract_features(data_csv, sec_struct_data, weights_dict)

    unique_ids = np.unique(uniprot_ids)
    np.random.seed(42)
    np.random.shuffle(unique_ids)
    train_ids = set(unique_ids[:int(0.8 * len(unique_ids))])
    test_ids = set(unique_ids[int(0.8 * len(unique_ids)):])

    train_mask = np.array([uid in train_ids for uid in uniprot_ids])
    test_mask = np.array([uid in test_ids for uid in uniprot_ids])

    X_train, X_test = X[train_mask], X[test_mask]
    Y_train, Y_test = Y[train_mask], Y[test_mask]
    weights_train, weights_test = weights[train_mask], weights[test_mask]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    Y_train = scaler_Y.fit_transform(Y_train)
    Y_test = scaler_Y.transform(Y_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    weights_train_tensor = torch.tensor(weights_train, dtype=torch.float32).unsqueeze(1).repeat(1, Y_train.shape[1])

    model = ConformationNN(X_train.shape[1], output_size=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def weighted_mse_loss(pred, target, weights):
        return torch.mean(weights * (pred - target) ** 2)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = weighted_mse_loss(outputs, Y_train_tensor, weights_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = scaler_Y.inverse_transform(predictions.numpy())
        actual = scaler_Y.inverse_transform(Y_test_tensor.numpy())
        mae = np.mean(np.abs(predictions - actual))
        r2 = r2_score(actual, predictions)
        diffs = np.linalg.norm(predictions - actual, axis=1)
        within_0_5 = np.sum(diffs < 0.5)
        within_1 = np.sum(diffs < 1.0)
        within_2 = np.sum(diffs < 2.0)

    with open("progress_output.txt", "a") as f:
        f.write(f"{idx+1}/{total} done: H={h_weight:.2f}, E={e_weight:.2f}, C={c_weight:.2f}, MAE={mae:.4f}\n")

    return (h_weight, e_weight, c_weight, mae, r2, within_0_5, within_1, within_2)

# Grid search with multiprocessing
steps = np.arange(0.05, 1.05, 0.05)
param_grid = [(h, e, c, idx, len(list(itertools.product(steps, steps, steps)))) for idx, (h, e, c) in enumerate(itertools.product(steps, steps, steps))]

with mp.Pool(processes=20) as pool:
    results_log = list(tqdm(pool.imap(evaluate_weights_with_progress, param_grid), total=len(param_grid)))

# Filter and sort top 5
results_log = [res for res in results_log if res is not None]
sorted_results = sorted(results_log, key=lambda x: x[3])  # Sort by MAE
top_5 = sorted_results[:5]

for result in top_5:
    h, e, c, mae, r2, w05, w1, w2 = result
    print(f"H={h:.2f}, E={e:.2f}, C={c:.2f}, MAE={mae:.4f}, R2={r2:.4f}, <0.5Å={w05}, <1Å={w1}, <2Å={w2}")

# Save results
results_df = pd.DataFrame(results_log, columns=["H_weight", "E_weight", "C_weight", "MAE", "R2", "<0.5A", "<1A", "<2A"])
results_df.to_csv("weight_search_results_uniprot_wise.csv", index=False)

# Heatmap for best H
best_h = top_5[0][0]
subset = results_df[results_df["H_weight"] == best_h]
pivot = subset.pivot(index="E_weight", columns="C_weight", values="MAE")

plt.figure(figsize=(10, 8))
pivot.index = [round(float(i), 1) for i in pivot.index]
pivot.columns = [round(float(i), 1) for i in pivot.columns]
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 6})
plt.title(f"MAE Heatmap (H_weight = {round(best_h, 1)})")
plt.xlabel("C_weight")
plt.ylabel("E_weight")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("mae_heatmap_uniprot_wise.png")
plt.show()
'''
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

# Load secondary structure data
def load_secondary_structure(sec_struct_csv):
    sec_df = pd.read_csv(sec_struct_csv)
    return sec_df.set_index(["uniprot_id", "residue_number"], drop=False)

# Extract features with weighted secondary structure
def extract_features(data_csv, sec_struct_data, weights_dict):
    df = pd.read_csv(data_csv)
    features, labels, weights = [], [], []

    for _, row in df.iterrows():
        uniprot_id, residue_num = row["uniprot_id"], row["residue_number"]
        key = (uniprot_id, residue_num)
        sec_struct = sec_struct_data.loc[key, "secondary_structure"] if key in sec_struct_data.index else "LOOP"

        # Use only the first character or default
        if isinstance(sec_struct, pd.Series):
            sec_struct = sec_struct.iloc[0]
        if len(sec_struct) > 1:
            sec_struct = sec_struct.strip().upper()

        weight = weights_dict.get(sec_struct, 0.3)

        feature_vector = [row["rmsd"], ord(sec_struct[0])] + eval(row["coords_apo"])
        features.append(feature_vector)
        labels.append(eval(row["coords_transformed"]))
        weights.append(weight)

    return np.array(features), np.array(labels), np.array(weights)

# Load data
sec_struct_csv = "CA_rmsd_less_than_6_cases_sec_struc.csv"
data_csv = "CA_rmsd_less_than_6_cases.csv"
sec_struct_data = load_secondary_structure(sec_struct_csv)

# Assign weights to structures
weights_dict = {
    "HELIX": 0.76,
    "SHEET": 0.48,
    "LOOP": 0.92
}

# Prepare data
X, Y, weights = extract_features(data_csv, sec_struct_data, weights_dict)
X_train, X_test, Y_train, Y_test, weights_train, weights_test = train_test_split(
    X, Y, weights, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Define model
class ConformationNN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(ConformationNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size - 2)  # predict only the coordinates
        )

    def forward(self, x):
        return self.model(x)

model = ConformationNN(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
weights_tensor = torch.tensor(weights_train, dtype=torch.float32).unsqueeze(1).repeat(1, Y_train.shape[1])

# Train
def weighted_mse(pred, target, weight):
    return torch.mean(weight * (pred - target) ** 2)

for epoch in range(100):
    optimizer.zero_grad()
    preds = model(X_train_tensor)
    loss = weighted_mse(preds, Y_train_tensor, weights_tensor)
    loss.backward()
    optimizer.step()

# Evaluate
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_preds = scaler_Y.inverse_transform(test_preds.numpy())
    actuals = scaler_Y.inverse_transform(Y_test_tensor.numpy())
    mae = np.mean(np.abs(test_preds - actuals))
    r2 = r2_score(actuals, test_preds)

    # Check how many predictions are within threshold
    threshold = 1.0
    abs_errors = np.abs(test_preds - actuals)
    within_thresh = (abs_errors < threshold).all(axis=1)
    count_within_thresh = np.sum(within_thresh)
    total = len(Y_test)

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"{count_within_thresh}/{total} predictions are within {threshold} Å")'''


