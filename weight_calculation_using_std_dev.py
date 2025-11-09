'''import pandas as pd

# Load secondary structure RMSD data
df = pd.read_csv("CA_rmsd_less_than_6_cases_sec_struc.csv")

# Keep only HELIX, SHEET, LOOP
relevant_structs = ["HELIX", "SHEET", "LOOP"]
df = df[df["secondary_structure"].isin(relevant_structs)]

# Calculate mean and std for each
stats = df.groupby("secondary_structure")["rmsd"].agg(["mean", "std"])

# Compute weights as inverse of std deviation (more consistent → higher weight)
epsilon = 1e-6
weights = 1 / (stats["std"] + epsilon)

# Optionally rescale to a more intuitive range like [0.3, 1.0] if needed
min_w, max_w = 0.3, 1.0
scaled_weights = (weights - weights.min()) / (weights.max() - weights.min()) * (max_w - min_w) + min_w

# Final weights
proposed_weights = scaled_weights.to_dict()

print("📊 Proposed weights (scaled 0.3–1.0) based on RMSD std deviation:")
for struct, weight in proposed_weights.items():
    print(f"{struct}: {weight:.2f}")'''
import pandas as pd

# Load the secondary structure RMSD data
df = pd.read_csv("CA_rmsd_less_than_6_cases_sec_struc.csv")

# Filter for relevant secondary structures
df = df[df["secondary_structure"].isin(["HELIX", "SHEET", "LOOP"])]

# Group by secondary structure and calculate mean and std deviation
stats = df.groupby("secondary_structure")["rmsd"].agg(["mean", "std"])

# Print the results
print("📊 RMSD statistics by secondary structure:")
print(stats)

