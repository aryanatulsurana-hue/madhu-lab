import pandas as pd

# Load the two CSV files
df1 = pd.read_csv("merged_output_final_local_1_global_3.csv")
df2 = pd.read_csv("with_binding_site_numbers.csv")

# Merge on common keys
merged_df = pd.merge(df1, df2, on=["uniprot_id", "lig_name", "lig_num"])

# Keep only the required columns
final_df = merged_df[["uniprot_id", "lig_name", "lig_num", "binding_site_number", "apo_pdb", "Holo_Chain","secondary_structure"]]

# Save to a new CSV
final_df.to_csv("binding_site_no_with_pdb_id.csv", index=False)
