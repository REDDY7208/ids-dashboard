"""Extract real attack samples from CICIDS2017 dataset"""
import pandas as pd
import numpy as np

print("=" * 70)
print("Extracting REAL Attack Samples from CICIDS2017 Dataset")
print("=" * 70)

# Dataset files
dataset_files = {
    'Benign': 'Datasets/Datasets/cic-ids/Benign-Monday-no-metadata.parquet',
    'Bot': 'Datasets/Datasets/cic-ids/Botnet-Friday-no-metadata.parquet',
    'Brute Force': 'Datasets/Datasets/cic-ids/Bruteforce-Tuesday-no-metadata.parquet',
    'DDoS': 'Datasets/Datasets/cic-ids/DDoS-Friday-no-metadata.parquet',
    'DoS': 'Datasets/Datasets/cic-ids/DoS-Wednesday-no-metadata.parquet',
    'Infiltration': 'Datasets/Datasets/cic-ids/Infiltration-Thursday-no-metadata.parquet',
    'PortScan': 'Datasets/Datasets/cic-ids/Portscan-Friday-no-metadata.parquet',
    'Web Attacks': 'Datasets/Datasets/cic-ids/WebAttacks-Thursday-no-metadata.parquet',
}

all_samples = []
samples_per_type = 5  # Get 5 real samples from each file

for attack_type, file_path in dataset_files.items():
    print(f"\nLoading {attack_type}...")
    try:
        df = pd.read_parquet(file_path)
        print(f"  Total records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
        # Get random samples
        if len(df) > samples_per_type:
            samples = df.sample(n=samples_per_type, random_state=42)
        else:
            samples = df.head(samples_per_type)
        
        all_samples.append(samples)
        print(f"  ✅ Extracted {len(samples)} samples")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Combine all samples
print("\n" + "=" * 70)
print("Combining samples...")
combined_df = pd.concat(all_samples, ignore_index=True)
print(f"✅ Total samples: {len(combined_df)}")
print(f"✅ Total features: {len(combined_df.columns)}")

# Save to CSV
output_file = 'real_attack_samples.csv'
combined_df.to_csv(output_file, index=False)
print(f"\n💾 Saved to: {output_file}")

# Show summary
print("\n" + "=" * 70)
print("Dataset Summary:")
print("=" * 70)
print(f"Total samples: {len(combined_df)}")
print(f"Total features: {len(combined_df.columns)}")
print(f"Samples per attack type: ~{samples_per_type}")
print("\n✅ Real attack samples extracted successfully!")
print("=" * 70)
