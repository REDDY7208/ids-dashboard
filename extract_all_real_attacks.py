"""Extract real samples for ALL 14 attack types from CICIDS2017"""
import pandas as pd
import pickle
import os

print("=" * 70)
print("Extracting REAL Samples for ALL 14 Attack Types")
print("=" * 70)

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    model_features = pickle.load(f)

# Load label encoder to see what attacks we need
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

target_attacks = label_encoder.classes_
print(f"\nTarget attack types ({len(target_attacks)}):")
for i, attack in enumerate(target_attacks, 1):
    print(f"   {i}. {attack}")

# CSV files from CICIDS2017
csv_files = [
    'Datasets/Datasets/3rd/Monday-WorkingHours.pcap_ISCX.csv',
    'Datasets/Datasets/3rd/Tuesday-WorkingHours.pcap_ISCX.csv',
    'Datasets/Datasets/3rd/Wednesday-workingHours.pcap_ISCX.csv',
    'Datasets/Datasets/3rd/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Datasets/Datasets/3rd/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Datasets/Datasets/3rd/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Datasets/Datasets/3rd/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Datasets/Datasets/3rd/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
]

all_samples = []
samples_per_attack = 3  # Get 3 samples per attack type

print("\n" + "=" * 70)
print("Scanning all CSV files for attack types...")
print("=" * 70)

for csv_file in csv_files:
    if not os.path.exists(csv_file):
        print(f"⚠️ File not found: {csv_file}")
        continue
    
    print(f"\nProcessing: {os.path.basename(csv_file)}")
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        print(f"  Records: {len(df):,}, Columns: {len(df.columns)}")
        
        # Check if Label column exists
        if 'Label' not in df.columns and ' Label' not in df.columns:
            print("  ⚠️ No Label column found")
            continue
        
        # Get label column name (might have space)
        label_col = 'Label' if 'Label' in df.columns else ' Label'
        
        # Get unique labels
        unique_labels = df[label_col].unique()
        print(f"  Attack types found: {len(unique_labels)}")
        
        for label in unique_labels:
            label_clean = str(label).strip()
            print(f"    - {label_clean}")
            
            # Get samples for this attack type
            attack_df = df[df[label_col] == label]
            
            if len(attack_df) > 0:
                # Get random samples
                n_samples = min(samples_per_attack, len(attack_df))
                samples = attack_df.sample(n=n_samples, random_state=42)
                
                # Extract only the 40 features the model needs
                extracted = pd.DataFrame()
                for feat in model_features:
                    if feat in samples.columns:
                        extracted[feat] = samples[feat].values
                    elif ' ' + feat in samples.columns:  # Try with space
                        extracted[feat] = samples[' ' + feat].values
                    else:
                        extracted[feat] = 0.0  # Fill missing
                
                # Add label
                extracted['Label'] = label_clean
                all_samples.append(extracted)
                print(f"      ✅ Extracted {len(extracted)} samples")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Combine all
if all_samples:
    print("\n" + "=" * 70)
    print("Combining all samples...")
    combined_df = pd.concat(all_samples, ignore_index=True)
    print(f"✅ Total samples: {len(combined_df)}")
    
    # Show distribution
    print(f"\n📊 Attack Type Distribution:")
    label_counts = combined_df['Label'].value_counts()
    for label, count in label_counts.items():
        print(f"   {label:35s}: {count} samples")
    
    # Save
    output_file = 'real_all_attacks.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   Unique attack types: {len(label_counts)}")
    print(f"   Features: 40 + Label")
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS! Real attack samples extracted!")
    print("=" * 70)
else:
    print("\n❌ No samples extracted")
