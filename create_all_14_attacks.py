"""Create CSV with ALL 14 attack types - 3 samples each = 42 total samples"""
import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility

# Feature names (40 features)
feature_names = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
    'Max Packet Length', 'Packet Length Mean'
]

print("=" * 70)
print("Creating CSV with ALL 14 Attack Types")
print("=" * 70)

attack_patterns = []

# 1. BENIGN - Normal traffic (3 samples)
print("1. Benign (Normal Traffic) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(80000, 150000),
        'Total Fwd Packets': np.random.uniform(8, 20),
        'Total Backward Packets': np.random.uniform(8, 20),
        'Total Length of Fwd Packets': np.random.uniform(4000, 8000),
        'Total Length of Bwd Packets': np.random.uniform(3000, 6000),
        'Fwd Packet Length Max': np.random.uniform(500, 700),
        'Fwd Packet Length Min': np.random.uniform(40, 80),
        'Fwd Packet Length Mean': np.random.uniform(450, 600),
        'Fwd Packet Length Std': np.random.uniform(100, 180),
        'Flow Bytes/s': np.random.uniform(60000, 120000),
        'Flow Packets/s': np.random.uniform(120, 250),
        'Fwd PSH Flags': 0,
        'Bwd PSH Flags': 0,
    })

# 2. BOT - Botnet traffic (3 samples)
print("2. Bot (Botnet Traffic) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(40000, 120000),
        'Total Fwd Packets': np.random.uniform(80, 250),
        'Total Backward Packets': np.random.uniform(80, 250),
        'Total Length of Fwd Packets': np.random.uniform(15000, 50000),
        'Total Length of Bwd Packets': np.random.uniform(15000, 50000),
        'Fwd Packet Length Max': np.random.uniform(600, 900),
        'Fwd Packet Length Min': np.random.uniform(120, 250),
        'Fwd Packet Length Mean': np.random.uniform(350, 550),
        'Fwd Packet Length Std': np.random.uniform(40, 90),  # Low variance (automated)
        'Flow Bytes/s': np.random.uniform(100000, 250000),
        'Flow Packets/s': np.random.uniform(200, 500),
        'Fwd PSH Flags': np.random.uniform(15, 40),
        'Bwd PSH Flags': np.random.uniform(15, 40),
    })

# 3. DDoS - Distributed Denial of Service (3 samples)
print("3. DDoS (Distributed Denial of Service) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(800, 4000),  # Very short
        'Total Fwd Packets': np.random.uniform(1500, 6000),  # Very high
        'Total Backward Packets': np.random.uniform(0, 15),  # Very low
        'Total Length of Fwd Packets': np.random.uniform(80000, 250000),
        'Total Length of Bwd Packets': np.random.uniform(0, 1500),
        'Fwd Packet Length Max': np.random.uniform(120, 250),
        'Fwd Packet Length Min': np.random.uniform(40, 70),
        'Fwd Packet Length Mean': np.random.uniform(60, 120),
        'Fwd Packet Length Std': np.random.uniform(15, 40),
        'Flow Bytes/s': np.random.uniform(800000, 3000000),  # Very high
        'Flow Packets/s': np.random.uniform(8000, 25000),  # Very high
        'Fwd PSH Flags': 0,
        'Bwd PSH Flags': 0,
    })

# 4. DoS GoldenEye - HTTP DoS (3 samples)
print("4. DoS GoldenEye (HTTP DoS) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(50000, 200000),
        'Total Fwd Packets': np.random.uniform(200, 800),
        'Total Backward Packets': np.random.uniform(20, 100),
        'Total Length of Fwd Packets': np.random.uniform(30000, 120000),
        'Total Length of Bwd Packets': np.random.uniform(2000, 10000),
        'Fwd Packet Length Max': np.random.uniform(400, 700),
        'Fwd Packet Length Min': np.random.uniform(80, 150),
        'Fwd Packet Length Mean': np.random.uniform(200, 400),
        'Fwd Packet Length Std': np.random.uniform(80, 180),
        'Flow Bytes/s': np.random.uniform(150000, 400000),
        'Flow Packets/s': np.random.uniform(500, 2000),
        'Fwd PSH Flags': np.random.uniform(30, 100),
        'Bwd PSH Flags': np.random.uniform(5, 20),
    })

# 5. DoS Hulk - HTTP DoS (3 samples)
print("5. DoS Hulk (HTTP DoS) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(30000, 150000),
        'Total Fwd Packets': np.random.uniform(300, 1000),
        'Total Backward Packets': np.random.uniform(30, 120),
        'Total Length of Fwd Packets': np.random.uniform(50000, 180000),
        'Total Length of Bwd Packets': np.random.uniform(3000, 15000),
        'Fwd Packet Length Max': np.random.uniform(500, 900),
        'Fwd Packet Length Min': np.random.uniform(100, 200),
        'Fwd Packet Length Mean': np.random.uniform(250, 500),
        'Fwd Packet Length Std': np.random.uniform(100, 220),
        'Flow Bytes/s': np.random.uniform(200000, 500000),
        'Flow Packets/s': np.random.uniform(800, 3000),
        'Fwd PSH Flags': np.random.uniform(40, 120),
        'Bwd PSH Flags': np.random.uniform(8, 30),
    })

# 6. DoS Slowhttptest - Slow HTTP DoS (3 samples)
print("6. DoS Slowhttptest (Slow HTTP DoS) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(150000, 600000),  # Very long
        'Total Fwd Packets': np.random.uniform(150, 600),
        'Total Backward Packets': np.random.uniform(15, 80),
        'Total Length of Fwd Packets': np.random.uniform(8000, 30000),
        'Total Length of Bwd Packets': np.random.uniform(800, 4000),
        'Fwd Packet Length Max': np.random.uniform(150, 400),
        'Fwd Packet Length Min': np.random.uniform(40, 100),
        'Fwd Packet Length Mean': np.random.uniform(80, 200),
        'Fwd Packet Length Std': np.random.uniform(30, 80),
        'Flow Bytes/s': np.random.uniform(8000, 30000),  # Low rate
        'Flow Packets/s': np.random.uniform(20, 80),  # Low rate
        'Fwd PSH Flags': np.random.uniform(25, 80),
        'Bwd PSH Flags': np.random.uniform(2, 15),
    })

# 7. DoS slowloris - Slow connection DoS (3 samples)
print("7. DoS slowloris (Slow Connection DoS) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(120000, 550000),  # Very long
        'Total Fwd Packets': np.random.uniform(120, 550),
        'Total Backward Packets': np.random.uniform(12, 60),
        'Total Length of Fwd Packets': np.random.uniform(6000, 25000),
        'Total Length of Bwd Packets': np.random.uniform(600, 3000),
        'Fwd Packet Length Max': np.random.uniform(120, 350),
        'Fwd Packet Length Min': np.random.uniform(40, 90),
        'Fwd Packet Length Mean': np.random.uniform(60, 180),
        'Fwd Packet Length Std': np.random.uniform(25, 70),
        'Flow Bytes/s': np.random.uniform(6000, 25000),  # Low rate
        'Flow Packets/s': np.random.uniform(15, 70),  # Low rate
        'Fwd PSH Flags': np.random.uniform(30, 120),
        'Bwd PSH Flags': 0,
    })

# 8. FTP-Patator - FTP Brute Force (3 samples)
print("8. FTP-Patator (FTP Brute Force) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(8000, 30000),
        'Total Fwd Packets': np.random.uniform(15, 40),
        'Total Backward Packets': np.random.uniform(15, 40),
        'Total Length of Fwd Packets': np.random.uniform(2500, 6000),
        'Total Length of Bwd Packets': np.random.uniform(1500, 4000),
        'Fwd Packet Length Max': np.random.uniform(250, 450),
        'Fwd Packet Length Min': np.random.uniform(60, 120),
        'Fwd Packet Length Mean': np.random.uniform(180, 350),
        'Fwd Packet Length Std': np.random.uniform(60, 160),
        'Flow Bytes/s': np.random.uniform(60000, 180000),
        'Flow Packets/s': np.random.uniform(150, 400),
        'Fwd PSH Flags': np.random.uniform(3, 15),
        'Bwd PSH Flags': np.random.uniform(3, 15),
    })

# 9. Infiltration - Network infiltration (3 samples)
print("9. Infiltration (Network Infiltration) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(60000, 250000),
        'Total Fwd Packets': np.random.uniform(40, 150),
        'Total Backward Packets': np.random.uniform(40, 150),
        'Total Length of Fwd Packets': np.random.uniform(8000, 35000),
        'Total Length of Bwd Packets': np.random.uniform(8000, 35000),
        'Fwd Packet Length Max': np.random.uniform(600, 1000),
        'Fwd Packet Length Min': np.random.uniform(100, 250),
        'Fwd Packet Length Mean': np.random.uniform(300, 600),
        'Fwd Packet Length Std': np.random.uniform(120, 280),
        'Flow Bytes/s': np.random.uniform(80000, 220000),
        'Flow Packets/s': np.random.uniform(180, 450),
        'Fwd PSH Flags': np.random.uniform(8, 30),
        'Bwd PSH Flags': np.random.uniform(8, 30),
    })

# 10. PortScan - Port scanning (3 samples)
print("10. PortScan (Port Scanning) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(150, 1200),  # Very short
        'Total Fwd Packets': np.random.uniform(1, 4),  # Very few
        'Total Backward Packets': np.random.uniform(0, 3),  # Very few
        'Total Length of Fwd Packets': np.random.uniform(50, 150),  # Very small
        'Total Length of Bwd Packets': np.random.uniform(0, 100),  # Very small
        'Fwd Packet Length Max': np.random.uniform(50, 90),
        'Fwd Packet Length Min': np.random.uniform(40, 65),
        'Fwd Packet Length Mean': np.random.uniform(45, 70),
        'Fwd Packet Length Std': np.random.uniform(2, 15),
        'Flow Bytes/s': np.random.uniform(1500, 6000),
        'Flow Packets/s': np.random.uniform(15, 60),
        'Fwd PSH Flags': 0,
        'Bwd PSH Flags': 0,
    })

# 11. SSH-Patator - SSH Brute Force (3 samples)
print("11. SSH-Patator (SSH Brute Force) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(6000, 25000),
        'Total Fwd Packets': np.random.uniform(12, 35),
        'Total Backward Packets': np.random.uniform(12, 35),
        'Total Length of Fwd Packets': np.random.uniform(2000, 5500),
        'Total Length of Bwd Packets': np.random.uniform(1200, 3500),
        'Fwd Packet Length Max': np.random.uniform(220, 420),
        'Fwd Packet Length Min': np.random.uniform(55, 115),
        'Fwd Packet Length Mean': np.random.uniform(160, 330),
        'Fwd Packet Length Std': np.random.uniform(55, 150),
        'Flow Bytes/s': np.random.uniform(55000, 170000),
        'Flow Packets/s': np.random.uniform(140, 380),
        'Fwd PSH Flags': np.random.uniform(2, 12),
        'Bwd PSH Flags': np.random.uniform(2, 12),
    })

# 12. Web Attack - Brute Force (3 samples)
print("12. Web Attack - Brute Force - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(15000, 60000),
        'Total Fwd Packets': np.random.uniform(30, 120),
        'Total Backward Packets': np.random.uniform(30, 120),
        'Total Length of Fwd Packets': np.random.uniform(12000, 55000),
        'Total Length of Bwd Packets': np.random.uniform(6000, 25000),
        'Fwd Packet Length Max': np.random.uniform(1200, 2200),
        'Fwd Packet Length Min': np.random.uniform(150, 350),
        'Fwd Packet Length Mean': np.random.uniform(600, 1200),
        'Fwd Packet Length Std': np.random.uniform(250, 600),
        'Flow Bytes/s': np.random.uniform(120000, 350000),
        'Flow Packets/s': np.random.uniform(250, 600),
        'Fwd PSH Flags': np.random.uniform(8, 25),
        'Bwd PSH Flags': np.random.uniform(8, 25),
    })

# 13. Web Attack - SQL Injection (3 samples)
print("13. Web Attack - SQL Injection - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(12000, 55000),
        'Total Fwd Packets': np.random.uniform(25, 110),
        'Total Backward Packets': np.random.uniform(25, 110),
        'Total Length of Fwd Packets': np.random.uniform(15000, 65000),  # Large payloads
        'Total Length of Bwd Packets': np.random.uniform(7000, 30000),
        'Fwd Packet Length Max': np.random.uniform(1500, 2500),  # Large packets
        'Fwd Packet Length Min': np.random.uniform(180, 380),
        'Fwd Packet Length Mean': np.random.uniform(700, 1400),
        'Fwd Packet Length Std': np.random.uniform(280, 650),
        'Flow Bytes/s': np.random.uniform(140000, 380000),
        'Flow Packets/s': np.random.uniform(280, 650),
        'Fwd PSH Flags': np.random.uniform(6, 22),
        'Bwd PSH Flags': np.random.uniform(6, 22),
    })

# 14. Web Attack - XSS (Cross-Site Scripting) (3 samples)
print("14. Web Attack - XSS (Cross-Site Scripting) - 3 samples")
for i in range(3):
    attack_patterns.append({
        'Flow Duration': np.random.uniform(10000, 50000),
        'Total Fwd Packets': np.random.uniform(22, 100),
        'Total Backward Packets': np.random.uniform(22, 100),
        'Total Length of Fwd Packets': np.random.uniform(13000, 60000),
        'Total Length of Bwd Packets': np.random.uniform(6500, 28000),
        'Fwd Packet Length Max': np.random.uniform(1400, 2400),
        'Fwd Packet Length Min': np.random.uniform(170, 370),
        'Fwd Packet Length Mean': np.random.uniform(650, 1350),
        'Fwd Packet Length Std': np.random.uniform(270, 640),
        'Flow Bytes/s': np.random.uniform(130000, 370000),
        'Flow Packets/s': np.random.uniform(270, 640),
        'Fwd PSH Flags': np.random.uniform(5, 20),
        'Bwd PSH Flags': np.random.uniform(5, 20),
    })

print("\n" + "=" * 70)
print(f"Total samples created: {len(attack_patterns)}")
print("=" * 70)

# Fill remaining features with reasonable values
for pattern in attack_patterns:
    for feat in feature_names:
        if feat not in pattern:
            if 'IAT' in feat:
                pattern[feat] = np.random.uniform(1000, 12000)
            elif 'Header' in feat:
                pattern[feat] = np.random.uniform(20, 60)
            elif 'Packets/s' in feat:
                pattern[feat] = pattern.get('Flow Packets/s', 150) / 2
            elif 'Packet Length' in feat:
                pattern[feat] = np.random.uniform(50, 600)
            elif 'URG' in feat:
                pattern[feat] = 0
            else:
                pattern[feat] = np.random.uniform(100, 1500)

# Create DataFrame
df = pd.DataFrame(attack_patterns)

# Ensure all 40 features are present
for feat in feature_names:
    if feat not in df.columns:
        df[feat] = np.random.uniform(0, 100, len(df))

# Reorder columns to match feature_names
df = df[feature_names]

# Save to CSV
output_file = 'all_14_attack_types.csv'
df.to_csv(output_file, index=False)

print(f"\n✅ SUCCESS! Created '{output_file}'")
print(f"\n📊 Dataset Summary:")
print(f"   Total samples: {len(df)}")
print(f"   Total features: {len(df.columns)}")
print(f"   Samples per attack type: 3")
print(f"\n🎯 Attack Types Included (14 types):")
print("   1. Benign (3 samples)")
print("   2. Bot (3 samples)")
print("   3. DDoS (3 samples)")
print("   4. DoS GoldenEye (3 samples)")
print("   5. DoS Hulk (3 samples)")
print("   6. DoS Slowhttptest (3 samples)")
print("   7. DoS slowloris (3 samples)")
print("   8. FTP-Patator (3 samples)")
print("   9. Infiltration (3 samples)")
print("   10. PortScan (3 samples)")
print("   11. SSH-Patator (3 samples)")
print("   12. Web Attack - Brute Force (3 samples)")
print("   13. Web Attack - SQL Injection (3 samples)")
print("   14. Web Attack - XSS (3 samples)")
print("\n" + "=" * 70)
print("Ready to use in dashboard! 🚀")
print("=" * 70)
