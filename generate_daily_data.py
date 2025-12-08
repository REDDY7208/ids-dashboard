"""
Generate 7 Days of Network Traffic Data
Each day has different attack patterns and volumes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Attack types and their typical characteristics
ATTACK_TYPES = {
    'BENIGN': {'prob': 0.70, 'features': 'normal'},
    'Bot': {'prob': 0.03, 'features': 'high_packets'},
    'DDoS': {'prob': 0.05, 'features': 'very_high_packets'},
    'DoS GoldenEye': {'prob': 0.02, 'features': 'high_rate'},
    'DoS Hulk': {'prob': 0.03, 'features': 'high_rate'},
    'DoS Slowhttptest': {'prob': 0.02, 'features': 'slow_rate'},
    'DoS slowloris': {'prob': 0.02, 'features': 'slow_rate'},
    'FTP-Patator': {'prob': 0.02, 'features': 'brute_force'},
    'Infiltration': {'prob': 0.01, 'features': 'stealth'},
    'PortScan': {'prob': 0.04, 'features': 'scan'},
    'SSH-Patator': {'prob': 0.02, 'features': 'brute_force'},
    'Web Attack Brute Force': {'prob': 0.01, 'features': 'web'},
    'Web Attack SQL Injection': {'prob': 0.02, 'features': 'web'},
    'Web Attack XSS': {'prob': 0.01, 'features': 'web'}
}

# Feature names (40 features)
FEATURE_NAMES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length'
]

def generate_attack_features(attack_type, attack_info):
    """Generate realistic features for each attack type"""
    features = {}
    
    if attack_info['features'] == 'normal':
        # Benign traffic
        features = {
            'Destination Port': np.random.choice([80, 443, 22, 21, 3306]),
            'Flow Duration': np.random.randint(1000, 50000),
            'Total Fwd Packets': np.random.randint(1, 50),
            'Total Backward Packets': np.random.randint(1, 50),
            'Total Length of Fwd Packets': np.random.randint(100, 5000),
            'Total Length of Bwd Packets': np.random.randint(100, 5000),
            'Flow Bytes/s': np.random.randint(1000, 100000),
            'Flow Packets/s': np.random.randint(1, 100),
        }
    
    elif attack_info['features'] == 'high_packets':
        # Bot, DDoS - High packet volume
        features = {
            'Destination Port': np.random.choice([80, 443]),
            'Flow Duration': np.random.randint(100, 10000),
            'Total Fwd Packets': np.random.randint(100, 1000),
            'Total Backward Packets': np.random.randint(50, 500),
            'Total Length of Fwd Packets': np.random.randint(10000, 100000),
            'Total Length of Bwd Packets': np.random.randint(5000, 50000),
            'Flow Bytes/s': np.random.randint(500000, 5000000),
            'Flow Packets/s': np.random.randint(500, 5000),
        }
    
    elif attack_info['features'] == 'very_high_packets':
        # DDoS - Very high volume
        features = {
            'Destination Port': np.random.choice([80, 443]),
            'Flow Duration': np.random.randint(50, 5000),
            'Total Fwd Packets': np.random.randint(500, 5000),
            'Total Backward Packets': np.random.randint(100, 1000),
            'Total Length of Fwd Packets': np.random.randint(50000, 500000),
            'Total Length of Bwd Packets': np.random.randint(10000, 100000),
            'Flow Bytes/s': np.random.randint(1000000, 10000000),
            'Flow Packets/s': np.random.randint(1000, 10000),
        }
    
    elif attack_info['features'] == 'high_rate':
        # DoS attacks - High rate
        features = {
            'Destination Port': np.random.choice([80, 443]),
            'Flow Duration': np.random.randint(100, 20000),
            'Total Fwd Packets': np.random.randint(200, 2000),
            'Total Backward Packets': np.random.randint(10, 100),
            'Total Length of Fwd Packets': np.random.randint(20000, 200000),
            'Total Length of Bwd Packets': np.random.randint(1000, 10000),
            'Flow Bytes/s': np.random.randint(800000, 8000000),
            'Flow Packets/s': np.random.randint(800, 8000),
        }
    
    elif attack_info['features'] == 'slow_rate':
        # Slow DoS - Low rate, long duration
        features = {
            'Destination Port': np.random.choice([80, 443]),
            'Flow Duration': np.random.randint(50000, 500000),
            'Total Fwd Packets': np.random.randint(5, 50),
            'Total Backward Packets': np.random.randint(1, 20),
            'Total Length of Fwd Packets': np.random.randint(500, 5000),
            'Total Length of Bwd Packets': np.random.randint(100, 1000),
            'Flow Bytes/s': np.random.randint(100, 10000),
            'Flow Packets/s': np.random.randint(1, 10),
        }
    
    elif attack_info['features'] == 'brute_force':
        # Brute force attacks
        features = {
            'Destination Port': np.random.choice([21, 22]),
            'Flow Duration': np.random.randint(1000, 10000),
            'Total Fwd Packets': np.random.randint(50, 200),
            'Total Backward Packets': np.random.randint(50, 200),
            'Total Length of Fwd Packets': np.random.randint(5000, 20000),
            'Total Length of Bwd Packets': np.random.randint(5000, 20000),
            'Flow Bytes/s': np.random.randint(50000, 200000),
            'Flow Packets/s': np.random.randint(50, 200),
        }
    
    elif attack_info['features'] == 'stealth':
        # Infiltration - Stealthy
        features = {
            'Destination Port': np.random.choice([80, 443, 22]),
            'Flow Duration': np.random.randint(10000, 100000),
            'Total Fwd Packets': np.random.randint(10, 100),
            'Total Backward Packets': np.random.randint(10, 100),
            'Total Length of Fwd Packets': np.random.randint(1000, 10000),
            'Total Length of Bwd Packets': np.random.randint(1000, 10000),
            'Flow Bytes/s': np.random.randint(10000, 100000),
            'Flow Packets/s': np.random.randint(10, 100),
        }
    
    elif attack_info['features'] == 'scan':
        # Port scanning
        features = {
            'Destination Port': np.random.randint(1, 65535),
            'Flow Duration': np.random.randint(100, 5000),
            'Total Fwd Packets': np.random.randint(1, 10),
            'Total Backward Packets': np.random.randint(0, 5),
            'Total Length of Fwd Packets': np.random.randint(100, 1000),
            'Total Length of Bwd Packets': np.random.randint(0, 500),
            'Flow Bytes/s': np.random.randint(1000, 50000),
            'Flow Packets/s': np.random.randint(10, 100),
        }
    
    elif attack_info['features'] == 'web':
        # Web attacks
        features = {
            'Destination Port': np.random.choice([80, 443, 8080]),
            'Flow Duration': np.random.randint(1000, 20000),
            'Total Fwd Packets': np.random.randint(20, 200),
            'Total Backward Packets': np.random.randint(20, 200),
            'Total Length of Fwd Packets': np.random.randint(5000, 50000),
            'Total Length of Bwd Packets': np.random.randint(5000, 50000),
            'Flow Bytes/s': np.random.randint(100000, 500000),
            'Flow Packets/s': np.random.randint(100, 500),
        }
    
    # Fill remaining features with random values
    for feature in FEATURE_NAMES:
        if feature not in features:
            features[feature] = np.random.uniform(0, 1000)
    
    return features

def generate_day_data(day_num, records_per_day=500):
    """Generate data for one day with varying attack patterns"""
    
    # Different attack patterns for each day
    day_patterns = {
        1: {'attack_multiplier': 0.8, 'dominant': 'BENIGN'},  # Monday - Normal
        2: {'attack_multiplier': 1.2, 'dominant': 'PortScan'},  # Tuesday - Scanning
        3: {'attack_multiplier': 1.5, 'dominant': 'DDoS'},  # Wednesday - DDoS
        4: {'attack_multiplier': 1.0, 'dominant': 'BENIGN'},  # Thursday - Normal
        5: {'attack_multiplier': 1.8, 'dominant': 'Web Attack SQL Injection'},  # Friday - Web attacks
        6: {'attack_multiplier': 0.6, 'dominant': 'BENIGN'},  # Saturday - Low activity
        7: {'attack_multiplier': 2.0, 'dominant': 'Bot'},  # Sunday - Bot activity
    }
    
    pattern = day_patterns.get(day_num, {'attack_multiplier': 1.0, 'dominant': 'BENIGN'})
    
    data = []
    base_date = datetime.now() - timedelta(days=7-day_num)
    
    for i in range(records_per_day):
        # Adjust probabilities based on day pattern
        adjusted_probs = {}
        for attack, info in ATTACK_TYPES.items():
            if attack == pattern['dominant']:
                adjusted_probs[attack] = info['prob'] * pattern['attack_multiplier'] * 1.5
            else:
                adjusted_probs[attack] = info['prob'] * pattern['attack_multiplier']
        
        # Normalize probabilities
        total_prob = sum(adjusted_probs.values())
        adjusted_probs = {k: v/total_prob for k, v in adjusted_probs.items()}
        
        # Select attack type
        attack_type = np.random.choice(
            list(adjusted_probs.keys()),
            p=list(adjusted_probs.values())
        )
        
        # Generate features
        features = generate_attack_features(attack_type, ATTACK_TYPES[attack_type])
        features['Label'] = attack_type
        
        # Add timestamp
        timestamp = base_date + timedelta(
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        features['Timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        data.append(features)
    
    return pd.DataFrame(data)

def main():
    """Generate 7 days of data"""
    print("🔄 Generating 7 days of network traffic data...")
    print("=" * 60)
    
    # Create daily_data directory
    os.makedirs('daily_data', exist_ok=True)
    
    # Generate data for each day
    for day in range(1, 8):
        print(f"\n📅 Day {day}:")
        
        # Generate data
        df = generate_day_data(day, records_per_day=500)
        
        # Calculate statistics
        attack_counts = df['Label'].value_counts()
        total = len(df)
        benign = attack_counts.get('BENIGN', 0)
        attacks = total - benign
        
        # Save to CSV
        date_str = (datetime.now() - timedelta(days=7-day)).strftime('%Y-%m-%d')
        filename = f'daily_data/day_{day}_{date_str}.csv'
        df.to_csv(filename, index=False)
        
        print(f"   ✅ Generated: {filename}")
        print(f"   📊 Records: {total}")
        print(f"   ✅ Benign: {benign} ({benign/total*100:.1f}%)")
        print(f"   ⚠️  Attacks: {attacks} ({attacks/total*100:.1f}%)")
        print(f"   🎯 Top Attack: {attack_counts.index[1] if len(attack_counts) > 1 else 'None'}")
    
    print("\n" + "=" * 60)
    print("✅ All 7 days generated successfully!")
    print(f"📁 Files saved in: daily_data/")
    print("\n💡 Usage:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("   2. Select '📅 Daily Upload & Tracking' mode")
    print("   3. Upload each day's CSV file")
    print("   4. View daily attack trends and accuracy")

if __name__ == "__main__":
    main()
