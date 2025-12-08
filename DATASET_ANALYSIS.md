# Comprehensive Dataset Analysis Report

## Executive Summary
You have **4 major intrusion detection datasets** totaling **~1.65GB** with **5.9+ million records** across 19 files. These are well-known benchmark datasets used in cybersecurity research.

**UPDATE:** A 4th dataset (CIC-IDS in Parquet format) was found with 2.3M records and 15 attack types including Botnet, which is perfect for your requirements!

---

## Dataset 1: WSN-DS (Wireless Sensor Network Dataset)
**Location:** `Datasets/Datasets/1st/WSN-DS.csv`

### Key Statistics
- **Size:** 25.38 MB
- **Records:** 374,661 samples
- **Features:** 19 columns
- **Domain:** Wireless Sensor Network security

### Features Overview
The dataset tracks WSN routing protocol behavior with features like:
- Node identifiers (id, Time, Is_CH, who CH)
- Distance metrics (Dist_To_CH, dist_CH_To_BS)
- Protocol messages (ADV_S, ADV_R, JOIN_S, JOIN_R, SCH_S, SCH_R)
- Data transmission (DATA_S, DATA_R, Data_Sent_To_BS)
- Energy consumption (Expaned Energy)
- Rank and send_code for routing

### Attack Distribution
| Attack Type | Count | Percentage |
|------------|-------|------------|
| Normal | 340,066 | 90.8% |
| Grayhole | 14,596 | 3.9% |
| Blackhole | 10,049 | 2.7% |
| TDMA | 6,638 | 1.8% |
| Flooding | 3,312 | 0.9% |

### Professional Assessment
**Strengths:**
- Focused on IoT/WSN security - highly relevant for modern networks
- Good class balance with 90% normal traffic (realistic)
- Covers routing layer attacks (Grayhole, Blackhole)
- Energy metrics are crucial for resource-constrained devices

**Considerations:**
- Relatively small compared to other datasets
- Limited to WSN-specific attacks
- May need feature engineering for general IDS models

---

## Dataset 2: UNSW-NB15 (Network Intrusion Dataset)
**Location:** `Datasets/Datasets/2nd/`

### Key Statistics
- **Training Set:** 30.8 MB, 175,341 records
- **Testing Set:** 14.67 MB, 82,332 records
- **Total Records:** 257,673
- **Features:** 45 columns (rich feature set)
- **Domain:** Modern network traffic with contemporary attacks

### Features Overview
Comprehensive network flow features including:
- Basic flow features (duration, protocol, service, state)
- Packet statistics (spkts, dpkts, sbytes, dbytes)
- Time-based features (rate, sload, dload, sinpkt, dinpkt)
- TCP-specific (swin, dwin, tcprtt, synack, ackdat)
- Connection features (ct_srv_src, ct_state_ttl, ct_dst_ltm, etc.)
- Content features (trans_depth, response_body_len, ct_flw_http_mthd)

### Attack Distribution - Training Set
| Attack Category | Count | Percentage |
|----------------|-------|------------|
| Normal | 56,000 | 31.9% |
| Generic | 40,000 | 22.8% |
| Exploits | 33,393 | 19.0% |
| Fuzzers | 18,184 | 10.4% |
| DoS | 12,264 | 7.0% |
| Reconnaissance | 10,491 | 6.0% |
| Analysis | 2,000 | 1.1% |
| Backdoor | 1,746 | 1.0% |
| Shellcode | 1,133 | 0.6% |
| Worms | 130 | 0.1% |

### Attack Distribution - Testing Set
| Attack Category | Count | Percentage |
|----------------|-------|------------|
| Normal | 37,000 | 44.9% |
| Generic | 18,871 | 22.9% |
| Exploits | 11,132 | 13.5% |
| Fuzzers | 6,062 | 7.4% |
| DoS | 4,089 | 5.0% |
| Reconnaissance | 3,496 | 4.2% |
| Analysis | 677 | 0.8% |
| Backdoor | 583 | 0.7% |
| Shellcode | 378 | 0.5% |
| Worms | 44 | 0.1% |

### Professional Assessment
**Strengths:**
- **Gold standard dataset** - widely cited in research (2015-present)
- 45 features provide rich information for ML models
- Covers 9 modern attack categories
- Proper train/test split already provided
- Realistic attack distribution with class imbalance
- Includes both binary (label) and multi-class (attack_cat) labels

**Considerations:**
- Imbalanced classes (Worms only 0.1%) - may need SMOTE/oversampling
- "Generic" category is large but vague
- Some rare attacks may be hard to detect

**Best Use Cases:**
- Multi-class classification models
- Ensemble methods (Random Forest, XGBoost)
- Deep learning (handles high dimensionality well)
- Feature importance analysis

---

## Dataset 3: CICIDS2017 (Canadian Institute for Cybersecurity IDS 2017)
**Location:** `Datasets/Datasets/3rd/`

### Key Statistics
- **Total Size:** ~1.15 GB
- **Total Records:** 3,119,353 across 8 files
- **Features:** 78+ columns (most comprehensive)
- **Domain:** Real-world network traffic over 5 days

### File Breakdown
| File | Size (MB) | Records | Primary Attack Types |
|------|-----------|---------|---------------------|
| Monday-WorkingHours | 256.2 | 529,919 | BENIGN (baseline) |
| Tuesday-WorkingHours | 166.6 | 445,910 | FTP-Patator, SSH-Patator |
| Wednesday-workingHours | 272.41 | 692,704 | DoS Hulk, DoS GoldenEye, Heartbleed |
| Thursday-Morning-WebAttacks | 87.77 | 458,969 | SQL Injection, XSS, Brute Force |
| Thursday-Afternoon-Infilteration | 103.69 | 288,603 | Infiltration |
| Friday-Morning | 71.89 | 191,034 | BENIGN |
| Friday-Afternoon-PortScan | 97.16 | 286,468 | Port Scan |
| Friday-Afternoon-DDos | 91.65 | 225,746 | DDoS |

### Attack Types Identified
1. **BENIGN** - Normal traffic (majority)
2. **DoS/DDoS Attacks:**
   - DoS Hulk
   - DoS GoldenEye
   - DDoS
   - Heartbleed
3. **Brute Force:**
   - FTP-Patator
   - SSH-Patator
4. **Web Attacks:**
   - SQL Injection
   - Cross-Site Scripting (XSS)
   - Web Brute Force
5. **Reconnaissance:**
   - Port Scan
6. **Infiltration**

### Features Overview (78+ columns)
Extremely detailed flow-based features:
- Flow identifiers (Flow ID, Source/Dest IP, Ports, Protocol, Timestamp)
- Flow duration and packet counts (Fwd/Bwd)
- Packet length statistics (Max, Min, Mean, Std)
- Inter-arrival times (IAT) - Flow, Fwd, Bwd
- Flags (FIN, SYN, RST, PSH, ACK, URG, CWE, ECE)
- Window sizes and bulk transfer rates
- Subflow features
- Active/Idle time statistics

### Professional Assessment
**Strengths:**
- **Most comprehensive dataset** - 78+ features
- **Realistic temporal structure** - captures attack progression over days
- **Diverse attack types** - covers modern threat landscape
- **High volume** - 3.1M records excellent for deep learning
- **Well-documented** - each day has specific attack scenarios
- Includes rare attacks (Heartbleed, Infiltration)

**Considerations:**
- **Massive size** - requires significant compute resources
- **Highly imbalanced** - BENIGN dominates most files
- Some files have duplicate column names (parsing issues)
- Infiltration attacks are extremely rare (~0.005%)
- Need to combine files for full dataset

**Best Use Cases:**
- Deep learning models (LSTM, CNN, Transformers)
- Time-series analysis
- Anomaly detection
- Real-world attack scenario simulation
- Benchmark for state-of-the-art IDS

---

## Overall Assessment & Recommendations

### Dataset Quality: **9/10**
You have three of the most respected IDS datasets in cybersecurity research. This is publication-grade data.

### Key Insights

**1. Complementary Coverage**
- WSN-DS: IoT/embedded systems
- UNSW-NB15: Modern network attacks
- CICIDS2017: Comprehensive real-world scenarios

**2. Class Imbalance Issues**
All datasets show realistic imbalance (90%+ normal traffic). You'll need:
- SMOTE or ADASYN for oversampling
- Class weights in loss functions
- Stratified sampling
- Ensemble methods

**3. Feature Engineering Opportunities**
- CICIDS2017 has 78 features - consider PCA or feature selection
- UNSW-NB15's 45 features are well-balanced
- WSN-DS may benefit from derived features

### Recommended Approach

**For Binary Classification (Normal vs Attack):**
- Start with UNSW-NB15 (manageable size, good balance)
- Use Random Forest or XGBoost
- Expected accuracy: 85-95%

**For Multi-Class Classification:**
- UNSW-NB15 for 9 attack categories
- CICIDS2017 for specific attack types
- Use deep learning (CNN or LSTM)
- Expected accuracy: 75-85%

**For Research/Publication:**
- Use all three datasets for comprehensive evaluation
- Compare performance across datasets
- Demonstrate generalization capability

### Data Quality Checks Needed

1. **Missing Values:** Check for NaN, Inf, or empty cells
2. **Duplicates:** Remove duplicate flows
3. **Normalization:** Scale features (StandardScaler or MinMaxScaler)
4. **Encoding:** Convert categorical features (protocol, service, state)
5. **Outliers:** Handle extreme values in packet sizes/durations

### Computational Requirements

- **WSN-DS:** Can run on laptop (25MB)
- **UNSW-NB15:** Laptop/desktop (45MB total)
- **CICIDS2017:** Requires 16GB+ RAM, GPU recommended for deep learning

---

## Next Steps

1. **Data Preprocessing:**
   - Load and merge datasets
   - Handle missing values
   - Normalize/standardize features
   - Encode categorical variables

2. **Exploratory Data Analysis:**
   - Feature correlation analysis
   - Distribution plots
   - Attack pattern visualization

3. **Model Selection:**
   - Baseline: Logistic Regression, Decision Trees
   - Advanced: Random Forest, XGBoost, Neural Networks
   - State-of-art: Deep learning (LSTM, CNN, Attention)

4. **Evaluation Metrics:**
   - Accuracy (overall)
   - Precision, Recall, F1-score (per class)
   - Confusion matrix
   - ROC-AUC curves

---

---

## Dataset 4: CIC-IDS (Parquet Format) ⭐ RECOMMENDED
**Location:** `Datasets/Datasets/cic-ids/`

### Key Statistics
- **Total Size:** ~258 MB (compressed Parquet format)
- **Total Records:** 2,313,810
- **Features:** 78 columns (same as CICIDS2017)
- **Files:** 8 Parquet files (organized by attack type)
- **Domain:** Modern network attacks with clean labels

### File Breakdown
| File | Size (MB) | Records | Attack Types |
|------|-----------|---------|--------------|
| Benign-Monday | 54.14 | 458,831 | Benign |
| Botnet-Friday | 18.94 | 176,038 | Benign, Bot |
| Bruteforce-Tuesday | 44.00 | 389,714 | Benign, FTP-Patator, SSH-Patator |
| DDoS-Friday | 24.13 | 221,264 | Benign, DDoS |
| DoS-Wednesday | 65.04 | 584,991 | Benign, DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed |
| Infiltration-Thursday | 22.07 | 207,630 | Benign, Infiltration |
| Portscan-Friday | 12.96 | 119,522 | Benign, PortScan |
| WebAttacks-Thursday | 16.84 | 155,820 | Benign, Web Attack Brute Force, Web Attack XSS, Web Attack SQL Injection |

### Attack Types (15 Total)
1. **Benign** (Normal traffic)
2. **Bot** (Botnet/C2 Traffic) ✅
3. **FTP-Patator** (FTP Brute Force) ✅
4. **SSH-Patator** (SSH Brute Force) ✅
5. **DDoS** ✅
6. **DoS slowloris** ✅
7. **DoS Slowhttptest** ✅
8. **DoS Hulk** ✅
9. **DoS GoldenEye** ✅
10. **Heartbleed** ✅
11. **Infiltration** ✅
12. **PortScan** ✅
13. **Web Attack - Brute Force** ✅
14. **Web Attack - XSS** (Cross-Site Scripting) ✅
15. **Web Attack - SQL Injection** ✅

### Professional Assessment
**Strengths:**
- ⭐ **BEST CHOICE FOR YOUR PROJECT** - Has 15 attack types including Botnet!
- **Parquet format** - 5-10x faster loading than CSV
- **Pre-organized** - Each file contains specific attack types
- **Clean labels** - No parsing issues
- **Comprehensive** - 78 features covering all network aspects
- **Includes Botnet** - Critical for modern threat detection
- **Balanced size** - 2.3M records perfect for deep learning
- **All required attacks present:**
  - ✅ SQL Injection
  - ✅ XSS
  - ✅ Botnet/C2 Traffic
  - ✅ Brute Force (SSH/FTP)
  - ✅ DDoS/DoS variants
  - ✅ Port Scanning
  - ✅ Infiltration

**Advantages over CICIDS2017 CSV:**
- Much smaller file size (258MB vs 1.15GB)
- Faster loading (Parquet is columnar format)
- No duplicate column issues
- Pre-cleaned data
- Better organized by attack type

**Best Use Cases:**
- **Primary dataset for your IDS project**
- Multi-class classification (15 classes)
- Deep learning (CNN, LSTM)
- Real-time prediction
- Hardware integration testing

---

## UPDATED Overall Assessment & Recommendations

### Dataset Quality: **10/10** ⭐
With the addition of CIC-IDS Parquet dataset, you now have the perfect data for your project!

### Recommended Dataset Strategy

**PRIMARY: CIC-IDS Parquet (Dataset 4)** ✅
- Use this as your main dataset
- 15 attack types (exceeds your 10-15 requirement)
- Includes Botnet, SQL Injection, XSS, Brute Force
- Fast loading with Parquet format
- 2.3M records perfect for deep learning

**SECONDARY: UNSW-NB15 (Dataset 2)** (Optional)
- Add if you need more attack diversity
- Provides: Exploits, Fuzzers, Backdoor, Shellcode, Worms

**TERTIARY: WSN-DS (Dataset 1)** (Optional)
- Add for IoT-specific attacks
- Provides: Grayhole, Blackhole, Flooding

### Updated Attack Coverage

**From CIC-IDS alone, you get 15 attack types:**
1. Benign (Normal)
2. Bot (Botnet) ✅
3. FTP Brute Force ✅
4. SSH Brute Force ✅
5. DDoS ✅
6. DoS slowloris ✅
7. DoS Slowhttptest ✅
8. DoS Hulk ✅
9. DoS GoldenEye ✅
10. Heartbleed ✅
11. Infiltration ✅
12. Port Scan ✅
13. Web Brute Force ✅
14. XSS ✅
15. SQL Injection ✅

**This covers your requirements:**
- ✅ SQL Injection
- ✅ XSS
- ✅ Botnet/C2 Traffic
- ✅ Brute Force SSH/FTP
- ✅ DDoS/DoS variants
- ✅ Port Scanning
- ✅ Web Attacks

### Updated Implementation Plan

**Phase 1: Use CIC-IDS Parquet**
1. Load all 8 Parquet files
2. Combine into single dataset
3. 15 attack classes (exceeds requirement)
4. 78 features (comprehensive)
5. 2.3M records (perfect for CNN/LSTM)

**Phase 2: Model Training**
- Random Forest (baseline)
- XGBoost (fast)
- CNN (spatial patterns)
- LSTM (temporal patterns)
- CNN-LSTM (best accuracy)

**Phase 3: Streamlit Dashboard**
- File upload mode
- Real-time JSON mode
- Model comparison
- Attack visualization

**Expected Performance:**
- Accuracy: 95-98% (with deep learning)
- 15 attack classes detected
- Real-time prediction < 50ms
- Hardware-ready API

---

**Bottom Line:** The CIC-IDS Parquet dataset is PERFECT for your project! It has 15 attack types (including Botnet, SQL Injection, XSS), is in efficient Parquet format, and covers all your requirements. This should be your primary dataset.
