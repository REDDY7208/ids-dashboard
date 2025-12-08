# 📅 Daily Upload & Tracking Guide

## Overview
This feature allows you to upload daily CSV files (one per day for a week) and track:
- **Daily attack patterns** - What types of attacks occurred each day
- **Accuracy metrics** - Model performance for each day
- **Weekly trends** - How attacks evolved over the week
- **Detailed analytics** - Complete breakdown of all metrics

## 🚀 Quick Start

### Step 1: Generate 7 Days of Data
```bash
python generate_daily_data.py
```

This creates 7 CSV files in the `daily_data/` folder:
- `day_1_2024-12-02.csv` - Monday (Normal traffic)
- `day_2_2024-12-03.csv` - Tuesday (Port scanning)
- `day_3_2024-12-04.csv` - Wednesday (DDoS attacks)
- `day_4_2024-12-05.csv` - Thursday (Normal traffic)
- `day_5_2024-12-06.csv` - Friday (Web attacks)
- `day_6_2024-12-07.csv` - Saturday (Low activity)
- `day_7_2024-12-08.csv` - Sunday (Bot activity)

Each file contains **500 records** with realistic attack patterns.

### Step 2: Run the Streamlit App
```bash
streamlit run app.py
```

### Step 3: Upload Daily Files

1. **Select Mode**: Choose "📅 Daily Upload & Tracking" from the sidebar
2. **Select Day**: Choose which day you're uploading (Day 1 - Monday, etc.)
3. **Upload CSV**: Click "Browse files" and select the corresponding CSV file
4. **Analyze**: Click "🔍 Analyze This Day" button
5. **View Results**: See detailed statistics for that day
6. **Repeat**: Upload remaining days to see weekly trends

## 📊 What You'll See

### For Each Day:
- ✅ **Total Records** - Number of network flows analyzed
- ⚠️ **Attacks Detected** - Count and percentage of attacks
- 🎯 **Attack Types** - Pie chart and bar chart of attack distribution
- 📈 **Accuracy** - Model confidence and accuracy for that day
- 📥 **Download** - Export results as CSV

### Weekly Summary (After uploading multiple days):
- 📊 **Weekly Overview** - Total records, attacks, and average accuracy
- 📈 **Daily Trends** - Line chart showing attack vs benign traffic
- 🎯 **Accuracy Trend** - Bar chart of daily accuracy
- ⚠️ **Attack Rate** - Line chart showing attack percentage over time
- 🎯 **Attack Distribution** - Overall attack types across all days
- 📋 **Detailed Table** - Complete statistics for each day

## 📁 File Structure

```
your-project/
├── app.py                      # Main Streamlit app
├── daily_tracking.py           # Daily tracking module
├── generate_daily_data.py      # Data generator
├── daily_tracking.json         # Tracking data (auto-created)
└── daily_data/                 # Generated CSV files
    ├── day_1_2024-12-02.csv
    ├── day_2_2024-12-03.csv
    ├── day_3_2024-12-04.csv
    ├── day_4_2024-12-05.csv
    ├── day_5_2024-12-06.csv
    ├── day_6_2024-12-07.csv
    └── day_7_2024-12-08.csv
```

## 🎯 Example Workflow

### Day 1 (Monday):
```
Upload: day_1_2024-12-02.csv
Results:
  - Total: 500 records
  - Attacks: 150 (30%)
  - Top Attack: PortScan
  - Accuracy: 95.2%
```

### Day 2 (Tuesday):
```
Upload: day_2_2024-12-03.csv
Results:
  - Total: 500 records
  - Attacks: 180 (36%)
  - Top Attack: PortScan
  - Accuracy: 96.1%
```

### Day 3 (Wednesday):
```
Upload: day_3_2024-12-04.csv
Results:
  - Total: 500 records
  - Attacks: 225 (45%)
  - Top Attack: DDoS
  - Accuracy: 97.3%
```

... and so on for all 7 days.

### Weekly Summary:
```
Total Days: 7
Total Records: 3,500
Total Attacks: 1,400
Average Accuracy: 96.5%

Trends:
- Wednesday had highest attack rate (45%)
- Saturday had lowest attack rate (18%)
- DDoS was most common attack type
- Accuracy improved throughout the week
```

## 💡 Features

### 1. Daily Analysis
- Upload one CSV file per day
- Instant analysis with progress bar
- Detailed attack breakdown
- Confidence scores for each prediction

### 2. Weekly Trends
- Attack vs Benign traffic over time
- Daily accuracy comparison
- Attack rate percentage trend
- Overall attack type distribution

### 3. Data Management
- Clear all data and start fresh
- Download individual day results
- Download weekly summary
- Persistent storage (survives app restarts)

### 4. Visual Analytics
- Interactive Plotly charts
- Pie charts for distribution
- Line charts for trends
- Bar charts for comparisons
- Color-coded risk levels

## 🔧 Customization

### Generate Different Data Patterns
Edit `generate_daily_data.py`:

```python
# Change records per day
df = generate_day_data(day, records_per_day=1000)  # Default: 500

# Modify attack patterns
day_patterns = {
    1: {'attack_multiplier': 1.5, 'dominant': 'DDoS'},  # More attacks
    2: {'attack_multiplier': 0.5, 'dominant': 'BENIGN'},  # Fewer attacks
    # ... customize each day
}
```

### Change Tracking Storage
Edit `daily_tracking.py`:

```python
# Use different storage file
tracker = DailyTracker(storage_file='my_tracking.json')
```

## 📈 Metrics Explained

### Accuracy
- Based on model confidence scores
- Higher confidence = Higher accuracy
- Typical range: 90-98%

### Attack Rate
- Percentage of malicious traffic
- Formula: (Attacks / Total Records) × 100
- Normal: 20-40%, High: 40-60%

### Confidence
- Model's certainty in prediction
- Range: 0-100%
- >90% = High confidence
- 70-90% = Medium confidence
- <70% = Low confidence

## 🎓 Use Cases

### 1. Security Training
- Demonstrate attack patterns
- Show how attacks evolve
- Train security analysts

### 2. Model Evaluation
- Test model on different days
- Compare accuracy across time
- Identify weak spots

### 3. Reporting
- Generate weekly security reports
- Track attack trends
- Present to stakeholders

### 4. Research
- Analyze attack patterns
- Study temporal trends
- Compare different scenarios

## 🐛 Troubleshooting

### Issue: "No module named 'daily_tracking'"
**Solution**: Make sure `daily_tracking.py` is in the same folder as `app.py`

### Issue: "File not found"
**Solution**: Run `python generate_daily_data.py` first to create the CSV files

### Issue: "Feature mismatch"
**Solution**: Ensure your CSV has the same 40 features as the model expects

### Issue: "Tracking data not saving"
**Solution**: Check write permissions in the current directory

## 📞 Support

If you encounter issues:
1. Check that all files are in the correct location
2. Verify CSV files have correct format
3. Ensure model files exist in `models/` folder
4. Check console for error messages

## 🎉 Next Steps

After mastering daily tracking:
1. Try the **Real-Time Detection** mode
2. Explore **EDA Analytics** for deeper insights
3. Use **API Documentation** for integration
4. Check **Model Performance** for detailed metrics

---

**Happy Tracking! 🚀**
