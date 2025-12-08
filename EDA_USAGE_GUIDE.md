# 📊 EDA Usage Guide - Step by Step

## 🚀 Quick Start

### Step 1: Launch the Dashboard
```bash
streamlit run app.py
```

### Step 2: Navigate to EDA
1. Look at the **sidebar** on the left
2. Find the **"Select Mode"** dropdown
3. Choose **"📊 Exploratory Data Analytics (EDA)"**

### Step 3: Select Your Data Source
You'll see three options:

#### Option A: 💾 Sample Network Data (Recommended for First Time)
- **Best for:** Quick demo and testing
- **Data:** Pre-loaded sample network traffic
- **Action:** Just select and start exploring!

#### Option B: 📁 Upload Custom CSV
- **Best for:** Analyzing your own network data
- **Data:** Your custom CSV file
- **Action:** Click "Browse files" and upload

#### Option C: 🗄️ Database History
- **Best for:** Analyzing past detections
- **Data:** Historical detection records
- **Action:** Select and view analytics

---

## 📋 Section-by-Section Guide

### 📊 Section 1: Dataset Overview
**What you'll see:**
- 📊 Total Records - How many data points
- 📈 Total Features - Number of columns
- 💾 Memory Usage - Dataset size
- ❓ Missing Data - Data quality metric

**Expandable sections:**
- 📊 Dataset Information - Data types breakdown
- 👀 Data Preview - First 10 rows

**Use this to:** Get a quick understanding of your data size and quality

---

### 📈 Section 2: Statistical Summary
**What you'll see:**
- Complete descriptive statistics table
- Top 15 features by mean values (blue bar chart)
- Top 15 features by standard deviation (red bar chart)

**Key metrics:**
- Mean, Median, Std Dev
- Min, Max, Quartiles
- Count, Unique values

**Use this to:** Understand the central tendency and spread of your features

---

### 📊 Section 3: Distribution Analysis
**Interactive controls:**
- **Multi-select:** Choose features to analyze
- **Plot Type:** Switch between Histogram, Box Plot, Violin Plot

**Visualization types:**

1. **Histogram**
   - Shows frequency distribution
   - Includes marginal box plot
   - Best for: Understanding data shape

2. **Box Plot**
   - Shows quartiles and outliers
   - Compare multiple features
   - Best for: Outlier detection

3. **Violin Plot**
   - Shows distribution density
   - Combines box plot + density
   - Best for: Detailed distribution shape

**Use this to:** Understand how your data is distributed

---

### 🔗 Section 4: Correlation Analysis
**What you'll see:**
- 🔥 Correlation Heatmap (top 20 features)
- 🔍 Highly Correlated Feature Pairs table

**Interactive controls:**
- **Correlation Threshold:** Adjust from 0.0 to 1.0
- **Top N Correlations:** Show 5 to 30 pairs

**Color coding:**
- 🔴 Red = Negative correlation
- ⚪ White = No correlation
- 🔵 Blue = Positive correlation

**Use this to:** Find relationships between features

---

### 🎯 Section 5: Outlier Detection
**What you'll see:**
- 📊 Outlier Summary table
  - Feature name
  - Outlier count
  - Outlier percentage
  - Lower/Upper bounds
- 📈 Top 10 Features by Outlier Count (bar chart)

**Method:** IQR (Interquartile Range)
- Lower Bound = Q1 - 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

**Use this to:** Identify anomalous data points

---

### 🎯 Section 6: Feature Importance & Variance
**What you'll see:**
- 📊 Top 15 Features by Variance (purple bar chart)
- 📈 Top 15 Features by Range (pink bar chart)

**Metrics shown:**
- Variance - How spread out the data is
- Std Dev - Standard deviation
- Range - Max - Min value

**Use this to:** Identify which features vary the most (potentially most informative)

---

### ❓ Section 7: Missing Data Analysis
**What you'll see:**
- 📊 Missing Data Summary table
- 📈 Missing Data Visualization (red bar chart)

**Metrics:**
- Missing Count - Number of missing values
- Missing % - Percentage of missing data

**Result:**
- ✅ "No missing data found!" (if clean)
- Or detailed breakdown of missing values

**Use this to:** Assess data quality and completeness

---

### 🔍 Section 8: Prediction & Attack Pattern Analysis
**The most powerful section!**

**Step 1:** Click **"🚀 Run Predictions on Dataset"**
- Progress bar shows analysis status
- Model predicts attack type for each record

**Step 2:** View Results

**Charts you'll get:**
1. **Attack Type Distribution** (pie chart)
   - Shows % of each attack type
   - Donut chart with legend

2. **Risk Level Distribution** (pie chart)
   - High (red), Medium (orange), Low (green)
   - Color-coded for clarity

3. **Confidence Score Distribution** (histogram)
   - Shows prediction confidence levels
   - Most predictions should be high confidence

**Detailed Statistics Table:**
- Attack Type
- Average Confidence
- Min/Max Confidence
- Standard Deviation
- Count

**Step 3:** Download Results
- Click **"📥 Download Complete Analysis"**
- Gets CSV with all predictions
- Includes: Original data + Predictions + Confidence + Risk Level

**Use this to:** Understand attack patterns in your network traffic

---

## 🎨 Tips for Best Results

### 1. Start with Sample Data
- Get familiar with the interface
- Understand what each section shows
- Then move to your own data

### 2. Use Multiple Plot Types
- Histogram for overall shape
- Box plot for outliers
- Violin plot for detailed distribution

### 3. Adjust Thresholds
- Correlation threshold: Start at 0.7, adjust as needed
- Top N: Start at 15, increase for more details

### 4. Focus on High Variance Features
- These are usually most informative
- Check their distributions
- Look for patterns

### 5. Check Correlations
- High correlations (>0.9) might indicate redundancy
- Negative correlations can be interesting
- Use for feature selection

### 6. Run Predictions Last
- Understand your data first
- Then see how the model performs
- Compare predictions with distributions

---

## 📊 Example Workflow

### For Network Security Analysis:

1. **Load Data** → Sample Network Data
2. **Check Overview** → Verify data quality
3. **View Statistics** → Understand feature ranges
4. **Check Distributions** → Look for anomalies
5. **Find Correlations** → Identify relationships
6. **Detect Outliers** → Find suspicious traffic
7. **Run Predictions** → Classify attacks
8. **Download Results** → Save for reporting

### For Research/Publication:

1. **Load Data** → Your research dataset
2. **Statistical Summary** → For methodology section
3. **Distribution Analysis** → For data description
4. **Correlation Heatmap** → For feature analysis
5. **Outlier Detection** → For data cleaning
6. **Feature Importance** → For feature selection
7. **Predictions** → For results section
8. **Export Charts** → Right-click → Save image

---

## 🎯 What Each Metric Means

### Statistical Metrics:
- **Mean:** Average value
- **Std:** How spread out the data is
- **Min/Max:** Range of values
- **25%/50%/75%:** Quartiles (percentiles)

### Correlation Values:
- **1.0:** Perfect positive correlation
- **0.0:** No correlation
- **-1.0:** Perfect negative correlation
- **>0.7:** Strong correlation
- **0.3-0.7:** Moderate correlation
- **<0.3:** Weak correlation

### Risk Levels:
- **High:** Confidence > 90%
- **Medium:** Confidence 70-90%
- **Low:** Confidence < 70%

---

## 💡 Common Questions

### Q: Which data source should I use?
**A:** Start with "Sample Network Data" to learn the interface, then use "Upload Custom CSV" for your own analysis.

### Q: How many features should I select for distribution analysis?
**A:** Start with 2-3 features. Too many makes charts hard to read.

### Q: What's a good correlation threshold?
**A:** 0.7 is standard. Lower it to 0.5 to see more relationships, raise to 0.9 for only very strong correlations.

### Q: Should I worry about outliers?
**A:** Depends on context. In network security, outliers might be attacks! Don't remove them automatically.

### Q: How long does prediction take?
**A:** ~1-2 seconds per 100 records. Sample data (4 records) is instant.

### Q: Can I export the charts?
**A:** Yes! Right-click any chart → "Save image as" or use the download button for CSV data.

---

## 🚀 Advanced Usage

### Comparing Multiple Datasets:
1. Run EDA on Dataset A
2. Download results
3. Upload Dataset B
4. Run EDA again
5. Compare statistics manually

### Feature Engineering:
1. Check correlations
2. Identify redundant features
3. Look at variance
4. Select most informative features
5. Use for model improvement

### Anomaly Detection:
1. Check outliers section
2. Look at distribution plots
3. Run predictions
4. Focus on high-confidence attacks
5. Investigate outlier records

---

## ✅ Success Checklist

After using EDA, you should know:
- ✅ How many records and features you have
- ✅ Data quality (missing values, outliers)
- ✅ Feature distributions (normal, skewed, etc.)
- ✅ Which features are correlated
- ✅ Which features have high variance
- ✅ Attack type distribution in your data
- ✅ Model confidence levels
- ✅ Risk level breakdown

---

## 🎓 Learning Path

### Beginner:
1. Use sample data
2. Explore each section
3. Read the metrics
4. Run predictions

### Intermediate:
1. Upload your own data
2. Adjust thresholds
3. Compare plot types
4. Analyze correlations

### Advanced:
1. Use database history
2. Temporal analysis
3. Feature engineering
4. Model optimization

---

**You now have a complete EDA toolkit for network security analysis!** 🎉

For questions or issues, check the main dashboard or API documentation sections.
