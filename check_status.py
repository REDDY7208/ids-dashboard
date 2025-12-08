"""
Quick Status Checker
Check if preprocessing and training are complete
"""

import os

def check_status():
    print("\n" + "="*60)
    print("IDS SYSTEM STATUS CHECK")
    print("="*60)
    
    # Check preprocessing
    print("\n📊 PREPROCESSING:")
    preprocess_files = [
        'data/processed/X_train.npy',
        'data/processed/X_test.npy',
        'data/processed/y_train.npy',
        'data/processed/y_test.npy',
        'models/scaler.pkl',
        'models/label_encoder.pkl',
        'models/feature_names.pkl'
    ]
    
    preprocess_done = all(os.path.exists(f) for f in preprocess_files)
    
    if preprocess_done:
        print("✅ Preprocessing COMPLETE")
        for f in preprocess_files:
            size = os.path.getsize(f) / (1024 * 1024)
            print(f"   ✓ {f} ({size:.2f} MB)")
    else:
        print("❌ Preprocessing NOT complete")
        print("   Run: python src/data_preprocessing.py")
    
    # Check training
    print("\n🤖 MODEL TRAINING:")
    model_file = 'models/cnn_lstm_final.h5'
    
    if os.path.exists(model_file):
        size = os.path.getsize(model_file) / (1024 * 1024)
        print(f"✅ Model training COMPLETE")
        print(f"   ✓ {model_file} ({size:.2f} MB)")
        
        # Check metrics
        if os.path.exists('models/cnn_lstm_metrics.json'):
            import json
            with open('models/cnn_lstm_metrics.json', 'r') as f:
                metrics = json.load(f)
            print(f"\n   📈 Model Performance:")
            print(f"      Accuracy: {metrics.get('accuracy', 0):.2%}")
            print(f"      Precision: {metrics.get('precision', 0):.2%}")
            print(f"      Recall: {metrics.get('recall', 0):.2%}")
            print(f"      F1-Score: {metrics.get('f1_score', 0):.2%}")
    else:
        print("❌ Model training NOT complete")
        print("   Status: Training may be in progress...")
        print("   Run: python src/cnn_lstm_model.py")
    
    # Check dashboard
    print("\n🎨 DASHBOARD:")
    if os.path.exists('app.py'):
        print("✅ Dashboard file exists")
        if preprocess_done and os.path.exists(model_file):
            print("   ✓ Ready to launch!")
            print("   Run: streamlit run app.py")
        else:
            print("   ⚠️  Waiting for preprocessing and training")
    else:
        print("❌ Dashboard file not found")
    
    # Overall status
    print("\n" + "="*60)
    if preprocess_done and os.path.exists(model_file):
        print("🎉 SYSTEM READY!")
        print("="*60)
        print("\nNext step:")
        print("  streamlit run app.py")
    elif preprocess_done:
        print("⏳ TRAINING IN PROGRESS")
        print("="*60)
        print("\nWait for training to complete (30-60 minutes)")
        print("Then run: streamlit run app.py")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("="*60)
        print("\nRun these commands:")
        print("  1. python src/data_preprocessing.py")
        print("  2. python src/cnn_lstm_model.py")
        print("  3. streamlit run app.py")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    check_status()
