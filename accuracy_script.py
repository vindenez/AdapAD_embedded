import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_tide_pressure(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    return df

def load_adapad_log(file_path):
    df = pd.read_csv(file_path)
    # Remove rows where anomalous is empty
    df = df.dropna(subset=['anomalous'])
    # Convert boolean strings to actual booleans
    df['anomalous'] = df['anomalous'].astype(bool)
    return df

def find_matching_start(tide_df, adapad_df, window_size=10):
    """Find where the ADAPAD log starts in the original dataset by matching sequences."""
    adapad_start_seq = adapad_df['observed'].iloc[:window_size].values
    
    for i in range(len(tide_df) - window_size):
        tide_seq = tide_df['value'].iloc[i:i+window_size].values
        # Check if sequences match (with small tolerance for floating point differences)
        if np.allclose(tide_seq, adapad_start_seq, rtol=1e-5):
            return i
    
    raise ValueError("Could not find matching start sequence in datasets")

def evaluate_predictions(tide_df, adapad_df):
    # Convert Tide_pressure anomalies to boolean for comparison
    tide_anomalies = tide_df['is_anomaly'].astype(bool)
    adapad_anomalies = adapad_df['anomalous']
    
    metrics = {
        'accuracy': accuracy_score(tide_anomalies, adapad_anomalies),
        'precision': precision_score(tide_anomalies, adapad_anomalies, zero_division=0),
        'recall': recall_score(tide_anomalies, adapad_anomalies, zero_division=0),
        'f1_score': f1_score(tide_anomalies, adapad_anomalies, zero_division=0),
        'confusion_matrix': confusion_matrix(tide_anomalies, adapad_anomalies)
    }
    
    return metrics

def analyze_predictions(tide_path, adapad_path):
    print(f"\nAnalyzing {adapad_path}")
    
    # Load datasets
    tide_df = load_tide_pressure(tide_path)
    adapad_df = load_adapad_log(adapad_path)
    
    try:
        # Find matching start position
        start_idx = find_matching_start(tide_df, adapad_df)
        print(f"Found matching start at index {start_idx} (timestamp: {tide_df['timestamp'].iloc[start_idx]})")
        
        # Align datasets
        aligned_tide_df = tide_df.iloc[start_idx:start_idx + len(adapad_df)].reset_index(drop=True)
        
        # Verify alignment
        if not np.allclose(aligned_tide_df['value'].values, adapad_df['observed'].values, rtol=1e-5):
            raise ValueError("Dataset alignment failed - values don't match")
        
        # Get metrics
        metrics = evaluate_predictions(aligned_tide_df, adapad_df)
        
        # Print results
        print("\nMetrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Find and display mismatches
        mismatches = pd.DataFrame({
            'timestamp': aligned_tide_df['timestamp'],
            'value': aligned_tide_df['value'],
            'true_anomaly': aligned_tide_df['is_anomaly'],
            'predicted_anomaly': adapad_df['anomalous']
        })
        mismatches = mismatches[mismatches['true_anomaly'] != mismatches['predicted_anomaly']]
        
        if len(mismatches) > 0:
            print(f"\nFound {len(mismatches)} mismatched predictions")
            print("Sample of mismatches:")
            print(mismatches.head())
        else:
            print("\nNo mismatched predictions found")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

def main():
    # Update these paths to match your file locations
    tide_path = "data/Tide_pressure.csv"
    adapad_files = [
        "adapad_log_40_7_2_40_5_0.000200lrG_0.000200lr_.csv"
    ]
    
    for adapad_path in adapad_files:
        analyze_predictions(tide_path, adapad_path)

if __name__ == "__main__":
    main()