import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_tide_pressure_with_anomalies():
    tide_data = pd.read_csv('data/Tide_pressure.validation_stage copy.csv')
    
    progress_data = pd.read_csv('results/cut/progress_0.0038.csv')
    value_log_data = pd.read_csv('results/cut/value_log_0.0045.csv')
    
    timesteps = range(1, len(tide_data) + 1)
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(timesteps, tide_data['value'], 
             color='black', linewidth=1, alpha=0.7, label='Tide pressure')
    
    tide_anomalies = tide_data[tide_data['is_anomaly'] == 1]
    if not tide_anomalies.empty:
        anomaly_timesteps = [i+1 for i in tide_anomalies.index]
        plt.scatter(anomaly_timesteps, tide_anomalies['value'], 
                   color='red', s=50, alpha=0.8, marker='o', 
                   label='Labeled Anomalies', zorder=5)
    
    progress_anomalies = progress_data[progress_data['anomalous'] == True]
    if not progress_anomalies.empty:
        progress_anomaly_indices = progress_anomalies.index
        if len(progress_anomaly_indices) > 0 and max(progress_anomaly_indices) < len(tide_data):
            progress_timesteps = [i+1 for i in progress_anomaly_indices]
            progress_values = tide_data.iloc[progress_anomaly_indices]['value']
            plt.scatter(progress_timesteps, progress_values, 
                       color='green', s=100, alpha=0.8, marker='x', 
                       label='AdapAD (Python) Anomalies', zorder=5)
    
    value_log_anomalies = value_log_data[value_log_data['anomalous'] == True]
    if not value_log_anomalies.empty:
        value_log_anomaly_indices = value_log_anomalies.index
        if len(value_log_anomaly_indices) > 0 and max(value_log_anomaly_indices) < len(tide_data):
            value_log_timesteps = [i+1 for i in value_log_anomaly_indices]
            value_log_values = tide_data.iloc[value_log_anomaly_indices]['value']
            plt.scatter(value_log_timesteps, value_log_values, 
                       color='blue', s=150, alpha=0.8, marker='+', 
                       label='Embedded AdapAD (C++) Anomalies', zorder=5)
    
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Tide Pressure Value', fontsize=12)
    plt.title('Tide Pressure Validation Stage', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.ylim(713, 763)
    
    plt.tight_layout()
    
    total_points = len(tide_data)
    tide_anomaly_count = len(tide_anomalies) if not tide_anomalies.empty else 0
    progress_anomaly_count = len(progress_anomalies)
    value_log_anomaly_count = len(value_log_anomalies)
    
    print(f"Dataset Statistics:")
    print(f"Total data points: {total_points}")
    print(f"Tide data anomalies: {tide_anomaly_count}")
    print(f"Progress anomalies: {progress_anomaly_count}")
    print(f"Value log anomalies: {value_log_anomaly_count}")
    
    plt.show()

def plot_detailed_comparison():
    """
    Create a more detailed comparison plot with subplots
    """
    tide_data = pd.read_csv('data/Tide_pressure.validation_stage copy.csv')
    progress_data = pd.read_csv('results/cut/progress_0.0038.csv')
    value_log_data = pd.read_csv('results/cut/value_log_0.0045.csv')
    
    timesteps = range(1, len(tide_data) + 1)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    axes[0].plot(timesteps, tide_data['value'], 
                color='blue', linewidth=1, alpha=0.7)
    tide_anomalies = tide_data[tide_data['is_anomaly'] == 1]
    if not tide_anomalies.empty:
        anomaly_timesteps = [i+1 for i in tide_anomalies.index]
        axes[0].scatter(anomaly_timesteps, tide_anomalies['value'], 
                       color='red', s=30, alpha=0.8)
    axes[0].set_ylabel('Tide Pressure')
    axes[0].set_title('Tide Pressure with Original Anomalies')
    axes[0].set_ylim(713, 763)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(timesteps, progress_data['predicted'], 
                color='green', linewidth=1, label='Predicted')
    axes[1].fill_between(timesteps, progress_data['low'], progress_data['high'], 
                        alpha=0.2, color='green', label='Confidence Interval')
    progress_anomalies = progress_data[progress_data['anomalous'] == True]
    if not progress_anomalies.empty:
        progress_indices = progress_anomalies.index
        progress_timesteps = [i+1 for i in progress_indices]
        axes[1].scatter(progress_timesteps, 
                       progress_data.iloc[progress_indices]['predicted'], 
                       color='green', s=30, alpha=0.8, marker='x')
    axes[1].set_ylabel('Progress Model')
    axes[1].set_title('Progress Model Predictions and Anomalies')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(timesteps, value_log_data['predicted'], 
                color='purple', linewidth=1, label='Predicted')
    axes[2].fill_between(timesteps, value_log_data['low'], value_log_data['high'], 
                        alpha=0.2, color='purple', label='Confidence Interval')
    value_log_anomalies = value_log_data[value_log_data['anomalous'] == True]
    if not value_log_anomalies.empty:
        value_log_indices = value_log_anomalies.index
        value_log_timesteps = [i+1 for i in value_log_indices]
        axes[2].scatter(value_log_timesteps, 
                       value_log_data.iloc[value_log_indices]['predicted'], 
                       color='blue', s=30, alpha=0.8, marker='+')
    axes[2].set_ylabel('Value Log Model')
    axes[2].set_title('Value Log Model Predictions and Anomalies')
    axes[2].set_xlabel('Timestep')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Creating tide pressure plot with anomalies...")
    plot_tide_pressure_with_anomalies()
    