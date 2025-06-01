import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries_comparison():
    progress_df = pd.read_csv('results/Original_Python_AdapAD_Tide_pressure_results.csv')
    value_log_df = pd.read_csv('results/Tide_pressure/value_log.csv')
    
    print(f"Progress dataset shape: {progress_df.shape}")
    print(f"Value log dataset shape: {value_log_df.shape}")
    
    obs_match = np.array_equal(progress_df['observed'].values, value_log_df['observed'].values)
    print(f"Observed values are aligned: {obs_match}")
    
    plt.figure(figsize=(15, 8))
    
    x_axis = range(len(progress_df))
    
    plt.plot(x_axis, progress_df['observed'], 
             color='black', 
             linewidth=1.5, 
             label='Observed Values', 
             alpha=0.8)
    
    plt.plot(x_axis, progress_df['predicted'], 
             color='blue', 
             linewidth=1.2, 
             label='Predicted - AdapAD', 
             alpha=0.7)
    
    plt.plot(x_axis, value_log_df['predicted'], 
             color='red', 
             linewidth=1.2, 
             label='Predicted - Embedded AdapAD', 
             alpha=0.7)
    
    plt.title('Time Series Comparison: Observed vs Predicted Values', fontsize=16, fontweight='bold')
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.ylim(713, 763)  
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    obs_mean = progress_df['observed'].mean()
    prog_pred_mean = progress_df['predicted'].mean()
    val_log_pred_mean = value_log_df['predicted'].mean()
    
    plt.tight_layout()
    
    plt.show()
    
    print(f"Observed values - Min: {progress_df['observed'].min():.3f}, Max: {progress_df['observed'].max():.3f}, Mean: {obs_mean:.3f}")
    print(f"Progress predicted - Min: {progress_df['predicted'].min():.3f}, Max: {progress_df['predicted'].max():.3f}, Mean: {prog_pred_mean:.3f}")
    print(f"Value log predicted - Min: {value_log_df['predicted'].min():.3f}, Max: {value_log_df['predicted'].max():.3f}, Mean: {val_log_pred_mean:.3f}")
    
    progress_mae = np.mean(np.abs(progress_df['observed'] - progress_df['predicted']))
    value_log_mae = np.mean(np.abs(value_log_df['observed'] - value_log_df['predicted']))
    
    print(f"\n=== PREDICTION ACCURACY ===")
    print(f"Progress model MAE: {progress_mae:.6f}")
    print(f"Value log model MAE: {value_log_mae:.6f}")
    print(f"Better model: {'Value Log' if value_log_mae < progress_mae else 'Progress'}")

def plot_with_confidence_intervals():
    progress_df = pd.read_csv('progress_0.0038.csv')
    value_log_df = pd.read_csv('value_log_0.0045.csv')
    
    plt.figure(figsize=(15, 10))
    
    x_axis = range(len(progress_df))
    
    plt.plot(x_axis, progress_df['observed'], 
             color='black', 
             linewidth=1.5, 
             label='Observed Values', 
             alpha=0.9,
             zorder=3)
    
    plt.plot(x_axis, progress_df['predicted'], 
             color='blue', 
             linewidth=1.2, 
             label='Predicted (Progress)', 
             alpha=0.8,
             zorder=2)
    
    plt.fill_between(x_axis, progress_df['low'], progress_df['high'],
                     color='blue', alpha=0.2, label='Progress CI')
    
    plt.plot(x_axis, value_log_df['predicted'], 
             color='red', 
             linewidth=1.2, 
             label='Predicted (Value Log)', 
             alpha=0.8,
             zorder=2)
    
    plt.fill_between(x_axis, value_log_df['low'], value_log_df['high'],
                     color='red', alpha=0.2, label='Value Log CI')
    
    plt.title('Time Series with Confidence Intervals: Observed vs Predicted Values', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time Index', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.ylim(713, 763)  
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Creating basic comparison plot...")
    plot_timeseries_comparison()
    
    print("\n" + "="*50)
    print("Creating enhanced plot with confidence intervals...")
    plot_with_confidence_intervals()