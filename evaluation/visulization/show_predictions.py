import pandas as pd
import matplotlib.pyplot as plt

def plot_observed_vs_predicted(filepath):
    df = pd.read_csv(filepath)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['observed'], color='black', linewidth=1.5, label='Observed')
    plt.plot(df['predicted'], color='red', linewidth=1.2, label='Predicted')
    
    plt.ylim(0, 20)  # Change these values as needed
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    plot_observed_vs_predicted('results/pressure_temperature_log.csv')