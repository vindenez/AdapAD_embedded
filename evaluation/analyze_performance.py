import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('performance_log.csv')

fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'black'
ax1.set_xlabel('Timestep', fontsize=14)
ax1.set_ylabel('Decision Time (seconds)', color=color1, fontsize=14)
line1 = ax1.plot(df['timestep'], df['total_decision_time'], color=color1, alpha=0.8, linewidth=1.5, label='Decision Time')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Context Switches', color=color2, fontsize=14)
line2 = ax2.plot(df['timestep'], df['total_switches'], color=color2, alpha=0.8, linewidth=1.5, label='Context Switches')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

plt.title('Decision Time vs Context Switches', fontsize=16, pad=20)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

plt.tight_layout()

plt.show()

correlation = df['total_switches'].corr(df['total_decision_time'])
print(f"Correlation between context switches and decision time: {correlation:.3f}")

print(f"\nDecision Time Statistics:")
print(f"- Mean: {df['total_decision_time'].mean():.3f}s")
print(f"- Std: {df['total_decision_time'].std():.3f}s")
print(f"- Min: {df['total_decision_time'].min():.3f}s")
print(f"- Max: {df['total_decision_time'].max():.3f}s")

print(f"\nContext Switches Statistics:")
print(f"- Mean: {df['total_switches'].mean():.1f}")
print(f"- Std: {df['total_switches'].std():.1f}")
print(f"- Min: {df['total_switches'].min()}")
print(f"- Max: {df['total_switches'].max()}")