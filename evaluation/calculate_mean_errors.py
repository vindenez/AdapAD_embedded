import os
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_mean_error(results_folder="../results"):
    
    if not os.path.exists(results_folder):
        print(f"Error: Folder '{results_folder}' does not exist.")
        return None
    
    csv_files = list(Path(results_folder).glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{results_folder}' folder.")
        return None
    
    print(f"Found {len(csv_files)} CSV files in '{results_folder}' folder.")
    
    all_errors = []
    file_summaries = []
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            if 'err' not in df.columns:
                print(f"Warning: 'err' column not found in {file_path.name}")
                continue
            
            errors = df['err'].dropna()
            
            errors = pd.to_numeric(errors, errors='coerce').dropna()
            
            if len(errors) > 0:
                file_mean = errors.mean()
                all_errors.extend(errors.tolist())
                file_summaries.append({
                    'file': file_path.name,
                    'count': len(errors),
                    'mean_error': file_mean,
                    'min_error': errors.min(),
                    'max_error': errors.max()
                })
                print(f"{file_path.name}: {len(errors)} error values, mean = {file_mean:.6f}")
            else:
                print(f"Warning: No valid error values found in {file_path.name}")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            continue
    
    if all_errors:
        overall_mean = np.mean(all_errors)
        total_values = len(all_errors)
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        file_summaries.sort(key=lambda x: x['mean_error'], reverse=True)
        
        def format_error(value):
            """Format error values for better readability"""
            if value == 0:
                return "0.000000"
            elif value < 0.000001:
                return f"{value:.2e}"
            elif value < 0.001:
                return f"{value:.6f}"
            else:
                return f"{value:.6f}"
        
        def shorten_filename(filename):
            """Shorten long filenames for better table display"""
            if len(filename) <= 30:
                return filename
            name = filename.replace('_log.csv', '').replace('.csv', '')
            if len(name) > 30:
                return name[:27] + "..."
            return name
        
        print("\n ERROR ANALYSIS BY FILE (sorted by mean error):")
        print("─" * 95)
        print(f"{'File':<32} {'Count':<8} {'Mean Error':<14} {'Min Error':<14} {'Max Error':<14}")
        print("─" * 95)
        
        for i, summary in enumerate(file_summaries, 1):
            short_name = shorten_filename(summary['file'])
            mean_str = format_error(summary['mean_error'])
            min_str = format_error(summary['min_error'])
            max_str = format_error(summary['max_error'])
            
            rank = f"{i:2d}."
            print(f"{rank} {short_name:<29} {summary['count']:<8} {mean_str:<14} {min_str:<14} {max_str:<14}")
        
        print("─" * 95)
        
        print(f"\n OVERALL STATISTICS:")
        print(f"   Total files processed: {len(file_summaries):,}")
        print(f"   Total error values:    {total_values:,}")
        print(f"   Overall mean error:    {format_error(overall_mean)}")
        print(f"   Standard deviation:    {format_error(np.std(all_errors))}")
        print(f"   Minimum error:         {format_error(np.min(all_errors))}")
        print(f"   Maximum error:         {format_error(np.max(all_errors))}")
        
        return overall_mean
    else:
        print("No valid error values found in any files.")
        return None

if __name__ == "__main__":
    mean_error = calculate_mean_error()
    
    if mean_error is not None:
        print(f"\nFinal Result: Mean Error = {mean_error:.6f}")
    else:
        print("\nNo results to display.")