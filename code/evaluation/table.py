import pandas as pd
import os

def highlight_best(df):
    """
    df: Pandas DataFrame
    Highlights (with ★) the best result for each metric *per video*.
    """
    df_highlighted = df.copy().astype(str)
    
    # We group by video and apply stars to the best row
    for video in df['Video'].unique():
        mask = df['Video'] == video
        video_data = df[mask]
        
        # FPS (Higher is better)
        best_fps_idx = video_data['FPS'].idxmax()
        df_highlighted.loc[best_fps_idx, 'FPS'] += " ★"
        
        # Avg Confidence (Higher is better)
        best_conf_idx = video_data['Avg Confidence'].idxmax()
        df_highlighted.loc[best_conf_idx, 'Avg Confidence'] += " ★"
        
        # Consistency (Lower is better)
        best_consistency_idx = video_data['Consistency'].idxmin()
        df_highlighted.loc[best_consistency_idx, 'Consistency'] += " ★"
        
        # Params (Lower is better)
        best_params_idx = video_data['Params (M)'].idxmin()
        df_highlighted.loc[best_params_idx, 'Params (M)'] += " ★"
        
    return df_highlighted

def export_results(results_list, output_dir):
    """
    results_list: list of dicts with metrics
    output_dir: base results directory
    """
    df = pd.DataFrame(results_list)
    df = df.round(3)
    
    # Export CSV (original numeric data)
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Highlight best results for Markdown
    df_starred = highlight_best(df)
    
    # Export Markdown
    md_path = os.path.join(output_dir, 'benchmark_results.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Benchmark Comparison Table (★ = Best per category)\n\n")
        f.write(df_starred.to_markdown(index=False))
        f.write("\n\n*Note: High FPS, High Confidence, Low Consistency, and Low Params are preferred.*")
    
    print(f"Results exported to {output_dir}")
    return df
