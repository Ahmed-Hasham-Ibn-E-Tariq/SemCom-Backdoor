import pandas as pd
import matplotlib.pyplot as plt

# Load your evaluation CSV
csv_path = 'outputs/proposed/ablation/evaluation_results.csv'
df = pd.read_csv(csv_path)

# 1) Boxplot the per-image IQA metrics
iqa_cols = ['PSNR_dB','MS_SSIM','LPIPS_Alex','LPIPS_VGG','PIEAPP','DISTS']
df_iqa = df.dropna(subset=iqa_cols)    # keep only rows with these columns
df_iqa[iqa_cols].boxplot()
plt.title('Per-Image IQA Metrics')
plt.ylabel('Score')
plt.savefig('iqa_metrics_boxplot.png', dpi=150)
print('✅ Saved iqa_metrics_boxplot.png')

# 2) Print FID and KID (assumes these are columns in your CSV)
for metric in ('FID','KID'):
    if metric in df.columns:
        vals = pd.to_numeric(df[metric].dropna(), errors='coerce')
        if len(vals):
            print(f'{metric} = {vals.iloc[0]:.4f}')
