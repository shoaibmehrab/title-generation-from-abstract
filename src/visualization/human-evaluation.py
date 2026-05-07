import matplotlib.pyplot as plt
import numpy as np

# 1. The Aggregated Data
metrics = ['Informativeness', 'Faithfulness', 'Fluency']
settings = ['Zero-Shot', '4-Shot']

# Format: [Zero-Shot Score, 4-Shot Score]
data = {
    'Gemma 2 (9B)': {
        'Informativeness': [4.80, 4.85],
        'Faithfulness': [4.70, 4.75],
        'Fluency': [4.50, 4.60],
        'color': '#2ca02c', # Green (Stable)
        'marker': 'o'
    },
    'Llama 3.1 (8B)': {
        'Informativeness': [4.70, 4.65],
        'Faithfulness': [4.60, 4.55],
        'Fluency': [4.55, 4.50],
        'color': '#1f77b4', # Blue (Stable)
        'marker': 's'
    },
    'Llama 3 (8B)': {
        'Informativeness': [4.60, 3.20],
        'Faithfulness': [4.55, 3.19],
        'Fluency': [4.50, 3.50],
        'color': '#d62728', # Red (Highlights the Mode Collapse)
        'marker': '^'
    }
}

# 2. Configure Presentation Styling
plt.rcParams.update({
    'font.size': 14, 
    'axes.titlesize': 16, 
    'axes.labelsize': 14,
    'font.family': 'sans-serif'
})

# Create a figure with 3 subplots side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

# 3. Plot the lines for each metric
for i, metric in enumerate(metrics):
    ax = axes[i]
    for model, props in data.items():
        ax.plot(settings, props[metric], marker=props['marker'], color=props['color'],
                linewidth=4, markersize=12, label=model)
    
    # Format each subplot
    ax.set_title(metric, fontweight='bold', pad=15)
    ax.set_ylim(2.5, 5.2) # Tight y-axis to emphasize the drop
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Clean up the borders (remove top and right lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Only add the Y-axis label to the very first chart
    if i == 0:
        ax.set_ylabel('Average Human Score (1-5)', fontweight='bold', labelpad=15)

# 4. Add a clean, centered legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), 
           frameon=False, prop={'size': 16, 'weight': 'bold'})

# 5. Final Layout Adjustments & Save
plt.tight_layout()
plt.subplots_adjust(bottom=0.2) # Make room for the bottom legend

# Save as a high-resolution transparent PNG for PowerPoint
plt.savefig('human_evaluation_slopegraph.png', dpi=600, transparent=True, bbox_inches='tight')
print("Chart successfully saved as 'human_evaluation_slopegraph.png'")
plt.show()