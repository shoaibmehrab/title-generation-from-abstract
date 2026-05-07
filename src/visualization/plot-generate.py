import matplotlib.pyplot as plt
import seaborn as sns
import math

# Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif'})

# Data
seq2seq_data = [
    ("BART Large", 49.77, 90.48),
    ("FLAN-T5", 39.32, 88.57),
    ("PEGASUS PubMed", 25.05, 86.83),
    ("PEGASUS XSum", 48.47, 90.44)
]

zeroshot_data = [
    ("Gemma 2 (0-shot)", 39.10, 89.30),
    ("Llama 3 (0-shot)", 39.80, 89.18),
    ("Llama 3.1 (0-shot)", 39.65, 89.21)
]

fewshot_data = [
    ("Gemma 2 (4-shot)", 38.87, 89.32),
    ("Llama 3 (4-shot)", 38.12, 88.20),
    ("Llama 3.1 (4-shot)", 36.69, 89.13)
]

s2s_names, s2s_x, s2s_y = zip(*seq2seq_data)
zs_names, zs_x, zs_y = zip(*zeroshot_data)
fs_names, fs_x, fs_y = zip(*fewshot_data)

plt.figure(figsize=(12, 8))

# Plot markers
plt.scatter(s2s_x, s2s_y, color='#1f77b4', marker='^', s=110, label='Fine-Tuned Seq2Seq', edgecolors='black', alpha=0.85)
plt.scatter(zs_x, zs_y, color='#d62728', marker='o', s=110, label='Zero-Shot LLMs', edgecolors='black', alpha=0.85)
plt.scatter(fs_x, fs_y, color='#2ca02c', marker='s', s=110, label='4-Shot LLMs', edgecolors='black', alpha=0.85)

all_x = list(s2s_x) + list(zs_x) + list(fs_x)
all_y = list(s2s_y) + list(zs_y) + list(fs_y)
all_labels = list(s2s_names) + list(zs_names) + list(fs_names)

# Per-label offsets (data units). Adjust numbers if you want small changes.
offsets = {
    "FLAN-T5":        (0.8,  -0.3),   # a bit to the right
    "PEGASUS XSum":   (0.0,  0.55),  # show on top
    "BART Large":     (0.0, -0.6),   # show below
    "Llama 3.1 (4-shot)": (0.0, -0.5),# show below
    "Llama 3.1 (0-shot)": (0.0, 0.5),# show below
    "Llama 3 (0-shot)": (0.3, 0.2),# show below
    "Gemma 2 (4-shot)":   (0.0,  1.0) # higher above following Y axis
}

# Default offset used for other labels
default_offset = (0.35, -0.25)

arrowprops = dict(arrowstyle='-', color='gray', lw=0.7, shrinkA=0, shrinkB=0, alpha=0.7)

def hv_align(dx, dy):
    # horizontal alignment
    if dx > 0.0:
        ha = 'left'
    elif dx < 0.0:
        ha = 'right'
    else:
        ha = 'center'
    # vertical alignment
    if dy > 0.0:
        va = 'bottom'
    elif dy < 0.0:
        va = 'top'
    else:
        va = 'center'
    return ha, va

for x, y, label in zip(all_x, all_y, all_labels):
    dx, dy = offsets.get(label, default_offset)
    ann_x = x + dx
    ann_y = y + dy

    ha, va = hv_align(dx, dy)

    plt.annotate(
        label,
        xy=(x, y),
        xytext=(ann_x, ann_y),
        fontsize=10,
        fontweight='bold',
        color='black',
        ha=ha,
        va=va,
        bbox=dict(facecolor='white', edgecolor='black', linewidth=0.6, pad=0.6, alpha=1.0),
        arrowprops=arrowprops,
        zorder=4
    )

plt.xlim(20, 55)
plt.ylim(85, 92)
plt.xlabel("ROUGE-1 Score (%)", fontweight='bold')
plt.ylabel("BERTScore F1 (%)", fontweight='bold')
plt.title("Semantic vs. Lexical Trade-off in Title Generation", fontweight='bold', pad=15)
plt.legend(loc='lower right', frameon=True, shadow=True, title="Model Paradigm")

plt.tight_layout()
plt.savefig("semantic_vs_lexical_tradeoff.png", dpi=300)
plt.savefig("semantic_vs_lexical_tradeoff.pdf", dpi=700)
plt.show()