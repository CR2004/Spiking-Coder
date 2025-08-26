import os
import json
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# save plot
def save_plot(fig, filename):
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filepath}")

# Multi-model comparisons
def plot_multi_model_comparison(reports, key, label):
    models = [r["model"] for r in reports]
    values = [r["global_scores"][key] for r in reports]

    fig, ax = plt.subplots()
    ax.bar(models, values, color="purple")
    ax.set_ylim(0, 1)
    ax.set_title(f"Global {label} Comparison")
    ax.set_ylabel(label)
    plt.xticks(rotation=30, ha="right")
    save_plot(fig, f"multi_global_{key}.png")


def plot_multi_category_comparison(reports, metric):
    categories = sorted({cat for r in reports for cat in r["category_scores"].keys()})
    models = [r["model"] for r in reports]

    width = 0.2
    group_spacing = 0.4  
    group_width = width * len(models) + group_spacing
    x = np.arange(len(categories)) * group_width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(reports):
        values = [r["category_scores"].get(c, {}).get(metric, 0) for c in categories]
        ax.bar(x + i * width, values, width, label=r["model"])

    ax.set_xticks(x + (len(models)-1) * width / 2)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Category Comparison - {metric}")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_plot(fig, f"multi_category_{metric}.png")


def plot_multi_framework_comparison(reports, metric):
    frameworks = sorted({fw for r in reports for fw in r["framework_scores"].keys()})
    models = [r["model"] for r in reports]

    width = 0.2
    group_spacing = 0.4  # extra space between framework groups
    group_width = width * len(models) + group_spacing
    x = np.arange(len(frameworks)) * group_width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(reports):
        values = [r["framework_scores"].get(f, {}).get(metric, 0) for f in frameworks]
        ax.bar(x + i * width, values, width, label=r["model"])

    ax.set_xticks(x + (len(models)-1) * width / 2)
    ax.set_xticklabels(frameworks, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"Framework Comparison - {metric}")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_plot(fig, f"multi_framework_{metric}.png")



# Main script
if __name__ == "__main__":
    reports = []
    for file in os.listdir("."):
        if file.endswith("_eval_report.json"):
            with open(file, "r", encoding="utf-8") as f:
                reports.append(json.load(f))

    if not reports:
        print("No *_eval_report.json files found in current directory.")
        exit()

    # Multi-model comparisons
    metrics = {
        "syntactic_pass_rate": "Syntactic Pass Rate",
        "semantic_score": "Semantic Score",
        "import_pass_rate": "Import Pass Rate",
        "functional_pass_rate": "Functional Pass Rate",
        "composite_snnbench_score": "Composite Score"
    }

    for key, label in metrics.items():
        plot_multi_model_comparison(reports, key, label)
        plot_multi_category_comparison(reports, key)
        plot_multi_framework_comparison(reports, key)
