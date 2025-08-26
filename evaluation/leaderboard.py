import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import re

def safe_name(name: str) -> str:
    """Make category/framework names safe for filesystem paths."""
    return re.sub(r'[^A-Za-z0-9_-]', '_', name)

def plot_framework_leaderboard(reports, framework, metric, save_dir="leaderboards"):
    fw_dir = os.path.join(save_dir, framework)
    os.makedirs(fw_dir, exist_ok=True)

    # Collect model + framework + metric score
    rows = []
    for r in reports:
        if framework in r.get("framework_scores", {}):
            rows.append({
                "model": r["model"],
                metric: r["framework_scores"][framework].get(metric, 0)
            })

    if not rows:
        return  

    df = pd.DataFrame(rows).sort_values(metric, ascending=False)

    # ---- Plot leaderboard ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["model"], df[metric], color="lightgreen")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"{framework} Leaderboard - {metric.replace('_', ' ').title()}")
    ax.invert_yaxis()  # best model at top

    for i, v in enumerate(df[metric]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")

    # expand x-axis to avoid clipping numbers
    ax.set_xlim(0, df[metric].max() + 0.1)

    fig.tight_layout()
    fig.savefig(os.path.join(fw_dir, f"{framework}_leaderboard_{metric}.png"))
    plt.close(fig)


def generate_framework_leaderboards(reports, save_dir="leaderboards"):
    metrics = ["syntactic_pass_rate", "semantic_score", "import_pass_rate", "functional_pass_rate", "composite_snnbench_score"]
    frameworks = sorted({fw for r in reports for fw in r["framework_scores"].keys()})

    for fw in frameworks:
        for metric in metrics:
            plot_framework_leaderboard(reports, fw, metric, save_dir)

    print(f"Framework leaderboards generated under {save_dir}/<framework>/")


# ---------- Category Leaderboards ----------
def plot_category_leaderboard(reports, category, metric, save_dir="leaderboards"):
    safe_cat = safe_name(category)
    cat_dir = os.path.join(save_dir, "categories", safe_cat)
    os.makedirs(cat_dir, exist_ok=True)

    rows = []
    for r in reports:
        if category in r.get("category_scores", {}):
            rows.append({
                "model": r["model"],
                metric: r["category_scores"][category].get(metric, 0)
            })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["model"], df[metric], color="orange")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"{category} Leaderboard - {metric.replace('_', ' ').title()}")
    ax.invert_yaxis()

    for i, v in enumerate(df[metric]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")

    ax.set_xlim(0, df[metric].max() + 0.1)

    fig.tight_layout()
    fig.savefig(os.path.join(cat_dir, f"{safe_cat}_leaderboard_{metric}.png"))
    plt.close(fig)


def generate_category_leaderboards(reports, save_dir="leaderboards"):
    metrics = ["syntactic_pass_rate", "semantic_score", "import_pass_rate",
               "functional_pass_rate", "composite_snnbench_score"]
    categories = sorted({cat for r in reports for cat in r["category_scores"].keys()})

    for cat in categories:
        for metric in metrics:
            plot_category_leaderboard(reports, cat, metric, save_dir)

    print(f"Category leaderboards generated under {save_dir}/categories/")


def plot_leaderboard(reports, metric, save_dir="leaderboards/global"):
    os.makedirs(save_dir, exist_ok=True)

    # Collect model + metric score
    rows = []
    for r in reports:
        rows.append({
            "model": r["model"],
            metric: r["global_scores"].get(metric, 0)
        })

    df = pd.DataFrame(rows).sort_values(metric, ascending=False)

    # ---- Plot leaderboard ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["model"], df[metric], color="skyblue")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Global Leaderboard - {metric.replace('_', ' ').title()}")
    ax.invert_yaxis()  # best model at top

    for i, v in enumerate(df[metric]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")

    # expand x-axis to avoid clipping numbers
    ax.set_xlim(0, df[metric].max() + 0.1)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"leaderboard_{metric}.png"))
    plt.close(fig)

def generate_all_leaderboards(reports, save_dir="leaderboards"):
    metrics = ["syntactic_pass_rate", "semantic_score", "import_pass_rate", "functional_pass_rate", "composite_snnbench_score"]

    # Global
    global_dir = os.path.join(save_dir, "global")
    for metric in metrics:
        plot_leaderboard(reports, metric, global_dir)

    # Framework-specific
    generate_framework_leaderboards(reports, save_dir)

    # Category-specific
    generate_category_leaderboards(reports, save_dir)

    print(f"Leaderboards generated under {save_dir}/")


if __name__ == "__main__":
    results_files = [
        "model reports/deepseek-coder-6.7b-Instruct_eval_report.json",
        "model reports/Llama-3.1-8B-Instruct_eval_report.json",
        "model reports/Llama-3.2-3B-Instruct_eval_report.json",
        "model reports/snn-finetuned-model-v1_eval_report.json",
        "model reports/snn-finetuned-model-v2_eval_report.json"
    ]
    reports = []
    for file in results_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                reports.append(json.load(f))

    if reports:
        generate_all_leaderboards(reports)
    else:
        print("No reports loaded. Check file paths.")
