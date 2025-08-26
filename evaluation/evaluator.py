#This script evaluates generated SNN code snippets based on syntax, semantics, imports, and functionality.
# It categorizes errors and computes scores across different categories and frameworks.
# The results are saved in a structured JSON format for further analysis.

import os
import ast
import json
import subprocess
from collections import defaultdict, Counter

# Syntax Check
def check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

# Semantic Check 
def check_semantics(code: str, rules: list[str]) -> tuple[int, int]:
    matched = sum(1 for rule in rules if rule in code)
    return matched, len(rules)


# Import Check 
def check_imports(code: str, framework: str) -> bool:
    if framework.lower() == "brian2":
        return "import brian2" in code or "from brian2" in code
    elif framework.lower() == "snntorch":
        return "import snntorch" in code or "from snntorch" in code
    elif framework.lower() == "norse":
        return "import norse" in code or "from norse" in code
    return False

# Functional Check 
def run_with_timeout(code: str, timeout: int = 50):
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return True, None
        return False, result.stderr.strip() if result.stderr else "Unknown runtime error"
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    except Exception as e:
        return False, str(e)
    
# Error Categorization
def categorize_error(error_msg: str) -> str:
    if not error_msg:
        return "no_error"

    if "SyntaxError" in error_msg:
        return "syntax_error"
    if "IndentationError" in error_msg:
        return "indentation_error"
    if "ModuleNotFoundError" in error_msg or "No module named" in error_msg:
        return "module_not_found"
    if "ImportError" in error_msg:
        return "import_error"
    if "NameError" in error_msg:
        return "name_error"
    if "AttributeError" in error_msg:
        return "attribute_error"
    if "TypeError" in error_msg:
        return "type_error"
    if "ValueError" in error_msg:
        return "value_error"
    if "DimensionMismatchError" in error_msg:
        return "brian2_dimension_mismatch"
    if "EquationError" in error_msg:
        return "brian2_equation_error"
    if "Brian equations/expressions do not support" in error_msg:
        return "brian2_expression_error"
    if "KeyError" in error_msg:
        return "key_error"

    return "runtime_error"

# Main Evaluator
def evaluate_model_results(file_path: str, output_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_scores = defaultdict(lambda: {
        "syntax_pass": 0,
        "semantic_matched": 0,
        "semantic_total": 0,
        "import_pass": 0,
        "functional_pass": 0,
        "count": 0
    })
    framework_scores = defaultdict(lambda: {
        "syntax_pass": 0,
        "semantic_matched": 0,
        "semantic_total": 0,
        "import_pass": 0,
        "functional_pass": 0,
        "count": 0
    })

    error_counter = Counter()

    total_syntax_pass = 0
    total_semantic_matched = 0
    total_semantic_total = 0
    total_import_pass = 0
    total_functional_pass = 0
    total_prompts = 0

    detailed_results = []

    for entry in data:
        code = entry.get("response", "")
        category = entry.get("category", "Unknown")
        framework = entry.get("framework", "Unknown")
        rules = entry.get("semantic_rules", [])

        total_prompts += 1
        category_scores[category]["count"] += 1

        # Syntax 
        syntax_ok = check_syntax(code)
        if syntax_ok:
            total_syntax_pass += 1
            category_scores[category]["syntax_pass"] += 1

        #  Semantics 
        matched, total = check_semantics(code, rules)
        total_semantic_matched += matched
        total_semantic_total += total
        category_scores[category]["semantic_matched"] += matched
        category_scores[category]["semantic_total"] += total

        # Imports 
        import_ok = check_imports(code, framework)
        if import_ok:
            total_import_pass += 1
            category_scores[category]["import_pass"] += 1

        # Functional 
        functional_ok, error = run_with_timeout(code)
        if functional_ok:
            total_functional_pass += 1
            category_scores[category]["functional_pass"] += 1
        else:
            if error:
                error_category = categorize_error(error)
                error_counter[error_category] += 1

        framework_scores[framework]["count"] += 1

        if syntax_ok:
            framework_scores[framework]["syntax_pass"] += 1

        framework_scores[framework]["semantic_matched"] += matched
        framework_scores[framework]["semantic_total"] += total

        if import_ok:
            framework_scores[framework]["import_pass"] += 1

        if functional_ok:
            framework_scores[framework]["functional_pass"] += 1



        detailed_results.append({
            "id": entry.get("id"),
            "category": category,
            "framework": framework,
            "syntax_ok": syntax_ok,
            "semantic_matched": matched,
            "semantic_total": total,
            "import_ok": import_ok,
            "functional_ok": functional_ok,
            "error": error if not functional_ok else None
        })

    # Global Scores 
    global_scores = {
        "syntactic_pass_rate": total_syntax_pass / total_prompts if total_prompts else 0,
        "semantic_score": (total_semantic_matched / total_semantic_total) if total_semantic_total else 0,
        "import_pass_rate": total_import_pass / total_prompts if total_prompts else 0,
        "functional_pass_rate": total_functional_pass / total_prompts if total_prompts else 0
    }

    # Composite Score 
    composite = (
        0.3 * global_scores["syntactic_pass_rate"] +
        0.3 * global_scores["semantic_score"] +
        0.1 * global_scores["import_pass_rate"] +
        0.3 * global_scores["functional_pass_rate"]
    )
    global_scores["composite_snnbench_score"] = composite

    # Category Scores
    category_final = {}
    for cat, stats in category_scores.items():
        count = stats["count"]
        category_final[cat] = {
            "syntactic_pass_rate": stats["syntax_pass"] / count if count else 0,
            "semantic_score": (stats["semantic_matched"] / stats["semantic_total"]) if stats["semantic_total"] else 0,
            "import_pass_rate": stats["import_pass"] / count if count else 0,
            "functional_pass_rate": stats["functional_pass"] / count if count else 0
        }
        cat_composite = (
            0.3 * category_final[cat]["syntactic_pass_rate"] +
            0.3 * category_final[cat]["semantic_score"] +
            0.1 * category_final[cat]["import_pass_rate"] +
            0.3 * category_final[cat]["functional_pass_rate"]
        )
        category_final[cat]["composite_snnbench_score"] = cat_composite
    # Framework Scores
    framework_final = {}
    for fw, stats in framework_scores.items():
        count = stats["count"]
        framework_final[fw] = {
            "syntactic_pass_rate": stats["syntax_pass"] / count if count else 0,
            "semantic_score": (stats["semantic_matched"] / stats["semantic_total"]) if stats["semantic_total"] else 0,
            "import_pass_rate": stats["import_pass"] / count if count else 0,
            "functional_pass_rate": stats["functional_pass"] / count if count else 0
        }
        fw_composite = (
            0.3 * framework_final[fw]["syntactic_pass_rate"] +
            0.3 * framework_final[fw]["semantic_score"] +
            0.1 * framework_final[fw]["import_pass_rate"] +
            0.3 * framework_final[fw]["functional_pass_rate"]
        )
        framework_final[fw]["composite_snnbench_score"] = fw_composite

    report = {
        "model": os.path.basename(file_path).replace("_results.json", ""),
        "global_scores": global_scores,
        "category_scores": category_final,
        "framework_scores": framework_final,
        "error_summary": dict(error_counter),
        "detailed_results": detailed_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation complete. Results saved to {output_path}")


# Run Example
if __name__ == "__main__":
    results_files = [
        "snn-finetuned-model-v2_results.json",
        "deepseek-coder-6.7b-Instruct.json",
        "Llama-3.1-8B-Instruct.json",
        "Llama-3.2-3B-Instruct.json",
        "snn-finetuned-model-v1.json",
    ]
    for file in results_files:
        output_file = file.replace(".json", "_eval_report.json")
        evaluate_model_results(file, output_file)
