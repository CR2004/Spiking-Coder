# 📊 Evaluation

This directory contains the **evaluation results** for Spiking-CODER models, Deepseek-Coder-6.7b-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct. 
We evaluate models **globally**, as well as **framework-wise** and **category-wise**.

---

## 📂 Contents
- **`leaderboards/`**  
  - **`global/`** → Overall model rankings across all prompts  
  - **`frameworks/`** → Rankings per SNN framework (e.g., Brian2, Norse, snnTorch)  
  - **`categories/`** → Rankings per prompt category (e.g., Neuron and Synapse Models, Learning Rules, Network Creation, Code Optimization)  
- **`plots/`** → Visualization of evaluation metrics across models
- **`code responses/`** → extracted  code blocks of every snnbench prompt from all the models.
- **`raw responses/`** → Raw responses including explanation text of every snnbench prompt from all the models.
- **`model reports/`** → Raw evaluation reports for each model
- **`snnbench_prompts.json`** → Contains **79 evaluation prompts** along with their corresponding **semantic rules**, **frameworks**, and **categories**. These are the benchmark tasks used to evaluate the models.

---

## 🧮 SNNBench Composite Score  

The composite score is computed as a weighted sum of four evaluation metrics:

**Composite Score = (0.3 × Syntactic Pass Rate) +  
(0.3 × Semantic Score) +  
(0.1 × Import Pass Rate) +  
(0.3 × Functional Pass Rate)**



