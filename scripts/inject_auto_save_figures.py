import nbformat as nbf
import os

notebooks_info = [
    ("notebooks/01_eda_and_feature_selection.ipynb", "nb01", 1),
    ("notebooks/02_model_and_verification.ipynb", "nb02", 1),
    ("notebooks/03_full_sota_pipeline_colab.ipynb", "nb03", 3),
    ("notebooks/04_ablation_with_vs_without_graphs_colab.ipynb", "nb04", 3),
    ("notebooks/05_fraud_typology_cost_function_analysis.ipynb", "nb05", 3),
    ("notebooks/06_walk_forward_temporal_ablation_colab.ipynb", "nb06", 3),
    ("notebooks/07_hyperparameter_tuning_and_significance_colab.ipynb", "nb07", 3),
]

def make_hook_code(prefix):
    return f"""# ==============================================================================
# 📸 AUTO-SAVE HOOK: SALVA AUTOMATICAMENTE TODO GRÁFICO GERADO EM 300 DPI
# ==============================================================================
import os
import re
import matplotlib.pyplot as plt

NOTEBOOK_PREFIX = "{prefix}"
FIG_OUTPUT_DIR = "figures_notebooks"
if os.path.exists("/content"):
    FIG_OUTPUT_DIR = "/content/figures_notebooks"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)

if not hasattr(plt, "_auto_save_installed"):
    _orig_show = plt.show
    _fig_counter = 0

    def auto_save_show(*args, **kwargs):
        global _fig_counter
        fig = plt.gcf()
        if fig and fig.axes:
            _fig_counter += 1
            # Extrai título dos eixos ou da figura
            title = ""
            for ax in fig.axes:
                t = ax.get_title()
                if t:
                    title = t
                    break
            if not title and hasattr(fig, "_suptitle") and fig._suptitle:
                title = fig._suptitle.get_text()
            
            clean_title = re.sub(r"[^\\w\\s-]", "", title).strip().replace(" ", "_")[:35]
            filename = f"{{NOTEBOOK_PREFIX}}_fig_{{_fig_counter:02d}}_{{clean_title}}.png" if clean_title else f"{{NOTEBOOK_PREFIX}}_fig_{{_fig_counter:02d}}.png"
            filepath = os.path.join(FIG_OUTPUT_DIR, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"📸 [Figura Salva]: {{filepath}}")
        _orig_show(*args, **kwargs)

    plt.show = auto_save_show
    plt._auto_save_installed = True
    print(f"✅ Auto-Save de Figuras ativado! Salvando em: [{{FIG_OUTPUT_DIR}}]")
"""

for path, prefix, cell_idx in notebooks_info:
    with open(path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # Check if hook already present
    hook_str = f'NOTEBOOK_PREFIX = "{prefix}"'
    cell_src = nb.cells[cell_idx].source
    if hook_str not in cell_src:
        hook_code = make_hook_code(prefix)
        nb.cells[cell_idx].source = cell_src + "\n\n" + hook_code
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Hook injetado com sucesso em {path} (Cell {cell_idx})")
    else:
        print(f"Hook já presente em {path}")

print("Todos os notebooks foram atualizados com o Auto-Save Hook!")
