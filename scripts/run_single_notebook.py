import sys
import time
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell

def run_notebook(nb_path):
    print(f"==================================================")
    print(f"🚀 Iniciando execução: {nb_path}")
    print(f"==================================================")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    shell = InteractiveShell.instance()
    
    # Ensure current working directory is the repository root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        
    total_cells = len(nb.cells)
    start_time = time.time()
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    
    print(f"Total de células: {total_cells} (Células de código: {len(code_cells)})")
    
    for count, idx in enumerate(code_cells, 1):
        cell = nb.cells[idx]
        source = cell.source
        
        # Skip pure pip install commands to save time
        lines = []
        for l in source.split("\n"):
            stripped = l.strip()
            if stripped.startswith("!pip install") or stripped.startswith("%pip install"):
                indent = len(l) - len(l.lstrip())
                lines.append(" " * indent + "pass")
            else:
                lines.append(l)
        clean_source = "\n".join(lines)
        if not clean_source.strip():
            continue
            
        print(f"\n[Célula {count}/{len(code_cells)} - Linha {idx}] Executando...")
        cell_start = time.time()
        
        # Execute cell in IPython shell
        res = shell.run_cell(clean_source)
        cell_dur = time.time() - cell_start
        
        if res.error_in_exec:
            print(f"❌ ERRO na célula {idx}: {res.error_in_exec}")
            raise res.error_in_exec
            
        print(f"✓ Concluída em {cell_dur:.1f}s")
        
    total_dur = time.time() - start_time
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"\n==================================================")
    print(f"✅ {nb_path} finalizado e salvo com sucesso em {total_dur/60:.2f} minutos!")
    print(f"==================================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/run_single_notebook.py <caminho_do_notebook>")
        sys.exit(1)
    run_notebook(sys.argv[1])
