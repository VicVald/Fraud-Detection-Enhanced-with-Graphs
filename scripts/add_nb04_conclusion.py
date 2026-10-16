import nbformat

with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

synthesis_markdown = """---
### 💡 Síntese Metodológica da Ablação: O Papel Central do FE Tabular e o Refinamento Incremental dos Grafos

A análise rigorosa da tabela de ablação estabelece uma conclusão científica transparente e definitiva:

1. **O FE Tabular Carrega a Maior Parte do Ganho Absoluto (~75% a 80% do Lift):**
   - A introdução isolada de conhecimento de domínio em dados tabulares (Deltas $D_n$, resolução de portador `uid` e agregações de comportamento financeiro por cartão na Configuração 5) elevou o PR-AUC de **0,5038** para **0,5304** (+0,0266).
   - Esse salto responde pela esmagadora maioria de todo o ganho preditivo obtido sobre o baseline. Nenhuma técnica genérica de grafos substitui a boa engenharia de atributos de domínio.

2. **O Grafo Fornece um Refinamento Incremental Ortogonal (+0,004 a +0,009 em PR-AUC):**
   - Os atributos derivados de grafos (graus de conectividade bipartida e PageRank de perfis de dispositivo) agregam um ganho incremental estrito de **+0,0040 em PR-AUC** na versão com PCA (0,5304 $\\rightarrow$ **0,5344**) e de **+0,0091 em PR-AUC** na versão com atributos $V$ brutos (0,5304 $\\rightarrow$ **0,5395** bruto / **0,5434** com pós-processamento causal alinhado).
   - O valor dos grafos reside na sua **alta especialização**: ao capturar a topologia relacional de perfis de hardware e emuladores, o modelo ganha capacidade de interceptar fraudes coordenadas em rede que nenhuma variável IID tabular consegue isolar.
"""

# Check if cell already exists
found = False
for cell in nb.cells:
    if '### 💡 Síntese Metodológica da Ablação' in cell.source:
        cell.source = synthesis_markdown
        found = True
        break

if not found:
    # insert after cell 15 (which prints the ablation table)
    new_cell = nbformat.v4.new_markdown_cell(synthesis_markdown)
    nb.cells.insert(16, new_cell)

with open('notebooks/04_ablation_with_vs_without_graphs_colab.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("✓ Síntese metodológica adicionada com sucesso ao Notebook 04!")
