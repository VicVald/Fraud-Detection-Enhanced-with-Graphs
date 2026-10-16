import nbformat

with open('notebooks/01_eda_and_feature_selection.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_ch9 = """---
## 9. Síntese Teórica & Blueprint de Engenharia de Atributos para o Modelo Final

> 📊 **Resultados Empíricos Consolidados na Partição de 50% (~295k Transações):**

| Técnica de Fraude Identificada | Evidência Estatística no IEEE-CIS (Amostra 50%) | Feature / Grafo Criado no Projeto | Proteção Anti-Leakage / Decisão de Modelagem |
| :--- | :--- | :--- | :--- |
| **Fraud Rings & Botnets** | **628 dispositivos com 3+ UIDs** (concentrando taxa média de 7,0% e anéis com até 100% de fraude) | Grafo Bipartido UID ↔ Dispositivo, `device_degree`, PageRank e Embeddings GNN GraphSAGE | Projeção indutiva sem rótulos supervisionados; poda de dispositivos nulos |
| **Device & OS Spoofing** | **Android 5.1.1 (29,89%)** e **iOS 11 (18,03%)** têm risco até 8,5x superior à taxa basal (3,5%) | Metadados de Identidade categorizados (`DeviceInfo`, `id_30`, `id_31`) | Fit dos encoders exclusivo na partição de treino histórico |
| **Card Velocity (Burstiness)** | **13,43% de taxa de fraude em rajadas ≤ 60s** (vs. 3,71% em ritmo normal — 3,6x mais arriscado) | `uid_tx_count_24h`, contadores deslizantes de janela rápida | Algoritmo causal $\\mathcal{O}(N \\log K)$ via busca binária (`np.searchsorted`) |
| **Cross-Border CNP Abuse** | **`ProductCD = C` concentra 11,50% de fraude** (28.128 transações analisadas) | Roteamento Dual de Identidade (Model A / Model B) e codificação de produtos | Separação estrita de modelos com e sem dados de identidade |
| **Anomalia de Ticket Financeiro** | Divergência relativa do valor transacionado vs. histórico ($Z$-Score) | `TransactionAmt_to_mean_uid`, `TransactionAmt_to_std_uid` | Agrupamento estatístico cumulativo ancorado no passado |
| **Concept Drift & Nós Inéditos** | **35,1% dos UIDs e 20,3% dos dispositivos na validação são inéditos** | Arquitetura Híbrida Indutiva (XGBoost + Topologia) | Proíbe memorização transducional de IDs; aprende propriedades estruturais de rede |

---

### 💡 Conclusão do Diagnóstico Experimental:
O pipeline de engenharia de atributos está **100% ancorado nas assinaturas empíricas de fraudes reais** comprovadas nesta EDA. A presença maciça de entidades inéditas no futuro (35,1%) descarta categoricamente modelos baseados em memorização de chaves primárias, fundamentando a necessidade da nossa abordagem híbrida baseada em **topologia de grafos bipartidos**, **velocidade causal** e **compressão espectral PCA**."""

for cell in nb.cells:
    if '## 9. Síntese Teórica' in cell.source:
        cell.source = new_ch9
        print("Capítulo 9 atualizado perfeitamente com todas as tags de código!")
        break

with open('notebooks/01_eda_and_feature_selection.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
