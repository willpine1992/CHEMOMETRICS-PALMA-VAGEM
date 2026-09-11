
```markdown
# Especificações do Dashboard: Estética & Layout para Quimiometria (Adsorção)

---

## 🎨 Guia de Estilo Visual

### 1. Paleta de Cores
* **Fundo da Página:** `#F4F6F9` ou `#EFEFEF` (Cinza claro/off-white suave)
* **Cartões e Seções:** `#FFFFFF` (Branco puro para destacar os blocos)
* **Texto Principal:** `#2C3E50` ou `#333333` (Cinza escuro para alto contraste)
* **Texto Secundário/Métricas:** `#7F8C8D` (Cinza médio)
* **Cores Principais de Dados (Gradiente Teal/Verde-Água):**
  * **Verde-Água Escuro:** `#008080` / `#00A896`
  * **Teal Intermediário:** `#02C39A`
  * **Verde-Água Claro:** `#80ED99`
  * **Sombra/Gradiente (Área):** Preenchimento translúcido da cor principal (opacidade entre 20% e 40%)
* **Cores de Destaque Secundário:**
  * **Azul Escuro (Substituindo o Rosa/Vermelho original):** `#2A6F97` (iOS / Métricas)
  * **Alerta/Declínio:** `#E74C3C` (Vermelho suave para tendências negativas)

---

### 2. Tipografia
* **Fonte do Sistema:** `Inter`, `Roboto`, `Helvetica Neue` ou `Segoe UI` (Sans-Serif limpa e moderna).
* **Pesos e Tamanhos:**
  * **KPI - Valores Principais:** `Bold`, ~24px - 28px
  * **KPI - Título Superior:** `Regular` ou `Medium`, ~11px - 12px (Caixa alta, cinza)
  * **KPI - Variação/Porcentagem:** `Medium`, ~11px (Verde para crescimento, vermelho para queda)
  * *Títulos das Seções:** `Bold` ou `Semi-Bold`, ~14px - 16px (Cinza escuro)
  * **Subtítulos de Gráficos:** `Regular`, ~12px (Cinza claro)
  * **Rótulos dos Eixos / Legendas:** `Regular`, ~10px - 11px

---

### 3. Componentes de UI e Estilo Geral
* **Bordas dos Cartões:** Sem bordas visíveis ou borda muito fina (`1px solid #E2E8F0`).
* **Arredondamento (Border Radius):** `4px` a `8px` para cartões e botões.
* **Sombra (Box Shadow):** Sombra sutil para efeito de elevação:
  `box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);`
* **Estilo do Gráfico Principal (Área Sumarizada):**
  * Curva suave (Spline / Smooth curve).
  * Linhas finas com gradiente de preenchimento até o eixo X.
  * Múltiplas séries sobrepostas com opacidade controlada para transparência.

---

## 📐 Layout Otimizado para Quimiometria (Adsorção)


```

+-------------------------------------------------------------------------------------------------+
|  [Topo - Filtros e Seleção]: Amostra/Sistema | Adsorvente | Adsorvato | pH | Faixa Espectral     |
+-------------------------------------------------------------------------------------------------+
|  [KPI 1]              | [KPI 2]              | [KPI 3]             | [KPI 4]                    |
|  $q_{máx}$ (mg/g)     | R² do Modelo PCA/PLS | RMSEP (Erro Pred.)  | Capacidade Remanescente    |
|  +4.2% vs. anterior   | 0.9982               | 0.012 mg/L          | 87.5%                      |
+-------------------------------------------------------------------------------------------------+
|  [Gráfico Principal - Área/Linha 3D]                                                            |
|  Perfil Espectral FTIR / NIR ao Longo do Tempo de Adsorção                                      |
|  (Absorbância vs. Número de Onda $\bar{\nu}$ com overlay de perfis temporais)                   |
+-------------------------------------------------------------------------------------------------+
|  [Bloco Esquerdo 1]           | [Bloco Central]                  | [Bloco Direito]              |
|  Importância de Variáveis     | Gráfico de Scores PCA / PLS      | Métricas do Modelo Quimiom.  |
|  (VIP Scores / loadings)      | (PC1 vs. PC2 por Adsorvato)      | (R²C, R²P, RMSEC, RMSEP)     |
+-------------------------------------------------------------------------------------------------+

```

---

## 💡 Adaptação dos Painéis do Layout Original

1. **Faixa Superior de KPIs (6 Cartões Finais):**
   * **$q_{máx}$ Estimado:** Capacidade máxima de adsorção predita.
   * **R² (PLS/PCR):** Coeficiente de determinação da calibração.
   * **RMSEP:** Erro médio quadrático de predição.
   * **Cinética ($k_1 / k_2$):** Constante de taxa ajustada via regressão.
   * **Eficiência de Remoção (%):** Remoção do contaminante na batelada atual.
   * **Número de Amostras Processadas:** Total de espectros analisados no lote.

2. **Painel Central ("Network Activities" no original):**
   * **Gráfico de Área Espectral Curvo:** Visualização dos espectros (FTIR, RAMAN ou UV-Vis) em tempo real ou em função do tempo de contato. Preenchimento em gradiente teal/verde para destacar variações nas bandas de absorção características do processo de adsorção.

3. **Painéis Inferiores:**
   * **Importância de Variáveis (Barra Horizontal):** VIP Scores do PLS para identificar quais comprimentos de onda/números de onda são mais críticos no mecanismo de adsorção.
   * **Distribuição de Grupos / Donut Chart:** Agrupamento de amostras via HCA/Clusterização (ex: Tipos de biochar, matrizes de efluente, ou status de saturação).
   * **Tabela/Status Rápidos:** Relação dos parâmetros operacionais (pH, temperatura, dosagem do adsorvente).

```

cria um botao escuro/claro com cores que constrata