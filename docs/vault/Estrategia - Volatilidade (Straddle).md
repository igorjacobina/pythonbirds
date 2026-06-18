# Estratégia - Volatilidade (Straddle)

> Família 4 do [[Placar de Estrategias|placar]]. **Veredito: ✅ EDGE REAL, FINO.** A única que sobreviveu. Atualmente em **forward test no DEMO**.

## A ideia: rompimento de volatilidade, direção-agnóstico

Um **straddle de rompimento** não tenta adivinhar **para onde** o preço vai — aposta no **tamanho** do movimento. A lógica:

1. Define-se uma faixa em torno do preço (derivada da volatilidade recente).
2. Colocam-se ordens stop dos **dois lados**: uma para comprar se o preço romper para cima, outra para vender se romper para baixo.
3. Quando o mercado **rompe** com força (expansão de volatilidade), a ordem do lado certo dispara e captura o movimento.

A aposta é: **quando a volatilidade explode, o movimento tende a continuar** — independentemente da direção. Isso é fundamentalmente diferente de prever alta vs. baixa (que falhou em [[Estrategia - Direcao (ML)]]).

## Resultados (H4, custo REALISTA de 40 pontos, threshold 0,50, 15 ativos)

- **Sharpe OOS +0,59**
- **12/15 ativos positivos** — edge **AMPLO**, não concentrado num único ativo
- **6/6 folds positivos** — **t-stat ≈ 4,9, p < 0,001** (a evidência mais limpa)
- **21.770 trades** — sobreviveu inclusive ao aumento do número de trades
- **DSR conservador ~31%** — medida calculada de forma conservadora; a consistência 6/6 folds é a evidência mais forte

### Núcleo forte (FX majors + prata)

| Ativo | Sharpe |
|-------|--------|
| XAGUSD (prata) | **+2,21** |
| GBPUSD | +1,30 |
| USDCHF | +1,17 |
| USDCAD | +1,09 |
| EURUSD | +1,07 |

### Pontos fracos (índices de ações)

US30, DE40 e HK50 ficaram **levemente negativos**. Índices de ações são o terreno onde o rompimento de volatilidade funciona pior.

## Por que faz sentido (coerência econômica)

O edge é **economicamente coerente**: o rompimento de volatilidade funciona melhor em **FX e metais** do que em **índices de ações**. Câmbio e metais têm dinâmicas de expansão de volatilidade (fluxos macro, eventos cambiais) que produzem movimentos sustentados após rompimento; índices de ações tendem a reverter mais. O resultado por ativo reflete essa lógica — não é um padrão arbitrário ajustado aos dados.

## Por que confiamos

- **Consistência**: 6/6 folds positivos com t-stat ≈ 4,9 é difícil de obter por acaso.
- **Amplitude**: 12/15 ativos positivos — não depende de um outlier.
- **Robustez a custo**: sobreviveu a 40 pontos de custo realista.
- **Robustez a trades**: continua positivo com 21.770 trades.
- **Validação externa**: é da **mesma família** que o método de [[Estrategia - Larry Williams|Larry Williams]], descoberto de forma totalmente independente.

O edge é **fino**, não espetacular — e é honesto chamá-lo assim. Mas é real.

## Estado atual

Em **forward test em conta DEMO** — etapa obrigatória antes de capital real. Ver [[Operacao - Como Rodar]] (`scripts/train_straddle.py`, `scripts/run_straddle_execution.py`).

## Relacionado

- [[Placar de Estrategias]]
- [[Estrategia - Larry Williams]]
- [[Estrategia - Direcao (ML)]]
- [[Metodologia]]
- [[Operacao - Como Rodar]]
- [[Glossario]]
