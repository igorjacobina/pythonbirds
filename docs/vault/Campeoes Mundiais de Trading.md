# Campeões Mundiais de Trading

> Os maiores operacionais validados no **World Cup Trading Championship**. O
> INNOVA EA testa os métodos SISTEMÁTICOS deles com o mesmo rigor honesto.

## O Top 3 de todos os tempos

| # | Campeão | Feito | Método (testável?) |
|---|---------|-------|--------------------|
| 1 | **Larry Williams** | 1987: US$10k → US$1,14M (**11.376%**, recorde até hoje) | **Volatility breakout** ✅ testável |
| 2 | **Andrea Unger** | **Único TETRACAMPEÃO** (2008, 2009, 2010, 2012); 672% em 2008 | **Opening Range Breakout** sistemático ✅ testável |
| 3 | **Chuck Hughes** | Múltiplos títulos (futuros e ações) | Baseado em **opções** (LEAPS, spreads) ❌ não testável com nossos dados |

## A convergência que importa

Os métodos **sistemáticos e testáveis** dos campeões convergem para **rompimento
de volatilidade / breakout** — exatamente a ÚNICA família que o INNOVA EA
aprovou no [[Placar de Estrategias|placar]], de forma independente:

- **Larry Williams** → rompimento da abertura ± k·range (ver [[Estrategia - Larry Williams]]).
- **Andrea Unger** → rompimento do range da **primeira hora** (Opening Range Breakout),
  com filtros lógicos como "deixar a segunda-feira de fora".

E a **filosofia de Unger** é idêntica à nossa: começar com ideias simples e
robustas, validar o edge estatístico básico, e só então adicionar filtros
LÓGICOS de contexto — nunca parâmetros otimizados no escuro. É o oposto do
*overfitting* (ver [[Metodologia]]).

> **Chuck Hughes** opera **opções**, que não temos nos dados (Forex/CFD via MT5),
> então não é testável aqui — honestidade antes de tudo.

## Como o INNOVA EA testa

- Larry Williams: `scripts/study_larry.py` (volatility breakout diário).
- Andrea Unger: `scripts/study_orb.py` (opening range breakout intradiário).

Ambos passam pelo mesmo filtro: custos reais, walk-forward, [[Glossario|Deflated Sharpe]].
Em dados aleatórios (sem edge), os dois dão Sharpe negativo — prova de que a
avaliação é honesta e não infla resultado.

## Relacionado

- [[Estrategia - Larry Williams]]
- [[Estrategia - Volatilidade (Straddle)]]
- [[Placar de Estrategias]]
- [[Metodologia]]
- [[INNOVA EA - MOC]]
