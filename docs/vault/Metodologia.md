# Metodologia

> Os pilares anti-autoengano do [[INNOVA EA - MOC|INNOVA EA]]. A regra número um da pesquisa quantitativa: **não enganar a si mesmo**. Cada pilar abaixo existe para tornar o autoengano mais difícil.

## 1. Sem look-ahead (causalidade rigorosa)

Uma estratégia só pode usar informação que **realmente existia** no momento da decisão. Na prática:

- As features são **causais**: nada do futuro vaza para o passado.
- O sinal decidido no **fechamento da barra `i`** só é executado na **abertura da barra `i+1`**. Você não negocia com um preço que ainda não aconteceu.
- O **walk-forward é PURGADO** (purga + embargo): quando os rótulos se sobrepõem no tempo, removemos as amostras de treino que se encavalam com o período de teste (purga) e deixamos um intervalo de segurança (embargo). Isso impede que o modelo "veja" indiretamente o futuro através de rótulos sobrepostos.

Ver [[Glossario]] para *look-ahead bias* e *walk-forward purgado*.

## 2. Custos sempre aplicados

Resultados **SEMPRE líquidos de custo**. Um backtest sem custos é ficção. Aplicamos:

- **Meio-spread** (metade do spread, pago na entrada e na saída)
- **Slippage** (a diferença entre o preço esperado e o executado)
- **Comissão round-turn** (ida e volta)
- **Swap** (custo de carregar a posição overnight)

Muitas estratégias parecem ótimas no bruto e morrem no líquido. É exatamente esse o teste que importa.

## 3. Margem e stop-out estilo MT5

O backtest simula a realidade de uma corretora MT5:

- **Bloqueio de ordens por margem livre**: se não há margem, a ordem não entra.
- **Liquidação forçada (stop-out)**: posições são liquidadas se o nível de margem cai demais.

Isso evita o erro clássico de assumir capital infinito e alavancagem ilimitada.

## 4. Deflated Sharpe Ratio (DSR) e Probabilistic Sharpe

Baseados em **Bailey & López de Prado**. O problema: se você testa muitas configurações, alguma vai parecer boa **por puro acaso**. O DSR e o PSR **penalizam o número de tentativas** — descontam a sorte associada a testar muitas variantes. Um Sharpe alto com DSR baixo é provavelmente sorte; um edge fino mas com consistência forte é mais confiável.

Ver [[Glossario]] para *Sharpe* e *Deflated Sharpe Ratio*.

## 5. Triple-barrier para rotulagem

Método de rotulagem de **López de Prado**. Em vez de rotular pelo retorno de horizonte fixo, definimos três barreiras: um alvo de lucro, um stop de perda e um limite de tempo. O rótulo é determinado por **qual barreira é atingida primeiro**. Isso reflete como um trade realmente termina.

Ver [[Glossario]] para *triple-barrier*.

## 6. Forward test em conta DEMO obrigatório

Antes de qualquer capital real, a estratégia roda em **conta DEMO** (dinheiro fake, fluxo de execução real). É o teste final fora da amostra, em condições de mercado ao vivo, sem o risco de perder dinheiro de verdade. Nenhuma estratégia pula esta etapa.

Ver [[Operacao - Como Rodar]] e [[Glossario]] (*forward test*).

## Por que tudo isso junto

Cada pilar fecha uma porta de autoengano. Juntos, eles fazem o sistema **reprovar o que não merece** — inclusive ideias famosas e populares. Foi assim que o [[Placar de Estrategias|placar]] reprovou pares e padrões de vela e aprovou apenas o straddle de volatilidade.

## Relacionado

- [[INNOVA EA - MOC]]
- [[Placar de Estrategias]]
- [[Glossario]]
- [[Estrategia - Volatilidade (Straddle)]]
