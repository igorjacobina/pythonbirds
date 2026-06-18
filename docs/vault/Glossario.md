# Glossário

> Definições simples dos termos técnicos usados no [[INNOVA EA - MOC|INNOVA EA]]. Sem jargão desnecessário.

## Sharpe Ratio

Mede o **retorno ajustado ao risco**: quanto retorno você ganha por unidade de volatilidade (oscilação). Quanto maior, melhor. Um Sharpe negativo significa que a estratégia perde dinheiro de forma ineficiente. É a régua básica para comparar estratégias. Ver [[Placar de Estrategias]].

## Deflated Sharpe Ratio (DSR)

Uma versão "honesta" do Sharpe que **penaliza o número de tentativas** (Bailey & López de Prado). Se você testa muitas variantes, alguma vai parecer boa por **acaso**. O DSR desconta essa sorte. Um Sharpe alto mas DSR baixo é provavelmente sorte; o DSR ajuda a separar **skill de sorte**. Ver [[Metodologia]].

## Walk-forward (purgado)

Forma de validar uma estratégia avançando no tempo: treina-se num período passado, testa-se no período seguinte, e repete-se "rolando" para frente — imitando como seria operar na vida real. A versão **purgada** remove amostras de treino que se sobrepõem ao teste (**purga**) e deixa um intervalo de segurança (**embargo**), para que rótulos sobrepostos não vazem informação do futuro. Ver [[Metodologia]].

## Look-ahead bias

O erro de usar, no momento da decisão, informação que **só estaria disponível no futuro**. Causa backtests fantasticamente bons e impossíveis de reproduzir ao vivo. O INNOVA EA evita isso com features causais e executando o sinal da barra `i` apenas na abertura de `i+1`. Ver [[Metodologia]].

## Triple-barrier

Método de **rotulagem** (López de Prado): para cada trade, definem-se três barreiras — alvo de lucro, stop de perda e limite de tempo. O rótulo é dado por **qual barreira é atingida primeiro**. Reflete como um trade realmente termina, em vez de usar um retorno de horizonte fixo. Ver [[Metodologia]].

## Overfitting

Quando um modelo "decora" o ruído dos dados de treino em vez de aprender o sinal real. Parece excelente no passado e falha no futuro. As defesas do projeto contra isso são o walk-forward purgado, os custos reais e o DSR. Ver [[Metodologia]].

## Slippage

A diferença entre o preço que você **esperava** executar e o preço que **realmente** saiu. Faz parte dos custos de transação. Ignorá-lo infla resultados. No INNOVA EA, slippage entra sempre no cálculo, junto de spread, comissão e swap. Ver [[Metodologia]].

## Cointegração

Propriedade estatística de dois ativos cujos preços **andam juntos no longo prazo**, mesmo oscilando no curto prazo. É a base do *pairs trading* / relative-value: aposta-se que o spread entre eles volta à média. Quando a cointegração **se quebra** (como ouro/prata na última década), a estratégia falha. Ver [[Estrategia - Pares (Relative-Value)]].

## Forward test (demo)

Teste final **fora da amostra**, rodando a estratégia ao vivo em **conta DEMO** (dinheiro fake, execução real) antes de qualquer capital real. É a etapa que confirma se o edge do backtest sobrevive em condições de mercado reais. Obrigatório no INNOVA EA. Ver [[Operacao - Como Rodar]].

## Relacionado

- [[INNOVA EA - MOC]]
- [[Metodologia]]
- [[Placar de Estrategias]]
