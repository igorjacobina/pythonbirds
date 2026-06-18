# Operação - Como Rodar

> Scripts principais do [[INNOVA EA - MOC|INNOVA EA]] e a ordem de operação. Esta nota apenas **referencia** os scripts existentes — consulte cada arquivo para opções detalhadas. Ver [[Arquitetura do Sistema]].

## Scripts principais

### Baixar dados (Dukascopy)
```
scripts/ingest_external.py
```
Ingestão de dados externos (Dukascopy, tick→M1) para o ParquetStore. Primeiro passo de qualquer pesquisa — sem dados limpos e em UTC, nada do resto faz sentido. Ver camada **data** em [[Arquitetura do Sistema]].

### Treinar / validar o candidato (straddle)
```
scripts/train_straddle.py
```
Treina e valida a estratégia vencedora — [[Estrategia - Volatilidade (Straddle)|straddle de rompimento de volatilidade]]. Produz as métricas de walk-forward (Sharpe OOS, folds, DSR).

### Execução paper / demo (straddle)
```
scripts/run_straddle_execution.py
```
Roda a execução event-driven do straddle em modo **paper** ou **demo**. É o que leva a estratégia ao forward test.

### Estudar pares
```
scripts/study_pairs.py
```
Pesquisa de relative-value / pares — ver [[Estrategia - Pares (Relative-Value)]] (reprovada).

### Treinar meta-labeling
```
scripts/train_meta.py
```
Treina o filtro de [[Estrategia - Meta-labeling|meta-labeling]] (reprovado).

## Ordem de operação típica

1. `ingest_external.py` — garantir os dados.
2. `train_straddle.py` — validar o candidato no walk-forward.
3. `run_straddle_execution.py` — paper, depois demo.

## ⚠️ Aviso sobre modos de execução

- **Paper mode**: simulação sem ordens reais. **Risco zero** — nem dinheiro fake.
- **Demo**: conta com **dinheiro fake**, mas com fluxo de execução real (preços, latência, slippage do ambiente).
- **Sempre validar no DEMO antes de qualquer capital real.** O forward test em conta demo é **obrigatório** (ver [[Metodologia]] e [[Glossario]]). Nenhuma estratégia pula essa etapa, nem mesmo o straddle aprovado.

## Relacionado

- [[INNOVA EA - MOC]]
- [[Arquitetura do Sistema]]
- [[Estrategia - Volatilidade (Straddle)]]
- [[Metodologia]]
- [[Glossario]]
