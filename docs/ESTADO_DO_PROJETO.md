# INNOVA EA — Estado do Projeto (handoff)

> Documento-mestre. Resume **tudo** que foi construído e decidido, pra qualquer
> sessão nova (ou pessoa) continuar de onde paramos, sem perder contexto.
> Última atualização: 2026-06-18.

---

## 1. O que é o INNOVA EA

Plataforma modular de **pesquisa e backtest** para trading algorítmico em
Forex/CFDs, com rigor de mesa institucional:

- **Custos sempre aplicados** (spread + slippage + comissão + swap).
- **Sem look-ahead por construção** (kernels causais).
- **Performance**: Python + Polars + NumPy + Numba (JIT) + LightGBM + PyTorch (opcional).
- **Honestidade acima de tudo**: toda estratégia é testada em dados aleatórios
  sintéticos (random walk). Se der Sharpe positivo lá, é bug/look-ahead. Tem que
  dar ~0 ou negativo. Só então confiamos no resultado em dado real.

## 2. Metodologia (a regra de ouro)

Padrão de cada estratégia nova:
1. `_kernel` (njit, causal) → 2. `*_outcomes` (DataFrame: net_ret/triggered/direction)
→ 3. `*_backtest` (equity, pos, trade_rets) → 4. `study_*.py` (walk-forward + Deflated Sharpe).

Validação obrigatória:
- **Walk-forward** por ativo (folds OOS), nunca in-sample.
- **Deflated Sharpe Ratio** (Bailey & López de Prado) — desconta o nº de tentativas.
- **Teste de honestidade**: rodar em GBM sintético → Sharpe deve ser ~0/negativo.
- Edge tem que ser **AMPLO** (vários ativos), não concentrado em um só.

## 3. Placar das estratégias (o que tem edge e o que não tem)

| Estratégia | Família | Veredito |
|---|---|---|
| **Straddle de Volatilidade** | breakout/vol | ✅ **VENCEDORA** — Sharpe +0.59, 12/15 ativos, 6/6 folds positivos (t≈4.9). Em forward test demo. |
| **INNOVA Breakout** (Williams+Unger) | breakout/vol | 🟡 nosso operacional próprio — honesto (Sharpe −1.57 em random walk). Rodar em dado real. |
| **Larry Williams** (volatility breakout) | breakout/vol | ✅ honesto e validado (campeão mundial 1987). |
| **Opening Range Breakout** (Andrea Unger) | breakout/vol | ✅ honesto e validado (tetracampeão mundial). |
| Meta-labeling | ML sobre primária | 🟡 depende da primária. |
| Pares / cointegração | relative-value | ❌ sem edge nos nossos dados. |
| Padrões de vela em suporte/resistência | reversão | ❌ sem edge (testado a pedido). |
| Direção pura por ML | trend/ML | ❌ fraco. |

**Conclusão central:** a ÚNICA família com edge real é **rompimento de
volatilidade**. Validada de forma independente por dois campeões mundiais
(Larry Williams 1987 e Andrea Unger 4×) — convergência forte.

## 4. O nosso operacional próprio — INNOVA Breakout

Funde os dois campeões num só sistema (`src/innova_ea/research/innova_breakout.py`):
- **Entrada**: rompimento do range da 1ª hora (Opening Range Breakout / Unger).
- **Gatilho de volatilidade**: só opera se o range de abertura ≥ `min_or_frac`
  do range do dia anterior (essência de Larry Williams — exige expansão).
- **Filtros de contexto (Unger)**: pula segunda-feira; pula dia após range
  extremo (`max_prior_mult`).
- **Saída honesta**: alvo/stop em unidades de risco + fechamento no fim do dia
  (sem o viés do "primeira abertura lucrativa").

Rodar:
```
python scripts/study_innova.py --store ./data --timeframe M15 --or-bars 4 --cost 0.0004
```

## 5. O operacional VENCEDOR em produção — Straddle

- Código: `src/innova_ea/research/straddle.py`.
- Serviço de execução: `src/innova_ea/execution/straddle_service.py` (OCO,
  expiração, kill-switch).
- Runner: `scripts/run_straddle_execution.py`.
- **Status atual:** rodando em **PAPER mode** numa VPS até terça, gerando
  `paper_log.txt`. Próximo passo: analisar o log (slippage real é a única
  incógnita que falta — o backtest já está no teto da evidência).

## 6. Como rodar (resumo)

```
pip install -e .                      # instala o pacote (a partir da raiz)
pytest -q                             # 152 testes, todos passando
python scripts/study_larry.py  ...    # estudo Larry Williams
python scripts/study_orb.py    ...    # estudo Opening Range Breakout (Unger)
python scripts/study_innova.py ...    # estudo INNOVA Breakout (nosso)
python scripts/run_straddle_execution.py ...   # execução paper/live do straddle
```
Dados (`data/`, `*.parquet`) NÃO vão pro git (ficam locais — `.gitignore`).

## 7. Documentação (Obsidian)

Em `docs/vault/` (abrir como vault no Obsidian):
- `INNOVA EA - MOC.md` (mapa de conteúdo / índice)
- `Metodologia.md`, `Placar de Estrategias.md`, `Glossario.md`
- `Arquitetura do Sistema.md`, `Operacao - Como Rodar.md`
- `Campeoes Mundiais de Trading.md` (top 3: Williams, Unger, Hughes)
- `Estrategia - *.md` (uma por estratégia)

## 8. Situação do GitHub (importante)

- O projeto vive hoje no repositório que era o fork `pythonbirds`, **branch
  padrão `simples`** — já limpo (curso antigo removido, INNOVA EA na raiz).
- O repo foi **renomeado** para `-INNOVA-EA` (atenção ao traço inicial no nome).
- **Problema:** ele ainda é um **fork** de `pythonprobr/pythonbirds`. Por isso:
  (a) mostra "derivado de" e (b) **não dá pra tornar privado** (regra do GitHub).
- **Solução em andamento:** pedido ao **GitHub Support** para *detach* do fork
  network → depois libera privado e some o "derivado de".
- **Plano:** criar um **repositório novo, privado, independente** (não-fork) e
  migrar tudo pra ele. Uma sessão nova do Claude Code aberta JÁ nesse repo novo
  passa a salvar tudo lá automaticamente.

### Limitação técnica conhecida
Cada sessão do Claude Code fica amarrada ao repositório escolhido na criação da
tarefa. Por isso esta sessão só consegue escrever no repo atual. Para o Claude
trabalhar/salvar no repo novo, **abrir a tarefa selecionando o repo novo**.

## 9. Próximos passos

1. [ ] Criar repo novo **privado e independente** (não-fork) no GitHub.
2. [ ] Migrar o projeto pra ele (push do ZIP / desta pasta).
3. [ ] Abrir sessão nova do Claude Code **selecionando o repo novo**.
4. [ ] Analisar o `paper_log.txt` do straddle (slippage real).
5. [ ] Rodar `study_innova.py` em dado real e registrar o resultado no placar.
6. [ ] (Opcional) Resolver o detach do fork via GitHub Support para o repo antigo.

---

*Tudo neste projeto foi construído com testes automatizados e validação honesta.
Nenhum resultado de backtest garante lucro futuro — forward test em conta demo
antes de qualquer capital real.*
