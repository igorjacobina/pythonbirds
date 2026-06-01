# Manual de Implantação — INNOVA EA (Fase 5)

Roteiro exato para sair do laboratório: extrair dados reais no Windows (MT5),
treinar os modelos e preparar a execução. Três planos: **ingestão (Windows)**,
**treino (Windows ou nuvem)** e **execução (Windows próximo ao broker)**.

---

## 1. Deploy local no Windows — ingestão dos 15 ativos

### 1.1 Pré-requisitos
- **Windows 10/11** (o pacote `MetaTrader5` é Windows-only).
- **Terminal MetaTrader 5** instalado e **logado** na sua conta do broker.
  - Em *Ferramentas → Opções → Gráficos → Máx. barras nos gráficos*: defina
    **Ilimitado** (essencial para histórico M1 profundo desde 2015).
  - Habilite *AlgoTrading* e deixe os 15 símbolos visíveis no *Market Watch*.
- **Python 3.11** (recomendado; compatível com numba/polars). `git`.

### 1.2 Clonar a branch e criar o ambiente isolado
Use um **venv** para não quebrar dependências do sistema. No **PowerShell**:

```powershell
git clone https://github.com/igorjacobina/pythonbirds.git
cd pythonbirds
git checkout claude/forex-ai-strategy-s0SEV
cd innova_ea

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Núcleo (Polars, NumPy, Numba, PyArrow, LightGBM) + o pacote em modo editável:
pip install -r requirements.txt
pip install -e .

# Conector MT5 (somente Windows):
pip install MetaTrader5
```

> Se o PowerShell bloquear a ativação do venv, rode uma vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 1.3 Credenciais (variáveis de ambiente)
No **PowerShell** (sessão atual):

```powershell
$env:MT5_LOGIN    = "12345678"
$env:MT5_PASSWORD = "sua_senha"
$env:MT5_SERVER   = "MeuBroker-Demo"
# Opcional: caminho do terminal, se não for o padrão
# $env:MT5_PATH   = "C:\Program Files\MetaTrader 5\terminal64.exe"
```

### 1.4 Disparar a ingestão (M1, 2015 → hoje, 15 ativos)
```powershell
python scripts\ingest_mt5.py --start 2015-01-01 --store .\data
```

- EUR/USD é processado **primeiro** (obrigatório), depois metais e índices.
- Para cada ativo: baixa M1 em chunks mensais → limpeza defensiva → grava
  Parquet particionado → deriva M5/M15/M30/H1/H4/D1 → relatório de gaps.
- **Fuso**: por padrão usa `Europe/Athens` (EET com DST). Se o seu broker usa
  offset fixo: `--server-utc-offset 2` (ou 3). Confirme com o suporte do broker.
- **Nomes de símbolo variam por corretora.** Se algum não existir, ajuste:
  ```powershell
  python scripts\ingest_mt5.py --symbols EURUSD XAUUSD NAS100 GER40 --start 2015-01-01
  ```
  Aliases frequentes: `US100≈NAS100`, `DE40≈GER40`, `UK100≈FTSE100`, `HK50≈HSI`.

> ⚠️ A primeira extração é pesada (M1 de 15 ativos × ~10 anos): horas de download
> e alguns GB em disco. É **idempotente e retomável** — se cair, basta rodar de
> novo; o `ParquetStore` deduplica e completa o que faltar.

### 1.5 Conferir os dados
```powershell
python -c "from innova_ea.data import ParquetStore; s=ParquetStore('./data'); print(s.available_symbols())"
```

### 1.6 Histórico profundo via fonte externa (recomendado p/ treino)

Contas demo de **prop firms (FTMO etc.) não servem histórico profundo de M1** —
elas são mesas de execução, não provedores de dados. Para a base de treino densa
(M1 desde 2015) dos 15 ativos, use um provedor dedicado. A FTMO/MT5 fica **só
para a execução ao vivo** (plano separado, como em `CLOUD_ARCHITECTURE.md`).

```powershell
# Dukascopy (download direto; profundidade máxima — FX, metais e índices):
python scripts\ingest_external.py --source dukascopy --start 2015-01-01 --store .\data

# HistData (baixe antes os CSVs mensais p/ um diretório; cobre 14 dos 15 ativos —
# Dow/US30 só existe na Dukascopy):
python scripts\ingest_external.py --source histdata --csv-dir .\histdata_raw --store .\data
```

Ambas gravam no MESMO `ParquetStore` que o treino lê. Dica de **dupla fonte**:
índices/Dow via Dukascopy; FX/metais por qualquer uma. Os códigos de símbolo são
mapeados internamente (DAX=`GRXEUR`/`DEUIDXEUR`, Nikkei=`JPXJPY`/`JPNIDXJPY`, etc.).

> ⚠️ Dukascopy baixa por hora (arquivos `.bi5`); a primeira carga de 11 anos é
> longa, porém idempotente/retomável (o `ParquetStore` deduplica). Para índices,
> confirme o código no datafeed se algum vier vazio.

---

## 2. Treinamento dos modelos (mercado real)

Assim que o Parquet estiver pronto, treine o baseline e/ou o Super Cérebro.
Roda no mesmo Windows ou — preferível para o Transformer — numa GPU na nuvem
(seção 3), apontando para o mesmo `./data` (ou um bucket S3/GCS).

```powershell
# Baseline LightGBM (CPU, minutos):
python scripts\train_models.py --model lightgbm --store .\data --out .\artifacts

# Super Cérebro (Transformer; requer PyTorch — GPU fortemente recomendada):
pip install torch                      # CPU; para CUDA veja pytorch.org
python scripts\train_models.py --model super_brain --store .\data --out .\artifacts

# Ambos de uma vez:
python scripts\train_models.py --model both --store .\data --out .\artifacts
```

O script:
1. carrega todos os ativos do Parquet e monta o **painel universal**;
2. roda **walk-forward purgado** medindo Sharpe OOS por ativo (com custos/risco)
   e o **Deflated Sharpe** agregado — a prova honesta de edge;
3. treina o **modelo final** em todo o histórico e salva o artefato em
   `./artifacts/<modelo>/` com um `manifest.json` (período, métricas, DSR).

**Critério de promoção (gate):** só leve para execução um artefato cujo
`deflated_sharpe` seja convincente (ex. > 0,95) e cujo Sharpe OOS seja positivo
e estável entre janelas. Caso contrário, **não opere** — ajuste features/rótulos.

Parâmetros úteis: `--timeframe M15`, `--max-horizon 16`, `--train-size`,
`--test-size`, `--threshold` (zona morta do sinal), `--no-eval` (pula validação).

---

## 3. Execução live

A execução roda na mesma VM Windows (terminal MT5 logado), carregando o artefato
promovido. **Padrão seguro: paper mode** — só registra decisões, não envia ordens.

```powershell
# Paper mode (valide o setup primeiro):
python scripts\run_execution.py --model lightgbm --artifacts .\artifacts --symbols EURUSD XAUUSD

# Conta DEMO (ordens reais em demo), com travas:
python scripts\run_execution.py --model lightgbm --artifacts .\artifacts `
    --symbols EURUSD --live --max-lots 0.1 --daily-max-loss 0.05 --leverage 100
```

A cada barra fechada: features → modelo → **conciliação** contra a posição real
(sem duplicidade/órfãs) → travas de **margem** → ordem. O **Kill-Switch** zera a
carteira e bloqueia entradas se a perda diária ultrapassar `--daily-max-loss` ou
a conexão ficar instável. `Ctrl+C` encerra com desconexão limpa.

> Sempre faça **forward test em conta DEMO** por semanas antes de capital real.
> Backtest e walk-forward não substituem o mercado ao vivo. Arquitetura de nuvem
> (planos separados de pesquisa e execução) em
> [`CLOUD_ARCHITECTURE.md`](CLOUD_ARCHITECTURE.md).
