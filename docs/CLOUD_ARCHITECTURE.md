# Arquitetura de Nuvem — INNOVA EA (Fase 5)

Como hospedar o modelo treinado conectado ao terminal de execução, com
separação rígida entre **pesquisa/treino** e **execução ao vivo**.

## Restrição que define o desenho
O `MetaTrader5` (dados e ordens) é **Windows-only** e a execução precisa de
**baixa latência até o servidor do broker**. Logo, separamos dois planos:

- **Plano de Pesquisa/Treino** — Linux, GPU, em lote, escalável e barato (spot).
- **Plano de Execução** — VM **Windows** enxuta, próxima ao broker, sempre ligada.

Eles se comunicam **apenas** por um artefato de modelo versionado (object
storage). Pesquisa nunca toca em ordens; execução só consome artefatos aprovados.

## Visão geral

```
┌───────────────────────── PLANO DE PESQUISA / TREINO (Linux, GPU) ──────────────────────────┐
│                                                                                            │
│   Object Storage (Parquet)        Treino (GPU, spot)            Model Registry             │
│   S3 / GCS  ──────────────►  train_models.py (LightGBM/         S3 / GCS (versionado)       │
│   barras M1..D1, 15 ativos    Super Cérebro) + walk-forward ──► artefato + manifest.json    │
│        ▲                       purgado + Deflated Sharpe         (gate: DSR > limiar)        │
│        │ upload Parquet                                                  │                   │
└────────┼────────────────────────────────────────────────────────────────┼──────────────────┘
         │                                                                 │ pull do artefato aprovado
┌────────┼───────────────────────── PLANO DE EXECUÇÃO (Windows, low-latency) ──────────────────┐
│        │                                                                 ▼                    │
│   Ingestão MT5 ──► Parquet local ──► (sync p/ S3/GCS)        Serviço de Execução              │
│   (terminal MT5)                                              • carrega artefato + features   │
│                                                              • a cada barra: predict →         │
│   Terminal MT5  ◄──── order_send / posições ◄─────────────── ModelStrategy → travas de risco  │
│   (broker)                                                    (margem, stop-out, kill-switch) │
│                                                              • reconciliação de fills          │
└──────────────────────────────────────── Observabilidade / Alertas ───────────────────────────┘
```

## Fluxo de ponta a ponta
1. **Ingestão** (Windows): `ingest_mt5.py` extrai M1 desde 2015 → Parquet local →
   sincroniza para **S3/GCS** (`aws s3 sync` / `gsutil rsync`).
2. **Treino** (Linux GPU): job lê o Parquet do bucket, monta o painel universal,
   roda walk-forward purgado, treina o modelo final e publica o artefato +
   `manifest.json` no **Model Registry** versionado.
3. **Gate de promoção**: só artefatos com `deflated_sharpe` acima do limiar e
   Sharpe OOS estável são marcados como "aprovado". Champion-challenger: o novo
   só substitui o vigente se vencer nas mesmas janelas OOS.
4. **Execução** (Windows): o serviço puxa o artefato aprovado, e a cada nova
   barra calcula as features (`default_feature_set`), chama o modelo, converte em
   alvo via `ModelStrategy`, aplica as travas de risco da Fase 3 e envia ordens.
5. **Observabilidade**: latência fim-a-fim, P&L, drawdown, fills e nível de
   margem em métricas/alertas; **kill-switch** por perda diária/limite de risco.

## Mapeamento de serviços

| Função                     | AWS                                   | GCP                                  |
|----------------------------|---------------------------------------|--------------------------------------|
| Object storage (Parquet)   | **S3** (versionado)                   | **Cloud Storage**                    |
| Treino GPU (lote)          | EC2 **g5/g4dn** spot ou **SageMaker** | **Vertex AI Training** ou GCE **g2** |
| Model registry             | S3 versionado / SageMaker Model Reg.  | GCS versionado / Vertex Model Reg.   |
| Execução (Windows)         | **EC2 Windows** (c7/c6i), região do broker | **GCE Windows**, região do broker |
| Agendar re-treino          | **EventBridge** + Batch/Step Functions| **Cloud Scheduler** + Workflows      |
| Segredos (credenciais)     | **Secrets Manager**                   | **Secret Manager**                   |
| Observabilidade            | **CloudWatch** + SNS                  | **Cloud Monitoring** + Pub/Sub       |
| Rede/isolamento            | VPC privada, SG restritivos           | VPC, regras de firewall              |

## Dimensionamento sugerido (ponto de partida)
- **Treino**: 1× GPU (A10G/L4) spot para o Super Cérebro; LightGBM roda em CPU
  (c6i.4xlarge). Treino é batch e idempotente → spot com checkpoint é ideal.
- **Execução**: instância Windows pequena/estável (4–8 vCPU) **na mesma região do
  servidor do broker** (muitos brokers FX ficam em LD4/Equinix Londres → use a
  região mais próxima, ex. `eu-west-2`). Inferência por barra é leve (CPU basta);
  GPU não é necessária para servir.
- **Latência**: para M15 a latência não é crítica; se evoluir para timeframes
  baixos, considere colocation/VPS do próprio broker.

## Risco, segurança e governança
- **Travas**: reaproveitar `AccountConfig`/engine (margem, stop-out) também na
  execução; **kill-switch** por perda diária máxima e limite de exposição.
- **Segredos**: credenciais do broker em Secrets/Secret Manager — **nunca** no
  código nem em variáveis versionadas. IAM de menor privilégio.
- **Isolamento**: o plano de pesquisa não tem permissão de enviar ordens; o de
  execução só lê artefatos aprovados (bucket somente-leitura).
- **Auditoria**: versionar artefatos, manifests e o commit do código que treinou
  cada modelo (reprodutibilidade total).

## Roadmap de execução (Fase 5)
1. **Forward test em DEMO** (semanas) com o serviço de execução real, comparando
   fills vivos vs. o backtest — calibra spread/slippage reais.
2. Promoção gated por DSR + estabilidade OOS → **conta real pequena**.
3. Escala gradual de capital sob monitoramento de drawdown e kill-switch.
4. Re-treino agendado (champion-challenger) à medida que novos dados chegam.
