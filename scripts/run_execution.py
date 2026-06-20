#!/usr/bin/env python
"""Serviço de Execução Live do INNOVA EA — pronto para plugar o artefato treinado.

Carrega o modelo treinado, conecta ao terminal MetaTrader 5 e roda o loop
event-driven (a cada barra fechada): features → modelo → conciliação contra a
corretora → ordem, sob as travas de risco e o kill-switch.

SEGURANÇA: por padrão roda em **paper mode** (``dry_run``) — apenas registra as
decisões, sem enviar ordens. Use ``--live`` para operar de verdade (faça forward
test em DEMO antes!).

Exemplos:
    # Paper mode (recomendado para validar o setup):
    python scripts/run_execution.py --model lightgbm --artifacts ./artifacts --symbols EURUSD XAUUSD

    # Operação real em conta DEMO:
    python scripts/run_execution.py --model lightgbm --artifacts ./artifacts \
        --symbols EURUSD --live --max-lots 0.1 --daily-max-loss 0.05
"""
from __future__ import annotations

import argparse
import logging
import os

from innova_ea.backtest.account import AccountConfig
from innova_ea.core.enums import Timeframe
from innova_ea.execution import (
    ExecutionConfig,
    ExecutionService,
    RiskLimits,
    RiskManager,
    build_model_strategies,
    get_mt5_broker,
)
from innova_ea.features.presets import default_feature_set
from innova_ea.universe import DEFAULT_UNIVERSE, asset_class_of


def load_model(kind: str, artifacts_root: str):
    """Carrega o artefato (LightGBM ou Super Cérebro) da pasta de artefatos."""
    path = os.path.join(artifacts_root, kind)
    if kind == "lightgbm":
        from innova_ea.research.models.gbm import LightGBMUniversal
        return LightGBMUniversal.load(path)
    if kind == "super_brain":
        from innova_ea.research.models.super_brain import SuperBrain
        return SuperBrain.load(path)
    raise SystemExit(f"modelo desconhecido: {kind}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Execução live INNOVA EA")
    ap.add_argument("--model", choices=["lightgbm", "super_brain"], default="lightgbm")
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--max-lots", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--lookback-bars", type=int, default=300)
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--account-currency", default="USD")
    ap.add_argument("--daily-max-loss", type=float, default=0.05,
                    help="fração do equity de início do dia (kill-switch).")
    ap.add_argument("--max-consecutive-errors", type=int, default=5)
    ap.add_argument("--server-tz", default=os.getenv("MT5_SERVER_TZ", "Europe/Athens"))
    ap.add_argument("--server-utc-offset", type=float, default=None)
    ap.add_argument("--live", action="store_true",
                    help="ENVIA ordens de verdade (padrão é paper/dry-run).")
    args = ap.parse_args()

    model = load_model(args.model, args.artifacts)
    feature_set = default_feature_set()
    strategies = build_model_strategies(
        model, feature_set, args.symbols, asset_class_of, threshold=args.threshold)

    broker = get_mt5_broker(
        login=int(os.environ["MT5_LOGIN"]) if os.getenv("MT5_LOGIN") else None,
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH"),
        server_timezone=None if args.server_utc_offset is not None else args.server_tz,
        server_utc_offset=args.server_utc_offset,
    )
    risk = RiskManager(
        AccountConfig(leverage=args.leverage, account_currency=args.account_currency),
        RiskLimits(daily_max_loss_pct=args.daily_max_loss,
                   max_consecutive_errors=args.max_consecutive_errors),
    )
    cfg = ExecutionConfig(
        timeframe=Timeframe(args.timeframe),
        lookback_bars=args.lookback_bars,
        max_lots=args.max_lots,
        poll_seconds=args.poll_seconds,
        dry_run=not args.live,
    )
    service = ExecutionService(broker, strategies, risk, cfg)

    mode = "LIVE (ordens reais)" if args.live else "PAPER (dry-run)"
    logging.info("INNOVA EA execução — modo %s | %d símbolos | modelo %s",
                 mode, len(strategies), args.model)
    try:
        service.run()
    except KeyboardInterrupt:
        service.stop()
        logging.info("execução interrompida pelo operador")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
