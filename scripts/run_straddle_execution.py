#!/usr/bin/env python
"""Execução LIVE do straddle de volatilidade — pronta para o forward test em demo.

Carrega o meta-modelo de straddle treinado, conecta ao MetaTrader 5 e roda o loop
event-driven: a cada barra fechada, arma o straddle (buy-stop + sell-stop com
SL/TP) quando P(rompimento lucrativo) ≥ limiar; gerencia OCO e expiração, sob o
kill-switch e as travas de margem.

SEGURANÇA: padrão = paper mode (``dry_run``). Use ``--live`` só após validar o
setup, e SEMPRE em conta DEMO primeiro.

Exemplo:
    python scripts/run_straddle_execution.py --artifacts ./art_full --timeframe H4 \\
        --threshold 0.50 --target-mult 2.5 --stop-mult 2.0
"""
from __future__ import annotations

import argparse
import logging
import os

from innova_ea.backtest.account import AccountConfig
from innova_ea.core.enums import Timeframe
from innova_ea.execution import RiskLimits, RiskManager, get_mt5_broker
from innova_ea.execution.straddle_service import StraddleConfig, StraddleExecutionService
from innova_ea.features.presets import default_feature_set
from innova_ea.research.metalabeling import MetaLabelModel
from innova_ea.universe import DEFAULT_UNIVERSE, asset_class_of


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Execução live do straddle (INNOVA EA)")
    ap.add_argument("--artifacts", default="./art_full",
                    help="pasta com o artefato 'straddle' (saída do train_straddle).")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--timeframe", default="H4")
    ap.add_argument("--lots", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--band-lookback", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--target-mult", type=float, default=2.5)
    ap.add_argument("--stop-mult", type=float, default=2.0)
    ap.add_argument("--atr-window", type=int, default=24)
    ap.add_argument("--lookback-bars", type=int, default=300)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--leverage", type=float, default=100.0)
    ap.add_argument("--daily-max-loss", type=float, default=0.05)
    ap.add_argument("--server-tz", default=os.getenv("MT5_SERVER_TZ", "Europe/Athens"))
    ap.add_argument("--server-utc-offset", type=float, default=None)
    ap.add_argument("--live", action="store_true", help="ENVIA ordens (padrão: paper).")
    args = ap.parse_args()

    model = MetaLabelModel.load(os.path.join(args.artifacts, "straddle"))
    feature_set = default_feature_set()
    # Opera só os símbolos que o modelo conhece.
    symbols = [s for s in args.symbols if s in model.asset_vocab]

    broker = get_mt5_broker(
        login=int(os.environ["MT5_LOGIN"]) if os.getenv("MT5_LOGIN") else None,
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH"),
        server_timezone=None if args.server_utc_offset is not None else args.server_tz,
        server_utc_offset=args.server_utc_offset,
    )
    risk = RiskManager(AccountConfig(leverage=args.leverage),
                       RiskLimits(daily_max_loss_pct=args.daily_max_loss))
    cfg = StraddleConfig(
        timeframe=Timeframe(args.timeframe), lookback_bars=args.lookback_bars,
        band_lookback=args.band_lookback, horizon=args.horizon,
        target_mult=args.target_mult, stop_mult=args.stop_mult, atr_window=args.atr_window,
        threshold=args.threshold, lots=args.lots, poll_seconds=args.poll_seconds,
        dry_run=not args.live,
    )
    service = StraddleExecutionService(broker, model, feature_set, symbols,
                                       asset_class_of, risk, cfg)

    logging.info("INNOVA EA straddle — modo %s | %d símbolos | %s",
                 "LIVE" if args.live else "PAPER", len(symbols), args.timeframe)
    try:
        service.run()
    except KeyboardInterrupt:
        service.stop()
        logging.info("execução interrompida pelo operador")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
