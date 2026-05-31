"""Demo end-to-end: dados → storage → resample → backtest → métricas.

Roda 100% offline com dados sintéticos (sem MT5/rede), provando o fluxo completo
da plataforma. Em produção, basta trocar ``SyntheticSource`` por ``get_mt5_source()``.

    python examples/run_backtest_demo.py
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone

from forex_quant.backtest import BacktestEngine, CostModel
from forex_quant.core import Timeframe, get_instrument
from forex_quant.data import DataPipeline, ParquetStore, SyntheticSource
from forex_quant.strategy import MovingAverageCrossover


def main() -> None:
    symbol = "EURUSD"
    base_tf = Timeframe.M15
    trade_tf = Timeframe.H1
    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 1, 1, tzinfo=timezone.utc)

    # 1) Pipeline: ingere M15 sintético e deriva H1/H4/D1 em Parquet.
    source = SyntheticSource(start_price=1.10, annual_vol=0.08, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        store = ParquetStore(tmp)
        pipeline = DataPipeline(source, store)
        report = pipeline.ingest(
            symbol, base_tf, start, end,
            derive=[Timeframe.H1, Timeframe.H4, Timeframe.D1],
        )
        print("== Ingestão ==")
        print(f"  barras {base_tf.value} buscadas: {report.bars_fetched:,}")
        for tf, n in report.bars_written.items():
            print(f"  gravadas {tf}: {n:,}")

        # 2) Lê o timeframe de operação do storage.
        bars = store.read(symbol, trade_tf, start, end)
        print(f"\n== Backtest ({trade_tf.value}, {bars.height:,} barras) ==")

        # 3) Backtest com custos realistas.
        inst = get_instrument(symbol)
        costs = CostModel(slippage_points=2.0)  # spread/comissão/swap vêm do instrumento
        # max_lots=0.1 => sizing conservador (~1 micro-lote por unidade de risco).
        # Margem/stop-out ainda não são modelados (roadmap Fase 3); manter o
        # sizing baixo evita que a equity fure zero artificialmente.
        engine = BacktestEngine(inst, costs, initial_capital=10_000.0, max_lots=0.1)
        strategy = MovingAverageCrossover(fast=20, slow=50)
        result = engine.run(bars, strategy, trade_tf)

        print(f"\nEstratégia: {strategy.name}\n")
        print(result.report.summary())

        print(
            "\nNota: resultado em dados SINTÉTICOS — serve para validar o "
            "pipeline, não para inferir lucratividade real."
        )


if __name__ == "__main__":
    main()
