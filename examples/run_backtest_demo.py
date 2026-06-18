"""Demo end-to-end: dados → storage → backtest com risco → walk-forward → DSR.

Roda 100% offline com dados sintéticos (sem MT5/rede), provando o fluxo completo
da plataforma INNOVA EA. Em produção, basta trocar ``SyntheticSource`` por
``get_mt5_source()``.

    python examples/run_backtest_demo.py
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone

from innova_ea.backtest import AccountConfig, BacktestEngine, CostModel, WalkForward
from innova_ea.backtest.metrics import periods_per_year
from innova_ea.core import Timeframe, get_instrument
from innova_ea.data import DataPipeline, ParquetStore, SyntheticSource
from innova_ea.research import deflated_sharpe_ratio, deannualize_sharpe
from innova_ea.strategy import MovingAverageCrossover


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

        bars = store.read(symbol, trade_tf, start, end)

        # 2) Backtest único com custos + RISCO (margem/stop-out estilo MT5).
        inst = get_instrument(symbol)
        costs = CostModel(slippage_points=2.0)
        account = AccountConfig(leverage=100, margin_call_level=1.0, stop_out_level=0.5)
        engine = BacktestEngine(inst, costs, account,
                                initial_capital=10_000.0, max_lots=0.1)
        strategy = MovingAverageCrossover(fast=20, slow=50)
        result = engine.run(bars, strategy, trade_tf)

        print(f"\n== Backtest ({trade_tf.value}, {bars.height:,} barras) — "
              f"{strategy.name} ==\n")
        print(result.report.summary())
        print("\n-- Risco / Margem (MT5) --")
        print(result.risk.summary())

        # 3) Walk-forward: avaliação out-of-sample com janelas deslizantes.
        wf = WalkForward(train_size=8000, test_size=4000)
        wf_result = wf.run(
            bars,
            strategy_factory=lambda train: MovingAverageCrossover(20, 50),
            engine_factory=lambda: BacktestEngine(
                inst, costs, account, initial_capital=10_000.0, max_lots=0.1
            ),
            timeframe=trade_tf,
        )
        print("\n== Walk-forward (out-of-sample) ==\n")
        print(wf_result.summary())

        # 4) Anti-overfitting: Deflated Sharpe sobre os Sharpes das janelas OOS.
        ppy = periods_per_year(trade_tf)
        trial_sr = [deannualize_sharpe(s, ppy) for s in wf_result.oos_sharpes]
        dsr = deflated_sharpe_ratio(trial_sr, n_obs=bars.height)
        print(f"\nDeflated Sharpe Ratio: {dsr:.2%} "
              f"(prob. de o edge ser real após {len(trial_sr)} janelas)")

        print(
            "\nNota: dados SINTÉTICOS (random walk) — espera-se ausência de edge. "
            "O objetivo é validar o rigor da plataforma, não inferir lucro."
        )


if __name__ == "__main__":
    main()
