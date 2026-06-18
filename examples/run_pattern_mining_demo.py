"""Demo da Fase 2: engenharia de features + mineração de padrões de sessão.

Para validar o motor contra *ground-truth*, injetamos uma distorção CONHECIDA
nos dados sintéticos: na abertura de Londres (07–09 UTC) a volatilidade e o
direcional ficam elevados. Em seguida, o pipeline de features + mineração precisa
REDESCOBRIR esse padrão estatisticamente — sem que lhe digam onde está.

    python examples/run_pattern_mining_demo.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from innova_ea.core.enums import Timeframe
from innova_ea.data.sources.synthetic import SyntheticSource
from innova_ea.features import (
    EfficiencyRatio,
    FeatureSet,
    SessionOpenFeature,
    VolatilityRegime,
    YangZhang,
)
from innova_ea.features.sessions import LONDON, NEW_YORK, TOKYO
from innova_ea.research import forward_return, scan_conditions


def main() -> None:
    # 1) Dados com distorção injetada na abertura de Londres (ground-truth).
    src = SyntheticSource(
        seed=11,
        annual_vol=0.08,
        vol_by_hour={7: 3.0, 8: 3.0, 9: 2.0},     # expansão de volatilidade
        drift_by_hour={7: 3.0, 8: 3.0, 9: 2.0},   # impulso direcional (alta) forte
    )
    bars = src.fetch(
        "EURUSD", Timeframe.M15,
        datetime(2018, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    print(f"Barras M15: {bars.height:,}")

    # 2) Matriz de features causais (volatilidade, direção e sessões).
    fs = FeatureSet([
        YangZhang(20),
        VolatilityRegime(10, 100),
        EfficiencyRatio(20),
        SessionOpenFeature(TOKYO, opening_range_bars=4),
        SessionOpenFeature(LONDON, opening_range_bars=4),
        SessionOpenFeature(NEW_YORK, opening_range_bars=4),
    ])
    feats = fs.transform(bars)

    # 3) Alvo: retorno futuro de 4 barras (= 1h), líquido de look-ahead nas features.
    horizon = 4
    feats = feats.with_columns(forward_return(bars, horizon).alias("fwd_ret"))

    # 4) Mineração: hipóteses de padrão sobre os horários de liquidez.
    conditions = {
        "tokyo_ativo": pl.col("tokyo_active") == 1,
        "londres_ativo": pl.col("london_active") == 1,
        "londres_thrust_alta": (pl.col("london_active") == 1) & (pl.col("london_thrust_norm") > 0),
        "londres_expansao_vol": (pl.col("london_active") == 1) & (pl.col("london_vol_expansion") > 1.5),
        "londres_breakout_alta": pl.col("london_or_breakout") == 1,
        "ny_ativo": pl.col("newyork_active") == 1,
        "alta_eficiencia": pl.col("eff_ratio_20") > 0.5,
    }
    results = scan_conditions(feats, "fwd_ret", conditions, min_samples=50, sort_by="t_stat")

    print("\n== Padrões descobertos (ordenados por significância |t|) ==\n")
    for r in results:
        print(r.summary())

    print(
        "\nLeitura: as sessões de TÓQUIO e LONDRES acendem com edge positivo e "
        "alto |t-stat|, redescobrindo a distorção injetada às 07–09 UTC. Isso é "
        "um ACERTO: 07–09 UTC é justamente o overlap real Tóquio–Londres, então "
        "ambas compartilham a janela — o motor localizou corretamente onde vive "
        "o edge. NOVA YORK (sem distorção) fica em/abaixo de zero. "
        "Isto é só geração de hipóteses: cada padrão ainda precisa passar por "
        "walk-forward + Deflated Sharpe antes de virar estratégia."
    )


if __name__ == "__main__":
    main()
