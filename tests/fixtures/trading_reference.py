"""Reference ("ground truth") implementation of the trading recalibration model.

This is a tidied, self-contained port of the original external simulator that the
trading sell/buy strategies were derived from. It exists so the strategy
verification tests have a fixed, in-repo ground truth to compare against, with no
dependency on any external module.

The model treats a matrix of organism/gene *performances* (values in ``[0, 1]``,
``NaN`` meaning "not applicable") as a market of exercises (genes) that probands
(organisms) trade in. Each recalibration round runs a **sell** phase followed by a
**buy** phase and returns the updated per-gene values.

Three sell strategies are supported, each the ground truth for one pikaia pair:

============  =======================================  ==========================
strategy      pikaia pair                              per-gene sell signal
============  =======================================  ==========================
``"hard"``    ``SellHard`` + ``BuyHard``               weighted by difficulty
``"uniform"`` ``SellUniform`` + ``BuyUniform``         uniform across genes
``"easy"``    ``SellEasy`` + ``BuyEasy``               weighted by ease (inverse)
============  =======================================  ==========================
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Strategy name -> mixFactor. mixFactor drives both the buy signal and the
# per-proband exclusiveness normalisation (see below).
_MIX_FACTOR = {"hard": 1.0, "uniform": 0.0, "easy": 1.0}

# Values whose absolute magnitude is within this of 0 or 1 are treated as trivial
# (a gene solved by everyone or by no-one) and contribute no sell/buy signal.
_EPS = 1e-6


def trading_recalibrate(
    performance: npt.ArrayLike,
    start_values: npt.ArrayLike,
    sell_strategy: str,
) -> np.ndarray:
    """Run one recalibration round and return the updated per-gene values.

    Args:
        performance: ``(N, M)`` array of organism/gene performances in ``[0, 1]``.
            ``NaN`` entries are skipped (that organism does not trade that gene).
        start_values: ``(M,)`` per-gene values at the start of the round.
        sell_strategy: one of ``"hard"``, ``"uniform"`` or ``"easy"``.

    Returns:
        ``(M,)`` array of per-gene values after the sell + buy phases.
    """
    if sell_strategy not in _MIX_FACTOR:
        raise ValueError(
            f"unknown sell_strategy {sell_strategy!r}; "
            f"expected one of {sorted(_MIX_FACTOR)}"
        )

    P = np.asarray(performance, dtype=float)
    N, M = P.shape
    values = np.array(start_values, dtype=float)
    start = values.copy()  # per-gene start value; the sell signal scales with it
    mix = _MIX_FACTOR[sell_strategy]

    valid = ~np.isnan(P)  # (N, M) — which organism/gene pairs participate
    n_valid = valid.sum(axis=0)  # per-gene count of participating organisms

    # Exclusiveness of a gene = fraction of participating organisms that did NOT
    # solve it (mean of 1 - performance). High => rare/hard; low => common/easy.
    one_minus_P = np.where(valid, 1.0 - P, 0.0)
    excl = np.divide(
        one_minus_P.sum(axis=0),
        n_valid,
        out=np.zeros(M),
        where=n_valid > 0,
    )

    # --- Per-gene sell/buy signals (computed once, up front) -----------------
    vdelta_buy = np.zeros(M)
    vdelta_sell = np.zeros(M)
    for j in range(M):
        e = excl[j]
        trivial = abs(e) <= _EPS or 1.0 - abs(e) <= _EPS
        if 1.0 - abs(e) > _EPS:
            # Blend of "buy hard" (excl) and "buy easy" (1 - excl) by mixFactor.
            vdelta_buy[j] = (1.0 - mix) * e + mix * (1.0 - e)
        if not trivial:
            if sell_strategy == "hard":
                vdelta_sell[j] = start[j] * (e / (1.0 - e)) / n_valid[j]
            elif sell_strategy == "uniform":
                vdelta_sell[j] = start[j] / n_valid[j]
            else:  # "easy" — the inverse of "hard"
                vdelta_sell[j] = -start[j] * (e / (1.0 - e)) / n_valid[j]

    # --- Per-organism exclusiveness norms ------------------------------------
    # norm  gates whether an organism buys at all (0 => perfect, no buying).
    # norm2 is the capital-distribution denominator, blended by mixFactor.
    excl_norm = (one_minus_P * (1.0 - excl)[None, :]).sum(axis=1)
    excl_norm2 = (
        (1.0 - mix) * one_minus_P * excl[None, :]
        + mix * one_minus_P * (1.0 - excl)[None, :]
    ).sum(axis=1)

    # An organism is "perfect" if it solved every gene it participates in; such
    # organisms have nothing to sell.
    perfect = np.array([bool(np.all(P[i, valid[i]] >= 1.0)) for i in range(N)])

    # --- Sequential sell-then-buy per organism -------------------------------
    # Each organism's contribution is independent (max_capital depends only on
    # the fixed sell signal), so the accumulation order does not affect results.
    for i in range(N):
        delta_sell = 0.0 if perfect[i] else 1.0

        # Sell: drain value from each solved gene, banking it as capital.
        max_capital = 0.0
        for j in range(M):
            if valid[i, j]:
                sold = P[i, j] * delta_sell * vdelta_sell[j]
                values[j] -= sold
                max_capital += sold
        capital = max_capital

        # Buy: redistribute capital to failed genes, weighted by the buy signal.
        delta_buy = 1.0 / excl_norm2[i] if excl_norm[i] != 0.0 else 0.0
        leftover_genes = []
        for j in range(M):
            if valid[i, j]:
                ratio = 1.0 - P[i, j]
                bought = ratio * max_capital * delta_buy * vdelta_buy[j]
                values[j] += bought
                capital -= bought
                if vdelta_buy[j] == 0.0:
                    leftover_genes.append(j)

        # Any capital that could not be spent (buy signal was 0 everywhere it
        # was needed) is dumped evenly across the zero-signal genes.
        if abs(capital) > 1e-5 and leftover_genes:
            share = capital / len(leftover_genes)
            for j in leftover_genes:
                values[j] += share

    return values


def trading_recalibrate_k(
    performance: npt.ArrayLike,
    start_values: npt.ArrayLike,
    sell_strategy: str,
    k: int,
) -> np.ndarray:
    """Run ``k`` recalibration rounds, feeding each round's output into the next."""
    values = np.array(start_values, dtype=float)
    for _ in range(k):
        values = trading_recalibrate(performance, values, sell_strategy)
    return values
