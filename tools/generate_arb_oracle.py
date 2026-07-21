#!/usr/bin/env python3
"""Generate rigorous large-money binomial-CDF certificates with Arb.

This is deliberately an offline oracle, not a dependency of `go test`.
python-flint exposes Arb ball arithmetic; its regularized incomplete beta
implementation is structurally independent of SelectF128's q^money-seeded
forward PMF recurrence. Each emitted vector carries dyadic Arb endpoints that
prove, using exact integer comparison,

    CDF(j-1) < digest/(2^256-1) <= CDF(j).

Setup and regenerate from the repository root:

    python3 -m venv .venv-oracle
    .venv-oracle/bin/pip install -r tools/requirements-oracle.txt
    .venv-oracle/bin/python tools/generate_arb_oracle.py \
        > testdata/f128_arb_certificates.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass

import flint
from flint import arb, ctx, fmpq


DIGEST_DEN = (1 << 256) - 1
MAX_MONEY = 1 << 56
PRECISIONS = (128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class Case:
    label: str
    money: int
    total: int
    expected: int
    digest: int


def max_minus_power(bit: int) -> int:
    return DIGEST_DEN - (1 << bit)


CASES = (
    Case("toy symmetric median", 100, 200, 100, 1 << 255),
    Case("toy low probability", 250, 1000, 1, DIGEST_DEN // 3),
    Case("realistic half stake median", 1_000_000_000_000_000, 2_000_000_000_000_000, 1500, 1 << 255),
    Case("base minimum ordinary ratio", 100_000, 10_000_000_000_000_000, 5000, 1 << 255),
    Case("payout minimum upper tail", 30_000_000_000, 10_000_000_000_000_000, 5000, max_minus_power(216)),
    Case("online stake 1500 tail", 2_000_000_000_000_000, 2_000_000_000_000_000, 1500, max_minus_power(196)),
    Case("online stake 6000 tail", 2_000_000_000_000_000, 2_000_000_000_000_000, 6000, max_minus_power(200)),
    Case("supply ceiling 5000 tail", 10_000_000_000_000_000, 10_000_000_000_000_000, 5000, max_minus_power(190)),
    Case("one third supply 2990 tail", 10_000_000_000_000_000 // 3, 10_000_000_000_000_000, 2990, max_minus_power(198)),
    Case("domain edge tiny probability", MAX_MONEY - 1, DIGEST_DEN >> 192, 1, max_minus_power(206)),
    Case("probability near one", 3000, 6001, 6000, 1 << 255),
    Case("large balanced probability", 3000, DIGEST_DEN >> 192, (DIGEST_DEN >> 192) // 2, 1 << 255),
)


def dyadic(x: arb) -> tuple[int, int]:
    """Return the exact dyadic point x as (integer mantissa, exponent)."""
    mantissa, exponent = x.man_exp()
    return int(mantissa), int(exponent)


def compare_rat_dyadic(num: int, den: int, point: tuple[int, int]) -> int:
    """Compare num/den with mantissa*2^exponent using integer arithmetic."""
    mantissa, exponent = point
    if exponent >= 0:
        left, right = num, (mantissa << exponent) * den
    else:
        left, right = num << (-exponent), mantissa * den
    return (left > right) - (left < right)


def cdf_ball(case: Case, j: int, precision: int) -> arb:
    ctx.prec = precision
    if j < 0:
        return arb(0)
    if j >= case.money:
        return arb(1)
    q = arb(fmpq(case.total - case.expected, case.total))
    return q.beta_lower(case.money - j, j + 1, regularized=True)


def classify(case: Case, j: int) -> bool:
    """Return whether exact ratio <= CDF(j), raising precision as needed."""
    for precision in PRECISIONS:
        cdf = cdf_ball(case, j, precision)
        lower, upper = dyadic(cdf.lower()), dyadic(cdf.upper())
        if compare_rat_dyadic(case.digest, DIGEST_DEN, lower) <= 0:
            return True
        if compare_rat_dyadic(case.digest, DIGEST_DEN, upper) > 0:
            return False
    raise RuntimeError(f"could not separate ratio from CDF({j}) for {case.label}")


def select(case: Case) -> int:
    if not 0 < case.expected < case.total:
        raise ValueError(f"certificate case must have 0 < expected < total: {case.label}")
    if not 0 < case.digest <= DIGEST_DEN:
        raise ValueError(f"certificate case must have 0 < digest <= max: {case.label}")
    # Do not binary-search the entire [0,n] interval immediately. For the
    # protocol-shaped tiny probabilities, asking incomplete beta for j near
    # n/2 is vastly harder than evaluating it around the committee-scale
    # quantile. Bracket exponentially from the nearer tail first.
    if 2 * case.expected <= case.total:
        if classify(case, 0):
            return 0
        lo, hi = 0, 1
        while hi < case.money and not classify(case, hi):
            lo, hi = hi, min(case.money, 2 * hi)
    else:
        if not classify(case, case.money - 1):
            return case.money
        hi, delta = case.money - 1, 2
        lo = max(-1, case.money - delta)
        while lo >= 0 and classify(case, lo):
            hi, delta = lo, 2 * delta
            lo = max(-1, case.money - delta)

    while hi - lo > 1:
        mid = lo + (hi - lo) // 2
        if classify(case, mid):
            hi = mid
        else:
            lo = mid
    return hi


def certify(case: Case, selected: int) -> dict:
    for precision in PRECISIONS:
        previous = cdf_ball(case, selected - 1, precision).upper()
        current = cdf_ball(case, selected, precision).lower()
        previous_dyadic, current_dyadic = dyadic(previous), dyadic(current)
        if (
            compare_rat_dyadic(case.digest, DIGEST_DEN, previous_dyadic) > 0
            and compare_rat_dyadic(case.digest, DIGEST_DEN, current_dyadic) <= 0
        ):
            return {
                "label": case.label,
                "money": case.money,
                "total_money": case.total,
                "expected_size": case.expected,
                "digest": f"{case.digest:064x}",
                "want": selected,
                "precision_bits": precision,
                "cdf_previous_upper": {
                    "mantissa": str(previous_dyadic[0]),
                    "exponent": previous_dyadic[1],
                },
                "cdf_selected_lower": {
                    "mantissa": str(current_dyadic[0]),
                    "exponent": current_dyadic[1],
                },
            }
    raise RuntimeError(f"could not certify selected boundary for {case.label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="generate only the case whose label contains this text")
    args = parser.parse_args()
    cases = tuple(case for case in CASES if args.case is None or args.case in case.label)
    if not cases:
        parser.error("--case did not match any vector")

    vectors = []
    for case in cases:
        started = time.monotonic()
        print(f"certifying {case.label}...", file=sys.stderr, flush=True)
        vectors.append(certify(case, select(case)))
        print(f"certified {case.label} in {time.monotonic() - started:.3f}s", file=sys.stderr, flush=True)
    output = {
        "format": 1,
        "generator": f"python-flint {flint.__version__} / Arb regularized incomplete beta",
        "semantics": "exact binomial CDF; frozen-tail definition is tested separately",
        "vectors": vectors,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
