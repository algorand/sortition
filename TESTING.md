# Testing sortition

The test suite combines bit-identical differential testing, independent exact
mathematics, metamorphic properties, certified offline vectors, and mutation
testing. No single oracle is treated as sufficient for the consensus-critical
`SelectF128` path.

## Common commands

```sh
# Complete ordinary suite
go test -count=1 ./...

# Property tests at the depth used by CI
go test -run TestRapid -rapid.checks=20000 -count=1 ./...

# Curated production and test-oracle mutation campaign
go run mutation_check.go

# Longer fuzzing runs
go test -run xxx -fuzz FuzzF128Ops -fuzztime 10m
go test -run xxx -fuzz FuzzSelectF128 -fuzztime 10m
```

## Assurance layers

| Approach | Purpose | Location |
|---|---|---|
| Ordinary and compatibility tests | Basic sortition behavior, benchmarks, agreement with the deployed C++/Boost path, defined frozen-tail behavior, and current-scale regression cases | `sortition_test.go`, `f128_test.go` |
| Bit-identical 128-bit oracle | Mirrors the production CDF walk with 128-bit `math/big.Float`; detects implementation defects in the hand-written integer arithmetic | `selectBigOracle` and related tests in `f128_test.go` |
| Primitive differential fuzzing | Checks f128 arithmetic against `big.Float`, with constructed tie/carry seeds and magnitude-banded rapid generators | `FuzzF128Ops` in `f128_test.go`, `TestRapidF128Ops` in `f128_rapid_test.go` |
| Exact-integer RNE oracle | Computes normalization and round-to-nearest-even solely with `big.Int` quotient/remainder arithmetic, avoiding shared reliance on `big.Float` | `f128_integer_oracle_test.go` |
| Division certificates | Proves sampled `divStep` digits satisfy `U = q*V + R` and `0 <= R < V`, including quotient-estimate corner paths | `TestDivStepExactCertificate` in `f128_integer_oracle_test.go` |
| Exact binomial formula | Builds the CDF from binomial coefficients and integer cross-products rather than the production PMF recurrence | `f128_exact_test.go`, `cdf_exhaustive_test.go` |
| Exhaustive boundary grids | Enumerates small parameter domains and checks digest integers below, at, and above every exact CDF boundary, with anti-vacuity counters | `cdf_exhaustive_test.go`, `oracle_hardening_test.go` |
| Higher-precision convergence | Requires the 512- and 1024-bit CDF walks to converge and compares them with exact small-domain results and certified large-money results | `selectAtPrecision` in `f128_exact_test.go`, tests in `oracle_hardening_test.go` |
| Internal trajectory and liveness | Checks PMF/CDF error envelopes and monotonicity, audits the maximum-domain exponent path, proves bounded freeze permanence, and pins CDF-evaluation limits | `f128_trajectory_test.go` |
| Metamorphic properties | Checks digest monotonicity, power-of-two and arbitrary common-factor probability scaling, primitive identities, and arithmetic order properties without a numeric oracle | `f128_rapid_test.go` |
| Distribution sanity | Checks aggregate selection weight against the expected binomial mean without reusing the CDF formula | `TestSelectF128Distribution` in `f128_exact_test.go` |
| Arb-certified quantiles | Uses Arb's regularized incomplete beta implementation to certify large-money quantile inequalities with rigorous dyadic endpoints | `tools/generate_arb_oracle.py`, `testdata/f128_arb_certificates.json`, `f128_arb_certificate_test.go` |
| Mutation testing | Applies curated bugs to production code and in-process test oracles; every non-equivalent mutant must be killed | `mutation_check.go` |

## Arb corpus

The Arb generator is an offline tool and is not a dependency of ordinary
`go test`. Set it up in an ignored virtual environment:

```sh
python3 -m venv .venv-oracle
.venv-oracle/bin/pip install -r tools/requirements-oracle.txt
```

Regenerate the corpus or verify it without rewriting the checked-in file:

```sh
.venv-oracle/bin/python tools/generate_arb_oracle.py \
  > testdata/f128_arb_certificates.json

.venv-oracle/bin/python tools/generate_arb_oracle.py \
  --check testdata/f128_arb_certificates.json
```

The Go certificate test exactly compares the stored dyadic endpoints with the
digest ratio. Trust in the external tool is limited to Arb's assertion that
those endpoints enclose the true incomplete-beta CDF.

## Important test semantics

- The near-one frozen-tail sliver is defined to return `money`. Exact-math
  tolerance tests exclude that interval deliberately; dedicated tests pin its
  behavior and liveness at current-scale values.
- `money` must remain below `SelectF128MaxMoney`. The maximum-domain exponent
  audit checks the worst representable `1-p` combination without performing a
  supply-sized walk.
- Large-money in-process oracles have a committee-scale step budget. A broken
  reference must fail loudly instead of hanging while attempting trillions of
  iterations.
- Randomized tests use deterministic seeds. Measure-zero events such as exact
  ties and inclusive boundaries are constructed explicitly rather than left
  to sampling.
- Tests with skip or clamp paths need anti-vacuity counts or mandatory spot
  cases proving that meaningful assertions ran.

## Continuous integration

- `.github/workflows/test.yml` runs builds, ordinary tests, and 20,000-check
  rapid properties across Linux amd64/arm64, macOS amd64/arm64, and Windows
  amd64. It also runs the mutation campaign on Linux.
- `.github/workflows/oracle.yml` regenerates and compares the complete Arb
  corpus nightly and on manual dispatch.

When arithmetic or recurrence semantics change, update the relevant
independent oracle and exact/certified vectors deliberately, then rerun the
full suite and mutation campaign. A changed knife-edge pin is evidence that
the numerical trajectory changed and should be explained in the commit.
