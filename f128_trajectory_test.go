// Copyright (C) 2019-2023 Algorand, Inc.
// This file is part of go-algorand
//
// go-algorand is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// go-algorand is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with go-algorand.  If not, see <https://www.gnu.org/licenses/>.

package sortition

import (
	"math/big"
	"testing"
)

// auditedMul performs the same multiplication used by intPow while proving
// with unbounded integers that both possible normalization exponents fit in
// int64. Checking both +127 and +128 is conservative and avoids duplicating
// mul's partial-product branch decision in the test.
func auditedMul(t *testing.T, a, b f128, minExp, maxExp *int64) f128 {
	t.Helper()
	if !a.isZero() && !b.isZero() {
		for _, normalization := range []int64{127, 128} {
			e := new(big.Int).SetInt64(a.exp)
			e.Add(e, new(big.Int).SetInt64(b.exp))
			e.Add(e, new(big.Int).SetInt64(normalization))
			if !e.IsInt64() {
				t.Fatalf("mul exponent overflows int64: %d + %d + %d = %s", a.exp, b.exp, normalization, e)
			}
		}
	}
	out := a.mul(b)
	assertCanonicalF128(t, "audited mul", out)
	if !out.isZero() {
		if out.exp < *minExp {
			*minExp = out.exp
		}
		if out.exp > *maxExp {
			*maxExp = out.exp
		}
	}
	return out
}

func auditedIntPow(t *testing.T, base f128, e uint64) (f128, int64, int64) {
	t.Helper()
	minExp, maxExp := base.exp, base.exp
	result, b := f128FromUint64(1), base
	for e > 0 {
		if e&1 == 1 {
			result = auditedMul(t, result, b, &minExp, &maxExp)
		}
		e >>= 1
		if e > 0 {
			b = auditedMul(t, b, b, &minExp, &maxExp)
		}
	}
	return result, minExp, maxExp
}

// TestSelectF128MaxDomainExponentAudit exercises the documented worst case:
// money is one below the exported limit and 1-p is the smallest positive
// rational representable by the uint64 API. The test audits every exponent
// addition in exponentiation-by-squaring with big.Int, so a future bound or
// normalization change fails before platform-independent int64 arithmetic can
// wrap.
func TestSelectF128MaxDomainExponentAudit(t *testing.T) {
	const total = ^uint64(0)
	money := SelectF128MaxMoney - 1
	expected := total - 1 // 1-p = 1/(2^64-1)
	q := f128FromUint64(1).div(f128FromUint64(total))
	pmf, minExp, maxExp := auditedIntPow(t, q, money)
	dist := newBinomialF128(expected, total, money)
	if dist == nil || pmf != dist.pmf {
		t.Fatalf("audited pmf(0)=%+v, constructor produced %+v", pmf, dist)
	}

	// q is just above 2^-64. Across money < 2^56 its logarithmic correction
	// is less than one, so floor(log2(q^money)) is exactly -64*money.
	wantExp := -int64(money<<6) - 127
	if pmf.exp != wantExp {
		t.Fatalf("domain-edge pmf exponent=%d, want %d", pmf.exp, wantExp)
	}
	if minExp < wantExp || maxExp > 0 {
		t.Fatalf("unexpected audited exponent range [%d,%d], final lower bound %d", minExp, maxExp, wantExp)
	}
	if got := SelectF128(money, total, expected, Digest{}); got != 0 {
		t.Fatalf("domain-edge ratio zero selected %d, want 0", got)
	}
}

// TestSelectF128DegenerateDistributionShortCircuit pins construction, not
// only the eventual return value. If p==1 accidentally enters the ordinary
// walk, its all-zero PMF produces the right answer for nonzero digests only
// after money iterations, recreating the liveness class that freeze prevents.
func TestSelectF128DegenerateDistributionShortCircuit(t *testing.T) {
	const money = SelectF128MaxMoney - 1
	for _, total := range []uint64{0, 1, 6000, ^uint64(0)} {
		if dist := newBinomialF128(total, total, money); dist != nil {
			t.Fatalf("p==1 constructed an ordinary distribution (total=%d): %+v", total, dist)
		}
		if total != ^uint64(0) {
			if dist := newBinomialF128(total+1, total, money); dist != nil {
				t.Fatalf("p>1 constructed an ordinary distribution (total=%d): %+v", total, dist)
			}
		}
	}
}

type highPrecisionBinomial struct {
	money uint64
	prec  uint
	at    uint64
	pq    *big.Float
	pmf   *big.Float
	cum   *big.Float
}

func newHighPrecisionBinomial(money, total, expected uint64, prec uint) *highPrecisionBinomial {
	q := new(big.Float).SetPrec(prec).Quo(
		new(big.Float).SetPrec(prec).SetUint64(total-expected),
		new(big.Float).SetPrec(prec).SetUint64(total))
	pq := new(big.Float).SetPrec(prec).Quo(
		new(big.Float).SetPrec(prec).SetUint64(expected),
		new(big.Float).SetPrec(prec).SetUint64(total-expected))
	pmf := bigIntPow(q, money, prec)
	return &highPrecisionBinomial{
		money: money,
		prec:  prec,
		pq:    pq,
		pmf:   pmf,
		cum:   new(big.Float).SetPrec(prec).Set(pmf),
	}
}

func (b *highPrecisionBinomial) advance() {
	b.at++
	factor := new(big.Float).SetPrec(b.prec).Quo(
		new(big.Float).SetPrec(b.prec).SetUint64(b.money-b.at+1),
		new(big.Float).SetPrec(b.prec).SetUint64(b.at))
	step := new(big.Float).SetPrec(b.prec).Mul(factor, b.pq)
	b.pmf = new(big.Float).SetPrec(b.prec).Mul(b.pmf, step)
	b.cum = new(big.Float).SetPrec(b.prec).Add(b.cum, b.pmf)
}

func binomialMode(money, total, expected uint64) uint64 {
	n := new(big.Int).SetUint64(money)
	n.Add(n, big.NewInt(1))
	n.Mul(n, new(big.Int).SetUint64(expected))
	n.Quo(n, new(big.Int).SetUint64(total))
	return n.Uint64()
}

func f128AsBig(x f128, prec uint) *big.Float {
	return new(big.Float).SetPrec(prec).Set(f128ToBig(x))
}

func absFloatDifference(a, b *big.Float, prec uint) *big.Float {
	d := new(big.Float).SetPrec(prec).Sub(a, b)
	return d.Abs(d)
}

func trajectoryErrorBound(money, at uint64, prec uint) *big.Float {
	// RNE unit roundoff is 2^-128. Sixty-four ulps per initialization trial
	// and recurrence index is deliberately conservative: it covers q^money's
	// amplified input rounding, four rounded recurrence operations per term,
	// and CDF summation without turning this into an empirical pinned maximum.
	units := new(big.Int).SetUint64(money)
	units.Add(units, new(big.Int).SetUint64(32*(at+1)+128))
	return new(big.Float).SetMantExp(new(big.Float).SetPrec(prec).SetInt(units), -122)
}

func assertTrajectoryError(t *testing.T, name string, at uint64, got f128, reference, bound *big.Float, relative bool) {
	t.Helper()
	const prec = uint(512)
	diff := absFloatDifference(f128AsBig(got, prec), reference, prec)
	if relative {
		if reference.Sign() == 0 {
			if diff.Sign() != 0 {
				t.Fatalf("%s(%d): nonzero value against zero reference", name, at)
			}
			return
		}
		diff.Quo(diff, reference)
	}
	if diff.Cmp(bound) > 0 {
		t.Fatalf("%s(%d) error %s exceeds bound %s", name, at, diff.Text('p', 8), bound.Text('p', 8))
	}
}

// TestSelectF128TrajectoryErrorBudget compares every production PMF/CDF state
// through and beyond the mode with a 512-bit recurrence. Exact/Arb tests cover
// shared-formula risk; this test instead detects substantial internal drift
// that happens not to move the sampled final quantile.
func TestSelectF128TrajectoryErrorBudget(t *testing.T) {
	tests := []struct {
		name                   string
		money, total, expected uint64
		maxAt                  uint64
	}{
		{"toy balanced", 256, 512, 256, 255},
		{"online 1500", 2_000_000_000_000_000, 2_000_000_000_000_000, 1500, 2500},
		{"one third supply", 10_000_000_000_000_000 / 3, 10_000_000_000_000_000, 2990, 1800},
		{"payout minimum", 30_000_000_000, 10_000_000_000_000_000, 5000, 120},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const prec = uint(512)
			got := newBinomialF128(test.expected, test.total, test.money)
			want := newHighPrecisionBinomial(test.money, test.total, test.expected, prec)
			mode := binomialMode(test.money, test.total, test.expected)
			observed := uint64(0)
			for at := uint64(0); at <= test.maxAt; at++ {
				if at > 0 {
					pmfPrev, cumPrev := got.pmf, got.cum
					got.cdf(at)
					want.advance()
					if got.cum.cmp(cumPrev) < 0 {
						t.Fatalf("CDF decreased at %d", at)
					}
					if at <= mode && got.pmf.cmp(pmfPrev) < 0 {
						t.Fatalf("PMF decreased before mode %d at %d", mode, at)
					}
					if at > mode && got.pmf.cmp(pmfPrev) > 0 {
						t.Fatalf("PMF increased after mode %d at %d", mode, at)
					}
				}
				bound := trajectoryErrorBound(test.money, at, prec)
				assertTrajectoryError(t, "pmf", at, got.pmf, want.pmf, bound, true)
				assertTrajectoryError(t, "cdf", at, got.cum, want.cum, bound, false)
				observed++
				if got.frozen {
					break
				}
			}
			if observed <= mode {
				t.Fatalf("trajectory stopped after %d states before crossing mode %d", observed, mode)
			}
		})
	}
}

// TestSelectF128FreezePermanence ignores the production short-circuit after a
// bounded walk freezes and explicitly advances the recurrence to money-1.
// Every later PMF must remain non-increasing and every rounded CDF add must be
// a no-op, proving on the exercised grid that the optimization returns the
// same answer as the otherwise impractical unshortened walk.
func TestSelectF128FreezePermanence(t *testing.T) {
	tests := []struct {
		money, total, expected uint64
	}{
		{256, 512, 256},
		{512, 1000, 1},
		{1000, 10_000, 20},
		{3000, 2_000_000_000_000_000, 1500},
	}
	frozenCases, checkedTailSteps := 0, uint64(0)
	for _, test := range tests {
		b := newBinomialF128(test.expected, test.total, test.money)
		for j := uint64(1); j < test.money && !b.frozen; j++ {
			b.cdf(j)
		}
		if !b.frozen {
			continue
		}
		frozenCases++
		pmf, cum, at := b.pmf, b.cum, b.at
		for at+1 < test.money {
			at++
			pmfNext := f128FromUint64(test.money - at + 1).divU(at).mul(b.pq).mul(pmf)
			if pmfNext.cmp(pmf) > 0 {
				t.Fatalf("PMF increased after freeze at %d (money=%d total=%d expected=%d)", at, test.money, test.total, test.expected)
			}
			if cumNext := cum.add(pmfNext); cumNext != cum {
				t.Fatalf("CDF changed after freeze at %d (money=%d total=%d expected=%d)", at, test.money, test.total, test.expected)
			}
			pmf = pmfNext
			checkedTailSteps++
		}
	}
	if frozenCases < 3 || checkedTailSteps < 100 {
		t.Fatalf("anti-vacuity: frozenCases=%d checkedTailSteps=%d", frozenCases, checkedTailSteps)
	}
}

func selectF128WithStepCount(money, total, expected uint64, d Digest) (selected, evaluations uint64, froze bool) {
	ratio := f128FromDigestRatio(d)
	dist := newBinomialF128(expected, total, money)
	if dist == nil {
		if ratio.isZero() {
			return 0, 0, false
		}
		return money, 0, false
	}
	for j := uint64(0); j < money; j++ {
		evaluations++
		boundary := dist.cdf(j)
		if ratio.cmp(boundary) <= 0 {
			return j, evaluations, false
		}
		if dist.frozen {
			return money, evaluations, true
		}
	}
	return money, evaluations, false
}

// TestSelectF128ConsensusStepBounds makes liveness deterministic by counting
// CDF evaluations rather than timing them. The cases cover certified tails
// outside the frozen sliver and defined money-returning cases inside it.
func TestSelectF128ConsensusStepBounds(t *testing.T) {
	const (
		online = uint64(2_000_000_000_000_000)
		supply = uint64(10_000_000_000_000_000)
	)
	tests := []struct {
		name                   string
		money, total, expected uint64
		digest                 Digest
		want                   uint64
		wantFreeze             bool
	}{
		{"online 1500 certified tail", online, online, 1500, maxDigestMinusPowerOfTwo(196), 1852, false},
		{"online 6000 certified tail", online, online, 6000, maxDigestMinusPowerOfTwo(200), 6667, false},
		{"supply 5000 certified tail", supply, supply, 5000, maxDigestMinusPowerOfTwo(190), 5667, false},
		{"proposer frozen tail", online - 1, online - 1, 20, maxDigestMinusPowerOfTwo(175), online - 1, true},
		{"base minimum frozen tail", 100_000, supply, 5000, maxDigestMinusPowerOfTwo(141), 100_000, true},
		{"supply frozen tail", supply, supply, 5000, maxDigestMinusPowerOfTwo(178), supply, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, evaluations, froze := selectF128WithStepCount(test.money, test.total, test.expected, test.digest)
			if got != test.want || got != SelectF128(test.money, test.total, test.expected, test.digest) {
				t.Fatalf("instrumented selection=%d, want %d", got, test.want)
			}
			if froze != test.wantFreeze {
				t.Fatalf("freeze=%v, want %v after %d evaluations", froze, test.wantFreeze, evaluations)
			}
			limit := test.expected + 4096
			if evaluations == 0 || evaluations > limit {
				t.Fatalf("walk used %d CDF evaluations, limit %d", evaluations, limit)
			}
		})
	}
}
