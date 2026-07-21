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
	"math/rand"
	"testing"
)

// The differential tests compare SelectF128 against an oracle that computes
// at the SAME 128-bit precision, so a defect mirrored into the oracle (as the
// pmf(0) plateau was) is invisible to them. The tests in this file compare
// against a 512-bit reference instead: the 128-bit walk must land within one
// boundary of the near-exact answer everywhere outside the documented
// frozen-tail sliver, and must agree with the oracle bit-for-bit at
// deliberately constructed knife-edge digests.

// selectHighPrec runs the binomial-CDF walk at 512-bit precision with the
// near-exact digest ratio. It is a reference for tolerance checking, not a
// bit-identical mirror of SelectF128.
func selectHighPrec(money, totalMoney, expectedSize uint64, vrfOutput Digest) uint64 {
	const prec = 512
	ratio := digestRatioBig(vrfOutput, prec)
	if expectedSize >= totalMoney { // p >= 1
		if ratio.Sign() <= 0 {
			return 0
		}
		return money
	}
	q := new(big.Float).SetPrec(prec).Quo(
		new(big.Float).SetPrec(prec).SetUint64(totalMoney-expectedSize),
		new(big.Float).SetPrec(prec).SetUint64(totalMoney))
	pq := new(big.Float).SetPrec(prec).Quo(
		new(big.Float).SetPrec(prec).SetUint64(expectedSize),
		new(big.Float).SetPrec(prec).SetUint64(totalMoney-expectedSize))
	pmf := bigIntPow(q, money, prec)
	cdf := new(big.Float).SetPrec(prec).Set(pmf)
	if cdf.Cmp(ratio) >= 0 {
		return 0
	}
	for j := uint64(1); j < money; j++ {
		factor := new(big.Float).SetPrec(prec).Quo(
			new(big.Float).SetPrec(prec).SetUint64(money-j+1),
			new(big.Float).SetPrec(prec).SetUint64(j))
		pmf = new(big.Float).SetPrec(prec).Mul(pmf, new(big.Float).SetPrec(prec).Mul(factor, pq))
		cdf = new(big.Float).SetPrec(prec).Add(cdf, pmf)
		if cdf.Cmp(ratio) >= 0 {
			return j
		}
	}
	return money
}

// inFrozenSliver reports whether the digest ratio is within the carved-out
// near-1.0 region where the 128-bit walk's answer is DEFINED as money (see
// the SelectF128 doc comment) and tolerance against exact math does not
// apply. The bound (money+2)*2^-122 covers the plateau's ~money*2^-129 with
// two orders of margin, including the boundary-crowding zone just above it.
func inFrozenSliver(money uint64, ratio *big.Float) bool {
	gap := new(big.Float).SetPrec(512).Sub(new(big.Float).SetPrec(512).SetInt64(1), ratio)
	bound := new(big.Float).SetMantExp(new(big.Float).SetPrec(64).SetUint64(money+2), -122)
	return gap.Cmp(bound) < 0
}

// TestSelectF128NearExactMath asserts |SelectF128 - selectHighPrec| <= 1 for
// every input outside the frozen sliver. A mirrored magnitude error like the
// pmf(0) plateau cannot hide from this: with the pre-freeze-fix code the
// supply-scale spots below would have diverged by ~2e15 (or hung without the
// short-circuit). Nothing else on this branch compares supply-scale results
// against any reference outside the sliver.
func TestSelectF128NearExactMath(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	// mustAssert guards against the comparison being skipped by the sliver
	// carve-out: the deliberately chosen spot cases must actually assert (an
	// earlier version of the carve-out bound was accidentally so wide that
	// the supply-scale spots passed vacuously).
	check := func(money, total, expected uint64, d Digest, mustAssert bool) {
		t.Helper()
		if inFrozenSliver(money, digestRatioBig(d, 512)) {
			if mustAssert {
				t.Fatalf("spot case unexpectedly in the frozen sliver (money=%d vrf=%x)", money, d)
			}
			return
		}
		j128 := SelectF128(money, total, expected, d)
		jhp := selectHighPrec(money, total, expected, d)
		diff := j128 - jhp
		if jhp > j128 {
			diff = jhp - j128
		}
		if diff > 1 {
			t.Fatalf("SelectF128=%d vs 512-bit reference=%d (money=%d total=%d expected=%d vrf=%x)",
				j128, jhp, money, total, expected, d)
		}
	}

	for i := 0; i < 400; i++ {
		money := rng.Uint64() % 2000
		total := rng.Uint64() >> uint(rng.Intn(40))
		if total == 0 {
			total = 1
		}
		expected := rng.Uint64() >> uint(rng.Intn(50))
		var d Digest
		if i%3 == 0 {
			rng.Read(d[:])
		} else {
			// near-maximum digests stress the tail, where boundaries crowd
			for j := range d {
				d[j] = 0xff
			}
			d[rng.Intn(14)] = byte(rng.Uint64())
		}
		check(money, total, expected, d, false)
	}

	// supply-scale spots, chosen outside the sliver (gaps 2^-56..2^-66 vs a
	// sliver bound near 2^-68 at the supply ceiling)
	const online = uint64(2_000_000_000_000_000)
	const supply = uint64(10_000_000_000_000_000)
	check(online, online, 1500, maxDigestMinusPowerOfTwo(196), true)   // ratio ~1-2^-60
	check(online, online, 6000, maxDigestMinusPowerOfTwo(200), true)   // ratio ~1-2^-56
	check(supply, supply, 5000, maxDigestMinusPowerOfTwo(190), true)   // ratio ~1-2^-66
	check(supply/3, supply, 2990, maxDigestMinusPowerOfTwo(198), true) // ratio ~1-2^-58
}

// exactCDFNum returns the numerator of the EXACT binomial CDF at j over the
// denominator totalMoney^money: sum_{i<=j} C(money,i) * E^i * (T-E)^(money-i),
// built from first principles with stdlib binomial coefficients. It shares no
// formula with the walk or the differential oracle, which both use the PMF
// recurrence pmf(j) = pmf(j-1)*(money-j+1)/j * pq -- a shared algebra error
// there would pass every differential test but not this.
func exactCDFNum(money, totalMoney, expectedSize, j uint64) *big.Int {
	e := new(big.Int).SetUint64(expectedSize)
	q := new(big.Int).SetUint64(totalMoney - expectedSize)
	sum := new(big.Int)
	for i := uint64(0); i <= j; i++ {
		term := new(big.Int).Binomial(int64(money), int64(i))
		term.Mul(term, new(big.Int).Exp(e, new(big.Int).SetUint64(i), nil))
		term.Mul(term, new(big.Int).Exp(q, new(big.Int).SetUint64(money-i), nil))
		sum.Add(sum, term)
	}
	return sum
}

// selectExactRat runs the walk against the exact CDF with pure integer
// arithmetic: ratio <= cdf(j) becomes t * T^money <= cdfNum(j) * (2^256-1).
// No rounding anywhere. Cost grows as T^money-sized integers, so callers keep
// money small.
func selectExactRat(money, totalMoney, expectedSize uint64, vrfOutput Digest) uint64 {
	tDig := new(big.Int).SetBytes(vrfOutput[:])
	if expectedSize >= totalMoney { // p >= 1
		if tDig.Sign() == 0 {
			return 0
		}
		return money
	}
	den256 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	lhs := new(big.Int).Mul(tDig,
		new(big.Int).Exp(new(big.Int).SetUint64(totalMoney), new(big.Int).SetUint64(money), nil))
	for j := uint64(0); j < money; j++ {
		rhs := new(big.Int).Mul(exactCDFNum(money, totalMoney, expectedSize, j), den256)
		if lhs.Cmp(rhs) <= 0 {
			return j
		}
	}
	return money
}

// TestSelectF128VsExactRat compares the 128-bit walk against exact
// mathematics for small money, including digests constructed to sit exactly
// on true CDF boundaries. Agreement must be within one boundary outside the
// frozen sliver; a formula error shared by the implementation and the
// differential oracle cannot hide here.
func TestSelectF128VsExactRat(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	den256 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	check := func(money, total, expected uint64, d Digest) {
		t.Helper()
		if inFrozenSliver(money, digestRatioBig(d, 512)) {
			return
		}
		got := SelectF128(money, total, expected, d)
		want := selectExactRat(money, total, expected, d)
		diff := got - want
		if want > got {
			diff = want - got
		}
		if diff > 1 {
			t.Fatalf("SelectF128=%d vs exact=%d (money=%d total=%d expected=%d vrf=%x)",
				got, want, money, total, expected, d)
		}
	}

	for i := 0; i < 300; i++ {
		money := rng.Uint64() % 51
		total := 1 + rng.Uint64()>>uint(rng.Intn(40))
		expected := rng.Uint64() >> uint(rng.Intn(50))
		var d Digest
		switch {
		case i%3 == 0:
			rng.Read(d[:])
		case i%3 == 1: // near-maximum digests
			for j := range d {
				d[j] = 0xff
			}
			d[rng.Intn(10)] = byte(rng.Uint64())
		default: // a digest exactly on (or one off) a true CDF boundary
			if money == 0 || expected >= total {
				rng.Read(d[:])
				break
			}
			j := rng.Uint64() % money
			tt := new(big.Int).Mul(exactCDFNum(money, total, expected, j), den256)
			tt.Div(tt, new(big.Int).Exp(new(big.Int).SetUint64(total), new(big.Int).SetUint64(money), nil))
			tt.Add(tt, big.NewInt(int64(rng.Intn(3)-1)))
			if tt.Sign() < 0 || tt.Cmp(den256) > 0 {
				rng.Read(d[:])
				break
			}
			tt.FillBytes(d[:])
		}
		check(money, total, expected, d)
	}
}

// TestSelectF128Distribution mirrors TestSortitionBasic for the pure-Go path:
// summed selection weight over many digests must track N * money * p. This is
// oracle-free AND formula-free -- a gross semantic error (wrong probability,
// shifted distribution) fails it even if perfectly mirrored everywhere else.
func TestSelectF128Distribution(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	cases := []struct {
		name                           string
		money, total, expected, rounds uint64
	}{
		{"half stake", 100, 200, 20, 1000},
		{"realistic stake", 1_000_000_000_000_000, 2_000_000_000_000_000, 1500, 200},
	}
	for _, c := range cases {
		var hits uint64
		for i := uint64(0); i < c.rounds; i++ {
			var d Digest
			rng.Read(d[:])
			hits += SelectF128(c.money, c.total, c.expected, d)
		}
		want := float64(c.rounds) * float64(c.expected) * float64(c.money) / float64(c.total)
		if diff := float64(hits) - want; diff < -0.02*want || diff > 0.02*want {
			t.Errorf("%s: %d selections over %d rounds, want %.0f +/- 2%%",
				c.name, hits, c.rounds, want)
		}
	}
}

// oracleCDFAt returns the oracle's cdf(j), mirroring selectBigOracle's
// 128-bit recurrence exactly.
func oracleCDFAt(money, totalMoney, expectedSize, j uint64) *big.Float {
	const prec = f128MantBits
	q := new(big.Float).SetPrec(prec).Quo(
		new(big.Float).SetPrec(prec).SetUint64(totalMoney-expectedSize),
		new(big.Float).SetPrec(prec).SetUint64(totalMoney))
	pq := new(big.Float).SetPrec(prec).Quo(
		new(big.Float).SetPrec(prec).SetUint64(expectedSize),
		new(big.Float).SetPrec(prec).SetUint64(totalMoney-expectedSize))
	pmf := bigIntPow(q, money, prec)
	cdf := new(big.Float).SetPrec(prec).Set(pmf)
	for i := uint64(1); i <= j; i++ {
		factor := new(big.Float).SetPrec(prec).Quo(
			new(big.Float).SetPrec(prec).SetUint64(money-i+1),
			new(big.Float).SetPrec(prec).SetUint64(i))
		step := new(big.Float).SetPrec(prec).Mul(factor, pq)
		pmf = new(big.Float).SetPrec(prec).Mul(pmf, step)
		cdf = new(big.Float).SetPrec(prec).Add(cdf, pmf)
	}
	return cdf
}

// TestSelectF128BoundaryStraddle constructs digests one ratio-ulp below, at,
// and above oracle CDF boundaries -- the knife edges where rounding bugs
// live, which uniform sampling hits with probability ~2^-128 -- and asserts
// bit-level agreement with the oracle at each, plus monotone results across
// the constructed digests.
func TestSelectF128BoundaryStraddle(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	den := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	denF := new(big.Float).SetPrec(600).SetInt(den)

	for iter := 0; iter < 150; iter++ {
		money := 2 + rng.Uint64()%250
		total := 2 + rng.Uint64()>>uint(rng.Intn(40))
		expected := rng.Uint64() % total // p < 1
		boundaries := []uint64{0, money / 2, money - 1, rng.Uint64() % money}
		for _, j := range boundaries {
			c := oracleCDFAt(money, total, expected, j)
			// One f128 ulp at c's own magnitude, in digest units: the ulp is
			// 2^(exp-128) for c in [2^(exp-1), 2^exp), so it scales with the
			// boundary -- a fixed 2^128 step would be one ulp only for c in
			// [0.5, 1) and many ulps for small boundaries like a large-mean
			// cdf(0). Clamp to one digest unit when the ulp is finer than the
			// digest grid.
			step := new(big.Int).Rsh(den, uint(128-c.MantExp(nil)))
			if step.Sign() == 0 {
				step.SetInt64(1)
			}
			tt, _ := new(big.Float).SetPrec(600).Mul(c, denF).Int(nil)
			prev := uint64(0)
			first := true
			clamped := false
			var loRatio, hiRatio *big.Float
			for k := int64(-2); k <= 2; k++ {
				ti := new(big.Int).Add(tt, new(big.Int).Mul(big.NewInt(k), step))
				if ti.Sign() < 0 || ti.Cmp(den) > 0 {
					clamped = true
					continue
				}
				var d Digest
				ti.FillBytes(d[:])
				got := SelectF128(money, total, expected, d)
				want := selectBigOracle(money, total, expected, d)
				if got != want {
					t.Fatalf("straddle mismatch at boundary j=%d offset %d ulp: SelectF128=%d oracle=%d (money=%d total=%d expected=%d vrf=%x)",
						j, k, got, want, money, total, expected, d)
				}
				if !first && got < prev {
					t.Fatalf("non-monotone across straddle at boundary j=%d: %d then %d (money=%d total=%d expected=%d)",
						j, prev, got, money, total, expected)
				}
				prev, first = got, false
				if loRatio == nil {
					loRatio = digestRatioBig(d, f128MantBits)
				}
				hiRatio = digestRatioBig(d, f128MantBits)
			}
			// The candidates must genuinely bracket the boundary (this is
			// what makes it a straddle rather than ordinary sampling); only
			// checkable when no candidate was clamped away at 0 or the
			// maximum digest.
			if !clamped && (loRatio.Cmp(c) > 0 || hiRatio.Cmp(c) < 0) {
				t.Fatalf("straddle window does not bracket boundary j=%d: [%v, %v] vs cdf=%v (money=%d total=%d expected=%d)",
					j, loRatio, hiRatio, c, money, total, expected)
			}
		}
	}
}
