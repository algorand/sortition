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
	ulp := new(big.Int).Lsh(big.NewInt(1), 128) // one 128-bit ratio ulp in digest units

	for iter := 0; iter < 150; iter++ {
		money := 2 + rng.Uint64()%250
		total := 2 + rng.Uint64()>>uint(rng.Intn(40))
		expected := rng.Uint64() % total // p < 1
		boundaries := []uint64{0, money / 2, money - 1, rng.Uint64() % money}
		for _, j := range boundaries {
			c := oracleCDFAt(money, total, expected, j)
			tt, _ := new(big.Float).SetPrec(600).Mul(c, denF).Int(nil)
			prev := uint64(0)
			first := true
			for k := int64(-2); k <= 2; k++ {
				ti := new(big.Int).Add(tt, new(big.Int).Mul(big.NewInt(k), ulp))
				if ti.Sign() < 0 || ti.Cmp(den) > 0 {
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
			}
		}
	}
}
