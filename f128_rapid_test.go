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
	"bytes"
	"math/big"
	"testing"

	"pgregory.net/rapid"
)

// These rapid property tests complement the go fuzz targets rather than
// replacing them. Go's fuzzer mutates locally around its corpus, so integer
// arguments dwell near seed values -- divU's shallow-quotient misrounding
// (u > ~2^62) survived ~30M fuzz execs with u=1 and u=7 seeds, while rapid
// found it within its first hundred generated cases. The generators below add
// explicit magnitude-band guidance on top of rapid's own boundary bias so
// every regime is sampled every run. Ordinary go test runs 100 cases per
// test; CI runs them again at -rapid.checks=20000, and a long run can go
// deeper:
//
//	go test -run TestRapid -rapid.checks=100000
//
// A failure writes a minimized reproducer under testdata/rapid/.

// bandedUint64 draws a uint64 spread across magnitude bands: the sortition
// walk's divisor domain, the beyond-domain deep-quotient region, and the two
// shallow-quotient bands where a divU rounding bug previously hid.
func bandedUint64(t *rapid.T, label string) uint64 {
	return rapid.OneOf(
		rapid.Uint64Range(0, SelectF128MaxMoney),     // walk domain (u = step index)
		rapid.Uint64Range(SelectF128MaxMoney, 1<<62), // deep quotients beyond the domain
		rapid.Uint64Range(1<<62, 1<<63),              // quotient round bit in the last digit
		rapid.Uint64Range(1<<63, ^uint64(0)),         // 128-bit shallow quotients
	).Draw(t, label)
}

// bandedMantissa draws mantissa words by structure, not just value: uniform
// bits, sparse (a few set bits), or dense (all ones with a few holes). Sparse
// and dense mantissas raise the density of exact ties and of carry-out
// renormalization, which sit near ~2^-64 measure under uniform draws.
func bandedMantissa(t *rapid.T, label string) (uint64, uint64) {
	switch rapid.IntRange(0, 2).Draw(t, label+"Kind") {
	case 0:
		return rapid.Uint64().Draw(t, label+"Hi"), rapid.Uint64().Draw(t, label+"Lo")
	case 1: // sparse
		var hi, lo uint64
		for range rapid.IntRange(1, 3).Draw(t, label+"Bits") {
			b := rapid.IntRange(0, 127).Draw(t, label+"Bit")
			if b >= 64 {
				hi |= 1 << (b - 64)
			} else {
				lo |= 1 << b
			}
		}
		return hi, lo
	default: // dense
		hi, lo := ^uint64(0), ^uint64(0)
		for range rapid.IntRange(0, 3).Draw(t, label+"Holes") {
			b := rapid.IntRange(0, 127).Draw(t, label+"Hole")
			if b >= 64 {
				hi &^= 1 << (b - 64)
			} else {
				lo &^= 1 << b
			}
		}
		return hi, lo
	}
}

// TestRapidF128Ops property-tests every f128 primitive against 128-bit
// big.Float, mirroring FuzzF128Ops, plus the oracle-independent commutativity
// of mul and add (asymmetric partial-product assembly cannot hide from these
// even if a bug were somehow mirrored into the reference computation).
func TestRapidF128Ops(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		ahi, alo := bandedMantissa(t, "a")
		aexp := rapid.Int64Range(-2000, 2000).Draw(t, "aexp")
		a := norm128(ahi, alo, aexp)
		// Band the exponent gap: add's alignment boundaries sit at gaps of
		// exactly 64, 127, 128, and 129, which a uniform pair of exponents
		// rarely produces.
		gap := rapid.OneOf(
			rapid.Int64Range(-4, 4),
			rapid.Int64Range(-68, -60), rapid.Int64Range(60, 68),
			rapid.Int64Range(-132, -124), rapid.Int64Range(124, 132),
			rapid.Int64Range(-2000, 2000),
		).Draw(t, "gap")
		bhi, blo := bandedMantissa(t, "b")
		b := norm128(bhi, blo, aexp-gap)
		u := bandedUint64(t, "u")
		ab, bb := f128ToBig(a), f128ToBig(b)
		check := func(name string, got f128, want *big.Float) {
			if g := f128ToBig(got); g.Cmp(want) != 0 {
				t.Fatalf("%s: f128=%v big.Float=%v (a=%v b=%v u=%d)", name, g, want, ab, bb, u)
			}
		}
		check("mul", a.mul(b), new(big.Float).SetPrec(f128MantBits).Mul(ab, bb))
		check("add", a.add(b), new(big.Float).SetPrec(f128MantBits).Add(ab, bb))
		if u != 0 {
			check("divU", a.divU(u),
				new(big.Float).SetPrec(f128MantBits).Quo(ab, new(big.Float).SetPrec(f128MantBits).SetUint64(u)))
		}
		if !b.isZero() {
			check("div", a.div(b), new(big.Float).SetPrec(f128MantBits).Quo(ab, bb))
		}
		if x, y := a.mul(b), b.mul(a); x != y {
			t.Fatalf("mul not commutative: %+v vs %+v (a=%v b=%v)", x, y, ab, bb)
		}
		if x, y := a.add(b), b.add(a); x != y {
			t.Fatalf("add not commutative: %+v vs %+v (a=%v b=%v)", x, y, ab, bb)
		}
	})
}

// TestRapidSelectF128VsOracle property-tests the full walk against the
// big.Float oracle, mirroring FuzzSelectF128. Digests are drawn both uniformly
// and from the near-maximum regime (mostly-0xff), where the ratio ~1 tail
// edges live.
func TestRapidSelectF128VsOracle(t *testing.T) {
	nearMaxDigest := rapid.Custom(func(t *rapid.T) []byte {
		b := bytes.Repeat([]byte{0xff}, DigestSize)
		i := rapid.IntRange(0, DigestSize-1).Draw(t, "hole")
		b[i] = rapid.Byte().Draw(t, "holeval")
		return b
	})
	rapid.Check(t, func(t *rapid.T) {
		money := rapid.Uint64Range(0, 3000).Draw(t, "money") // bound the walk, as in FuzzSelectF128
		total := rapid.OneOf(
			rapid.Uint64Range(0, 1_000_000),
			rapid.Uint64Range(1_000_000, 10_000_000_000_000_000), // through the supply ceiling
			rapid.Uint64Range(10_000_000_000_000_000, ^uint64(0)),
		).Draw(t, "total")
		expected := rapid.OneOf(
			rapid.Uint64Range(0, 10_000),          // committee sizes
			rapid.Uint64Range(10_000, ^uint64(0)), // through and beyond p >= 1
		).Draw(t, "expected")
		var d Digest
		copy(d[:], rapid.OneOf(
			rapid.SliceOfN(rapid.Byte(), DigestSize, DigestSize),
			nearMaxDigest,
		).Draw(t, "vrf"))
		got := SelectF128(money, total, expected, d)
		want := selectBigOracle(money, total, expected, d)
		if got != want {
			t.Fatalf("SelectF128=%d != oracle=%d (money=%d total=%d expected=%d vrf=%x)",
				got, want, money, total, expected, d)
		}
	})
}

// TestRapidSelectF128ScaleInvariance checks an ORACLE-INDEPENDENT exact
// property: scaling totalMoney and expectedSize by a common power of two
// leaves every rational in the walk identical -- the same quotients round to
// the same 128-bit values -- so the result must be BIT-IDENTICAL. Unlike
// tolerance-based metamorphic properties (monotonicity in p or money), this
// cannot flake at knife edges, and it catches numerator/denominator
// mishandling without consulting the oracle.
func TestRapidSelectF128ScaleInvariance(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		money := rapid.Uint64Range(0, 3000).Draw(t, "money")
		total := rapid.Uint64Range(0, 1<<40).Draw(t, "total")
		expected := rapid.Uint64Range(0, 1<<40).Draw(t, "expected")
		k := rapid.IntRange(1, 23).Draw(t, "k")
		var d Digest
		copy(d[:], rapid.SliceOfN(rapid.Byte(), DigestSize, DigestSize).Draw(t, "vrf"))
		base := SelectF128(money, total, expected, d)
		scaled := SelectF128(money, total<<k, expected<<k, d)
		if base != scaled {
			t.Fatalf("scale variance: SelectF128=%d but <<%d gives %d (money=%d total=%d expected=%d vrf=%x)",
				base, k, scaled, money, total, expected, d)
		}
	})
}

// TestRapidSelectF128DigestMonotonic checks an ORACLE-INDEPENDENT property:
// for fixed (money, total, expected), the selection count is non-decreasing in
// the digest. This holds exactly -- the digest-to-ratio conversion is monotone
// (round-to-nearest of a monotone quotient, and the halfway correction only
// ever rounds up), and a larger ratio can only cross the same CDF boundaries
// later or freeze to money. The differential tests share one structural blind
// spot: a defect mirrored into the big.Float oracle (as the pmf(0) plateau
// was) is invisible to them; a property test against mathematics is not.
func TestRapidSelectF128DigestMonotonic(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		money := rapid.Uint64Range(0, 3000).Draw(t, "money")
		total := rapid.Uint64Range(0, 10_000_000_000_000_000).Draw(t, "total")
		expected := rapid.Uint64Range(0, 10_000).Draw(t, "expected")
		var d1, d2 Digest
		copy(d1[:], rapid.SliceOfN(rapid.Byte(), DigestSize, DigestSize).Draw(t, "vrf"))
		// OR-ing random bits into d1 yields d2 >= d1 as a 256-bit integer.
		d2 = d1
		for _, i := range rapid.SliceOfN(rapid.IntRange(0, DigestSize-1), 1, 8).Draw(t, "orBytes") {
			d2[i] |= rapid.Byte().Draw(t, "orVal")
		}
		low, high := SelectF128(money, total, expected, d1), SelectF128(money, total, expected, d2)
		if low > high {
			t.Fatalf("monotonicity violated: SelectF128(d1)=%d > SelectF128(d2)=%d with d1 <= d2 (money=%d total=%d expected=%d d1=%x d2=%x)",
				low, high, money, total, expected, d1, d2)
		}
	})
}
