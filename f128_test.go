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
	"math/rand"
	"testing"
)

// selectBigOracle is an independent, "obviously correct" reference for SelectF128:
// the SAME binomial-CDF walk, but every arithmetic step uses math/big.Float (Go's
// standard arbitrary-precision float, round-to-nearest-even) at the f128 mantissa
// width instead of the hand-rolled f128. SelectF128 is fuzz-checked to be
// bit-identical to this (FuzzSelectF128), which is what justifies trusting the
// hand-rolled integer arithmetic in f128.go. Mirrors binomialCDFWalkF128 exactly.
//
// Valid range: big.Float's exponent is an int32, so pmf0 = (1-p)^money underflows
// to exactly 0 once it drops below ~2^(-2^31). That requires the mean money*p to
// exceed ~1.5e9 (a committee of ~1.5 billion) -- far outside any reachable
// sortition input, where the mean is a committee size (<= a few thousand). So the
// oracle is exact for every reachable input: FuzzSelectF128 stays well inside it
// (money %= 3001), and TestF128MatchesOracleLargeMoney exercises realistic large
// money (up to ~2^51) where it is likewise exact. (A truly unbounded oracle would
// need big.Rat, which is infeasible at large money -- (1-p)^money has a
// total^money denominator -- so this range, covering all reachable inputs, is the
// practical maximum.)
func selectBigOracle(money uint64, totalMoney uint64, expectedSize uint64, vrfOutput Digest) uint64 {
	const prec = f128MantBits
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
	pmf := bigIntPow(q, money, prec) // (1-p)^money
	cdf := new(big.Float).SetPrec(prec).Set(pmf)
	if cdf.Cmp(ratio) >= 0 {
		return 0
	}
	for j := uint64(1); j < money; j++ {
		factor := new(big.Float).SetPrec(prec).Quo(
			new(big.Float).SetPrec(prec).SetUint64(money-j+1),
			new(big.Float).SetPrec(prec).SetUint64(j))
		step := new(big.Float).SetPrec(prec).Mul(factor, pq)
		pmf = new(big.Float).SetPrec(prec).Mul(pmf, step)
		cdf = new(big.Float).SetPrec(prec).Add(cdf, pmf)
		if cdf.Cmp(ratio) >= 0 {
			return j
		}
	}
	return money
}

func digestRatioBig(d Digest, prec uint) *big.Float {
	numerator := new(big.Int).SetBytes(d[:])
	denominator := new(big.Int).Lsh(big.NewInt(1), DigestSize*8)
	denominator.Sub(denominator, big.NewInt(1))
	return new(big.Float).SetPrec(prec).Quo(
		new(big.Float).SetInt(numerator),
		new(big.Float).SetInt(denominator),
	)
}

func bigIntPow(base *big.Float, e uint64, prec uint) *big.Float {
	result := new(big.Float).SetPrec(prec).SetInt64(1)
	b := new(big.Float).SetPrec(prec).Set(base)
	for e > 0 {
		if e&1 == 1 {
			result = new(big.Float).SetPrec(prec).Mul(result, b)
		}
		e >>= 1
		if e > 0 {
			b = new(big.Float).SetPrec(prec).Mul(b, b)
		}
	}
	return result
}

// FuzzSelectF128 differentially fuzzes the hand-rolled f128 implementation
// against the independent math/big.Float oracle.
func FuzzSelectF128(f *testing.F) {
	seedVRF := append(bytes.Repeat([]byte{0xff}, 7), make([]byte, 25)...)
	f.Add(uint64(1954), uint64(1_999_999_999_999_964), uint64(1500), seedVRF)
	f.Add(uint64(1141), uint64(1000), uint64(250), seedVRF)
	f.Add(uint64(0), uint64(2_000_000_000_000_000), uint64(20), make([]byte, 32))
	f.Add(uint64(1000), uint64(1000), uint64(1000), make([]byte, 32))
	f.Add(uint64(100), uint64(1000), uint64(2000), make([]byte, 32)) // expectedSize > totalMoney
	// all-0xff digest: ratio is exactly 1.0, the cdf-reaches-1.0 regime
	f.Add(uint64(1954), uint64(1_999_999_999_999_964), uint64(1500), bytes.Repeat([]byte{0xff}, 32))

	// The uint64 expectedSize needs no input filtering: NaN/Inf/negative/
	// fractional sizes are unrepresentable, and expectedSize >= totalMoney
	// (including totalMoney == 0) takes the exact-integer p >= 1 path in both
	// implementations.
	f.Fuzz(func(t *testing.T, money, total, expected uint64, vrf []byte) {
		money %= 3001 // bound the walk so each fuzz exec stays fast
		var d Digest
		copy(d[:], vrf)
		got := SelectF128(money, total, expected, d)
		want := selectBigOracle(money, total, expected, d)
		if got != want {
			t.Fatalf("SelectF128=%d != big.Float oracle=%d (money=%d total=%d expected=%d vrf=%x)",
				got, want, money, total, expected, d)
		}
	})
}

// TestF128AgreesWithCurrent checks broad agreement with the deployed
// Boost-double implementation. Knife-edge differences remain expected because
// SelectF128 uses an f128 digest ratio and CDF.
func TestF128AgreesWithCurrent(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const total = uint64(2_000_000_000_000_000)
	committees := []uint64{20, 1500, 2990, 6000}
	const n = 200000
	match, considered, diffs := 0, 0, 0
	for i := 0; i < n; i++ {
		exp := committees[rng.Intn(len(committees))]
		mean := 0.05 + rng.Float64()*30
		money := uint64(mean * float64(total) / float64(exp))
		if money == 0 {
			continue
		}
		var d Digest
		rng.Read(d[:])
		considered++
		cpp := Select(money, total, float64(exp), d)
		f := SelectF128(money, total, exp, d)
		if cpp == f {
			match++
			continue
		}
		diffs++
		if diffs <= 10 {
			t.Logf("knife-edge diff: money=%d exp=%d cpp=%d f128=%d", money, exp, cpp, f)
		}
	}
	t.Logf("SelectF128 vs C++ Select: %d/%d agree (%d differ)", match, considered, diffs)
	if match*1000 < considered*999 {
		t.Errorf("agreement %d/%d too low", match, considered)
	}
}

func TestF128DigestRatioMatchesBigFloat(t *testing.T) {
	cases := make([]Digest, 0, 1389)
	cases = append(cases, Digest{})
	setBit := func(d *Digest, bit int) {
		d[len(d)-1-bit/8] |= byte(1) << uint(bit%8)
	}

	// Exercise every possible normalization shift.
	for bit := 0; bit < DigestSize*8; bit++ {
		var d Digest
		setBit(&d, bit)
		cases = append(cases, d)
	}

	// Exercise exact halfway tails at every shift where a tail remains. The
	// denominator correction must make each of these round upward.
	for leading := 0; leading < f128MantBits; leading++ {
		var d Digest
		setBit(&d, DigestSize*8-1-leading)
		setBit(&d, f128MantBits-1-leading)
		cases = append(cases, d)
	}

	var one Digest
	one[len(one)-1] = 1
	cases = append(cases, one)

	var halfway Digest
	halfway[0] = 0x80
	halfway[16] = 0x80
	cases = append(cases, halfway)

	var maximum Digest
	for i := range maximum {
		maximum[i] = 0xff
	}
	cases = append(cases, maximum)

	var nearMaximum Digest
	for i := 0; i < 7; i++ {
		nearMaximum[i] = 0xff
	}
	cases = append(cases, nearMaximum)

	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 1000; i++ {
		var d Digest
		rng.Read(d[:])
		cases = append(cases, d)
	}

	for _, d := range cases {
		got := f128ToBig(f128FromDigestRatio(d))
		want := digestRatioBig(d, f128MantBits)
		if got.Cmp(want) != 0 {
			t.Fatalf("digest ratio mismatch for %x: f128=%v big.Float=%v", d, got, want)
		}
	}
}

func TestSelectF128NearMaximumDigest(t *testing.T) {
	var d Digest
	for i := 0; i < 7; i++ {
		d[i] = 0xff
	}
	got := SelectF128(1954, 1_999_999_999_999_964, 1500, d)
	if got != 1 {
		t.Fatalf("SelectF128=%d, want 1 for exact near-maximum digest ratio", got)
	}
}

// TestSelectF128RatioExactlyOne pins the walk when the f128 ratio is exactly
// 1.0: mathematically for the all-0xff digest, and by 128-bit rounding for any
// digest with at least 129 leading one bits. With the f128-rounded threshold
// fixed at 1.0, money is the exact-CDF count; the walk returns an earlier j
// only when the accumulated f128 CDF happens to round up to exactly 1.0 (see
// the SelectF128 doc comment). Each case pins one branch:
//
//   - money=1954 with total=1_999_999_999_999_964: stops early at j=3, while
//     the same distribution with total=2_000_000_000_000_000 (36 more) falls
//     through to money. A hair-trigger pair pinned together: if a rounding
//     change flips either, the cdf trajectory moved by an ulp -- the walk did
//     not break.
//   - money=100, p=1/2: provably falls through to money -- cdf(99) is
//     1 - 2^-100, which sits 2^28 ulps below 1.0, a gap no rounding can
//     bridge.
//   - money=129, p=1/2: the exact boundary -- true cdf(128) = 1 - 2^-129 is
//     precisely the rounding midpoint, and ties-to-even rounds it up to
//     exactly 1.0, stopping at j=128.
func TestSelectF128RatioExactlyOne(t *testing.T) {
	one := f128FromUint64(1)

	var maximum Digest
	for i := range maximum {
		maximum[i] = 0xff
	}
	// exactly 129 leading one bits: the minimal digest that rounds to 1.0
	var minLeadingOnes Digest
	for i := 0; i < 16; i++ {
		minLeadingOnes[i] = 0xff
	}
	minLeadingOnes[16] = 0x80

	cases := []struct {
		money, total, expected uint64
		want                   uint64
	}{
		{1954, 1_999_999_999_999_964, 1500, 3},
		{1954, 2_000_000_000_000_000, 1500, 1954},
		{100, 200, 100, 100},
		{129, 258, 129, 128},
	}
	for _, d := range []Digest{maximum, minLeadingOnes} {
		if f128FromDigestRatio(d).cmp(one) != 0 {
			t.Fatalf("digest %x: ratio is not exactly 1.0", d)
		}
		for _, c := range cases {
			got := SelectF128(c.money, c.total, c.expected, d)
			if oracle := selectBigOracle(c.money, c.total, c.expected, d); got != oracle {
				t.Fatalf("digest %x money=%d: SelectF128=%d != oracle=%d", d, c.money, got, oracle)
			}
			if got != c.want {
				t.Fatalf("digest %x money=%d: SelectF128=%d, want %d", d, c.money, got, c.want)
			}
		}
	}
}

// FuzzF128Ops validates the f128 arithmetic primitives directly against
// math/big.Float at the same mantissa width.
func FuzzF128Ops(f *testing.F) {
	f.Add(uint64(1)<<63, uint64(0), 0, uint64(3)<<62, uint64(0), 0, uint64(7))
	f.Add(uint64(0), uint64(0), 0, uint64(1)<<63, uint64(1), -5, uint64(1))
	f.Fuzz(func(t *testing.T, ahi, alo uint64, aexp int, bhi, blo uint64, bexp int, u uint64) {
		a := norm128(ahi, alo, aexp%4000-2000)
		b := norm128(bhi, blo, bexp%4000-2000)
		ab, bb := f128ToBig(a), f128ToBig(b)
		check := func(name string, got f128, want *big.Float) {
			gotBig := f128ToBig(got)
			if gotBig.Cmp(want) != 0 {
				t.Fatalf("%s: f128=%v big.Float=%v (a=%v b=%v u=%d)", name, gotBig, want, ab, bb, u)
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
	})
}

// TestF128MatchesOracleLargeMoney extends the strict SelectF128 == oracle check
// beyond FuzzSelectF128's money<=3000 cap to realistic LARGE money (up to ~2^51),
// where the big.Float oracle is still exact (mean = money*p stays a committee
// size). Each committee's mean is drawn in [0.05, size] so money stays <= total.
func TestF128MatchesOracleLargeMoney(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const total = uint64(2_000_000_000_000_000)
	committees := []uint64{20, 1500, 2990, 6000}
	for i := 0; i < 3000; i++ {
		size := committees[rng.Intn(len(committees))]
		mean := 0.05 + rng.Float64()*float64(size)             // mean <= size  =>  money <= total
		money := uint64(mean * float64(total) / float64(size)) // up to ~2^51
		if money == 0 {
			continue
		}
		var d Digest
		rng.Read(d[:])
		if got, want := SelectF128(money, total, size, d), selectBigOracle(money, total, size, d); got != want {
			t.Fatalf("SelectF128=%d != oracle=%d (money=%d size=%d vrf=%x)", got, want, money, size, d)
		}
	}
}

// BenchmarkSelectF128 mirrors BenchmarkSortition (same parameters) so the pure-Go
// deterministic path can be compared directly against the cgo/Boost Select.
func BenchmarkSelectF128(b *testing.B) {
	b.StopTimer()
	keys := make([]Digest, b.N)
	for i := 0; i < b.N; i++ {
		rand.Read(keys[i][:])
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		SelectF128(1000000, 1000000000000, 2500, keys[i])
	}
}

// f128ToBig returns the exact value of an f128 as a big.Float (test helper).
func f128ToBig(x f128) *big.Float {
	hi := new(big.Float).SetPrec(300).SetUint64(x.hi)
	hi.SetMantExp(hi, 64)
	m := new(big.Float).SetPrec(300).Add(hi, new(big.Float).SetPrec(300).SetUint64(x.lo))
	return m.SetMantExp(m, x.exp)
}

// TestDivVsBig checks f128.div against a 128-bit round-nearest-even big.Float
// divide on broad random operands. The SelectF128 fuzz only exercises div with
// the p/(1-p) shape, so this independently validates the hand-rolled Knuth
// long division across the full input space (spanning exponents and the
// Q<2^128 vs Q>=2^128 normalization cases).
func TestDivVsBig(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	randF128 := func() f128 {
		return f128{rng.Uint64() | 1<<63, rng.Uint64(), rng.Intn(4000) - 2000} // normalized
	}
	for i := 0; i < 5_000_000; i++ {
		a, b := randF128(), randF128()
		got := f128ToBig(a.div(b))
		want := new(big.Float).SetPrec(f128MantBits).Quo(f128ToBig(a), f128ToBig(b))
		if got.Cmp(want) != 0 {
			t.Fatalf("div mismatch: a=%+v b=%+v got=%s want=%s", a, b, got.Text('p', 0), want.Text('p', 0))
		}
	}
}
