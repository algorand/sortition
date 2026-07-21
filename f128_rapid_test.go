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
// every regime is sampled every run. Raise the case count in a long run with:
//
//	go test -run TestRapid -rapid.checks=100000
//
// A failure writes a minimized reproducer under testdata/rapid/.

// bandedUint64 draws a uint64 spread across magnitude bands: the sortition
// walk's divisor domain, the beyond-domain deep-quotient region, and the two
// shallow-quotient bands where a divU rounding bug previously hid.
func bandedUint64(t *rapid.T, label string) uint64 {
	return rapid.OneOf(
		rapid.Uint64Range(0, 1<<57),          // walk domain (u = step index)
		rapid.Uint64Range(1<<57, 1<<62),      // deep quotients beyond the domain
		rapid.Uint64Range(1<<62, 1<<63),      // quotient round bit in the last digit
		rapid.Uint64Range(1<<63, ^uint64(0)), // 128-bit shallow quotients
	).Draw(t, label)
}

// TestRapidF128Ops property-tests every f128 primitive against 128-bit
// big.Float, mirroring FuzzF128Ops.
func TestRapidF128Ops(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		a := norm128(rapid.Uint64().Draw(t, "ahi"), rapid.Uint64().Draw(t, "alo"),
			rapid.Int64Range(-2000, 2000).Draw(t, "aexp"))
		b := norm128(rapid.Uint64().Draw(t, "bhi"), rapid.Uint64().Draw(t, "blo"),
			rapid.Int64Range(-2000, 2000).Draw(t, "bexp"))
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
