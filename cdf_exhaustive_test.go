// Copyright (C) 2019-2026 Algorand Foundation Ltd.
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

func exactSelectFromCDFNumerators(money uint64, cdfNums []*big.Int, cdfDen, digestDen, digest *big.Int) uint64 {
	lhs := new(big.Int).Mul(digest, cdfDen)
	for j, cdfNum := range cdfNums {
		rhs := new(big.Int).Mul(cdfNum, digestDen)
		if lhs.Cmp(rhs) <= 0 {
			return uint64(j)
		}
	}
	return money
}

func digestFromBigInt(t *testing.T, x, digestDen *big.Int) Digest {
	t.Helper()
	if x.Sign() < 0 || x.Cmp(digestDen) > 0 {
		t.Fatalf("digest integer outside [0, 2^256-1]: %s", x)
	}
	var d Digest
	x.FillBytes(d[:])
	return d
}

func uint64Distance(a, b uint64) uint64 {
	if a >= b {
		return a - b
	}
	return b - a
}

// TestSelectF128ExhaustiveSmallDomain complements the sampled exact-rational
// test with a complete finite grid. Every non-degenerate rational probability,
// every CDF boundary, and the adjacent digest integers are constructed from
// the first-principles binomial formula. The f128 walk is allowed to move one
// boundary because its per-operation RNE semantics intentionally differ from
// exact mathematics at knife edges.
func TestSelectF128ExhaustiveSmallDomain(t *testing.T) {
	digestDen := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	const maxMoney = uint64(16)
	const maxTotal = uint64(32)

	var parameters, boundaries, bracketed, candidates, asserted, skippedSliver uint64
	for money := uint64(1); money <= maxMoney; money++ {
		for total := uint64(2); total <= maxTotal; total++ {
			cdfDen := new(big.Int).Exp(new(big.Int).SetUint64(total), new(big.Int).SetUint64(money), nil)
			for expected := uint64(1); expected < total; expected++ {
				parameters++
				cdfNums := make([]*big.Int, money)
				points := make(map[string]*big.Int, 3*money+2)
				points["0"] = new(big.Int)
				points[digestDen.Text(16)] = new(big.Int).Set(digestDen)

				for j := uint64(0); j < money; j++ {
					boundaries++
					cdfNum := exactCDFNum(money, total, expected, j)
					cdfNums[j] = cdfNum
					scaled := new(big.Int).Mul(cdfNum, digestDen)
					at, rem := new(big.Int), new(big.Int)
					at.QuoRem(scaled, cdfDen, rem)

					// at/digestDen <= CDF(j) < (at+1)/digestDen. This
					// assertion makes every generated window prove that it
					// really straddles the intended exact boundary.
					atScaled := new(big.Int).Mul(new(big.Int).Set(at), cdfDen)
					above := new(big.Int).Add(new(big.Int).Set(at), big.NewInt(1))
					aboveScaled := new(big.Int).Mul(new(big.Int).Set(above), cdfDen)
					if atScaled.Cmp(scaled) > 0 || aboveScaled.Cmp(scaled) <= 0 || at.Cmp(digestDen) >= 0 {
						t.Fatalf("invalid exact boundary bracket (money=%d total=%d expected=%d j=%d at=%s remainder=%s)",
							money, total, expected, j, at, rem)
					}
					bracketed++

					if at.Sign() > 0 {
						below := new(big.Int).Sub(new(big.Int).Set(at), big.NewInt(1))
						points[below.Text(16)] = below
					}
					points[at.Text(16)] = new(big.Int).Set(at)
					points[above.Text(16)] = above
				}

				for _, point := range points {
					candidates++
					d := digestFromBigInt(t, point, digestDen)
					got := SelectF128(money, total, expected, d)
					if bitExact := selectBigOracle(money, total, expected, d); got != bitExact {
						t.Fatalf("SelectF128=%d vs 128-bit walk oracle=%d (money=%d total=%d expected=%d digest=%x)",
							got, bitExact, money, total, expected, d)
					}
					if inFrozenSliver(money, digestRatioBig(d, 512)) {
						skippedSliver++
						continue
					}
					want := exactSelectFromCDFNumerators(money, cdfNums, cdfDen, digestDen, point)
					asserted++
					if uint64Distance(got, want) > 1 {
						t.Fatalf("SelectF128=%d vs exact=%d (money=%d total=%d expected=%d digest=%x)",
							got, want, money, total, expected, d)
					}
				}
			}
		}
	}

	if boundaries == 0 || bracketed != boundaries {
		t.Fatalf("anti-vacuity: bracketed %d of %d exact boundaries", bracketed, boundaries)
	}
	if parameters < 7_000 || asserted < 100_000 || skippedSliver >= candidates/2 {
		t.Fatalf("anti-vacuity: parameters=%d boundaries=%d candidates=%d asserted=%d skippedSliver=%d",
			parameters, boundaries, candidates, asserted, skippedSliver)
	}
	t.Logf("exhaustive exact grid: parameters=%d boundaries=%d candidates=%d asserted=%d skippedSliver=%d",
		parameters, boundaries, candidates, asserted, skippedSliver)
}

// TestSelectF128ExhaustiveSmallDegenerateDomain keeps p=0, p>=1,
// totalMoney=0, and money=0 out of the regular-grid special cases above. It
// exhausts them separately with endpoint digests so none is lost to a skip.
func TestSelectF128ExhaustiveSmallDegenerateDomain(t *testing.T) {
	var maximum Digest
	for i := range maximum {
		maximum[i] = 0xff
	}
	var one Digest
	one[len(one)-1] = 1
	digests := []Digest{{}, one, maximum}

	for money := uint64(0); money <= 16; money++ {
		for total := uint64(0); total <= 32; total++ {
			expectedValues := map[uint64]struct{}{0: {}, total: {}}
			if total != ^uint64(0) {
				expectedValues[total+1] = struct{}{}
			}
			for expected := range expectedValues {
				for _, d := range digests {
					got := SelectF128(money, total, expected, d)
					var want uint64
					switch {
					case money == 0:
						want = 0
					case expected < total: // only expected==0: all mass at zero
						want = 0
					case d == (Digest{}): // p>=1 and ratio==0 crosses cdf(0)==0
						want = 0
					default:
						want = money
					}
					if got != want {
						t.Fatalf("degenerate SelectF128=%d, want %d (money=%d total=%d expected=%d digest=%x)",
							got, want, money, total, expected, d)
					}
				}
			}
		}
	}
}
