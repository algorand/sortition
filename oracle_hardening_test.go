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
	"encoding/json"
	"math/big"
	"os"
	"testing"
)

// TestSelectHighPrecisionCertifiedVectors makes the ordinary big.Float
// tolerance oracle earn its precision assumption: 256, 512, and 1024-bit
// walks must converge to every independently Arb-certified large-money
// quantile. This catches precision-sensitive or shared harness defects even
// when the production result itself still matches the checked-in want value.
func TestSelectHighPrecisionCertifiedVectors(t *testing.T) {
	raw, err := os.ReadFile("testdata/f128_arb_certificates.json")
	if err != nil {
		t.Fatal(err)
	}
	var certificates arbCertificateFile
	if err := json.Unmarshal(raw, &certificates); err != nil {
		t.Fatal(err)
	}
	if len(certificates.Vectors) < 10 {
		t.Fatalf("too few certified vectors: %d", len(certificates.Vectors))
	}
	for _, vector := range certificates.Vectors {
		t.Run(vector.Label, func(t *testing.T) {
			var d Digest
			decoded, ok := new(big.Int).SetString(vector.Digest, 16)
			if !ok || decoded.Sign() < 0 || decoded.BitLen() > 256 {
				t.Fatalf("invalid digest %q", vector.Digest)
			}
			decoded.FillBytes(d[:])
			for _, prec := range []uint{256, 512, 1024} {
				if got := selectAtPrecision(vector.Money, vector.TotalMoney, vector.ExpectedSize, d, prec); got != vector.Want {
					t.Fatalf("%d-bit oracle selected %d, Arb certificate wants %d", prec, got, vector.Want)
				}
			}
		})
	}
}

func exactDigestBoundary(point, digestDen, cdfDen *big.Int, cdfNums []*big.Int) bool {
	lhs := new(big.Int).Mul(point, cdfDen)
	for _, cdfNum := range cdfNums {
		if lhs.Cmp(new(big.Int).Mul(cdfNum, digestDen)) == 0 {
			return true
		}
	}
	return false
}

// TestSelectHighPrecisionExhaustiveSmallDomain independently checks the
// high-precision harness over a complete finite parameter/boundary grid. At
// non-equality points the 512/1024-bit walks must equal exact rational
// mathematics; on exact boundaries their rounded recurrence may land one
// inclusive boundary later, but the two high precisions must still converge.
func TestSelectHighPrecisionExhaustiveSmallDomain(t *testing.T) {
	digestDen := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	const maxMoney = uint64(10)
	const maxTotal = uint64(20)
	var parameters, candidates, asserted, exactBoundaries, coarseDifferences uint64

	for money := uint64(1); money <= maxMoney; money++ {
		for total := uint64(2); total <= maxTotal; total++ {
			cdfDen := new(big.Int).Exp(new(big.Int).SetUint64(total), new(big.Int).SetUint64(money), nil)
			for expected := uint64(1); expected < total; expected++ {
				parameters++
				cdfNums := make([]*big.Int, money)
				points := map[string]*big.Int{
					"0":                new(big.Int),
					digestDen.Text(16): new(big.Int).Set(digestDen),
				}
				for j := uint64(0); j < money; j++ {
					cdfNum := exactCDFNum(money, total, expected, j)
					cdfNums[j] = cdfNum
					scaled := new(big.Int).Mul(cdfNum, digestDen)
					at := new(big.Int).Quo(scaled, cdfDen)
					if at.Sign() > 0 {
						below := new(big.Int).Sub(new(big.Int).Set(at), big.NewInt(1))
						points[below.Text(16)] = below
					}
					points[at.Text(16)] = new(big.Int).Set(at)
					if at.Cmp(digestDen) < 0 {
						above := new(big.Int).Add(new(big.Int).Set(at), big.NewInt(1))
						points[above.Text(16)] = above
					}
				}

				for _, point := range points {
					candidates++
					d := digestFromBigInt(t, point, digestDen)
					if inFrozenSliver(money, digestRatioBig(d, 512)) {
						continue
					}
					asserted++
					want := exactSelectFromCDFNumerators(money, cdfNums, cdfDen, digestDen, point)
					got256 := selectAtPrecision(money, total, expected, d, 256)
					got512 := selectAtPrecision(money, total, expected, d, 512)
					got1024 := selectAtPrecision(money, total, expected, d, 1024)
					if got512 != got1024 {
						t.Fatalf("oracle failed to converge: 512=%d 1024=%d (money=%d total=%d expected=%d digest=%x)",
							got512, got1024, money, total, expected, d)
					}
					if got256 != got512 {
						coarseDifferences++
						if uint64Distance(got256, got512) > 1 {
							t.Fatalf("256-bit oracle differs by more than one boundary: 256=%d 512=%d", got256, got512)
						}
					}
					if exactDigestBoundary(point, digestDen, cdfDen, cdfNums) {
						exactBoundaries++
						if uint64Distance(got1024, want) > 1 {
							t.Fatalf("1024-bit oracle=%d vs exact boundary=%d", got1024, want)
						}
					} else if got1024 != want {
						t.Fatalf("1024-bit oracle=%d vs exact=%d (money=%d total=%d expected=%d digest=%x)",
							got1024, want, money, total, expected, d)
					}
				}
			}
		}
	}
	if parameters < 1_000 || asserted < 20_000 || exactBoundaries == 0 || coarseDifferences < 1_000 {
		t.Fatalf("anti-vacuity: parameters=%d candidates=%d asserted=%d exactBoundaries=%d coarseDifferences=%d",
			parameters, candidates, asserted, exactBoundaries, coarseDifferences)
	}
	t.Logf("oracle grid: parameters=%d candidates=%d asserted=%d exactBoundaries=%d 256vs512=%d",
		parameters, candidates, asserted, exactBoundaries, coarseDifferences)
}
