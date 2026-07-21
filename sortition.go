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

// #cgo CFLAGS: -O3
// #cgo CXXFLAGS: -std=c++11 -Wno-deprecated
// #include <stdint.h>
// #include <stdlib.h>
// #include "sortition.h"
import "C"

import (
	"crypto/sha512"
	"fmt"
	"math/big"
	"strings"
)

// DigestSize is the number of bytes in the preferred hash Digest used here.
const DigestSize = sha512.Size256

// Digest represents a 32-byte value holding the 256-bit Hash digest.
type Digest [DigestSize]byte

const precision = uint(8 * (DigestSize + 1))

var maxFloat *big.Float

// Select runs the sortition function and returns the number of time the key was selected
func Select(money uint64, totalMoney uint64, expectedSize float64, vrfOutput Digest) uint64 {
	binomialN := float64(money)
	binomialP := expectedSize / float64(totalMoney)

	t := &big.Int{}
	t.SetBytes(vrfOutput[:])

	h := big.Float{}
	h.SetPrec(precision)
	h.SetInt(t)

	ratio := big.Float{}
	cratio, _ := ratio.Quo(&h, maxFloat).Float64()

	return uint64(C.sortition_binomial_cdf_walk(C.double(binomialN), C.double(binomialP), C.double(cratio), C.uint64_t(money)))
}

// SelectF128 is a deterministic sortition function. It evaluates both the VRF
// ratio and binomial CDF at f128 precision using software integer arithmetic, so
// its result is bit-reproducible across platforms.
//
// Unlike Select, SelectF128 takes the committee size as the exact uint64 it is
// in the protocol rather than a float64. The distribution constants are then
// single correctly-rounded f128 divides of exact integers -- float64(totalMoney)
// is inexact above 2^53, and a float64 p = expectedSize/totalMoney would stack
// further roundings -- and invalid probabilities (NaN, Inf, negative,
// fractional) are unrepresentable. p >= 1, i.e. expectedSize >= totalMoney, is
// an exact integer comparison, handled as all probability mass at money.
//
// CONSENSUS / MIGRATION NOTE. SelectF128 is not bit-identical to the deployed
// Boost-double Select. They agree on the overwhelming majority of inputs but can
// differ at knife-edge VRF outputs near a CDF boundary. SelectF128 also preserves
// the digest ratio at f128 precision instead of first rounding it to float64;
// this avoids the multi-step tail divergence when a near-maximum digest rounds
// to 1.0 in float64. The success probability is likewise formed at f128
// precision from the integer expectedSize/totalMoney instead of a float64
// quotient. Generic CDF-boundary differences can still change a selection by
// one, so replacing Select with SelectF128 remains a protocol-gated,
// network-coordinated consensus change.
//
// One tail edge: the top ~2^-129 of digest space rounds to an f128 ratio of
// exactly 1.0 (only the all-0xff digest IS exactly 1.0; the rest of the
// interval rounds up to it). With the threshold fixed at 1.0, the exact CDF is
// < 1 for every j < money and the walk never evaluates cdf(money) == 1, so the
// exact-CDF count is money -- and the walk returns money unless the accumulated
// f128 CDF happens to round up to exactly 1.0 at an earlier j (a rounding
// artifact), in which case it returns that j. Both outcomes occur, decided
// per-distribution at ulp granularity: the money=1954 case in
// TestSelectF128RatioExactlyOne stops at j=3, while the same distribution with
// total=2_000_000_000_000_000 falls through to money. Both match the 128-bit
// big.Float oracle. Boost's double CDF -- evaluated independently per j via
// ibetac rather than accumulated -- can also saturate to 1.0 on its far coarser
// grid, potentially at a different (typically earlier) j. At these near-maximum
// digests the two implementations can therefore return wildly different counts:
// one may stop within a few steps of the binomial tail while the other returns
// the full trial count money. Such inputs are cryptographically unreachable (a
// VRF hash in the top 2^-129), so this is a documented property, not a case
// worth special-casing.
func SelectF128(money uint64, totalMoney uint64, expectedSize uint64, vrfOutput Digest) uint64 {
	ratio := f128FromDigestRatio(vrfOutput)
	return binomialCDFWalkF128(expectedSize, totalMoney, ratio, money)
}

func init() {
	var b int
	var err error
	maxFloatString := fmt.Sprintf("0x%s", strings.Repeat("ff", DigestSize))
	maxFloat, b, err = big.ParseFloat(maxFloatString, 0, precision, big.ToNearestEven)
	if b != 16 || err != nil {
		err = fmt.Errorf("failed to parse big float constant in sortition : %w", err)
		panic(err)
	}
}
