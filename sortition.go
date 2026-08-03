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

// SelectF128MaxMoney bounds SelectF128's money argument. The f128 stored
// exponent of a value v is about log2(v) - 127 (the mantissa is normalized to
// [2^127, 2^128)), and the smallest representable 1-p exceeds 2^-64, so
// pmf(0) = (1-p)^money carries a stored exponent no lower than
// -64*money - 128, and the worst intermediate -- the a.exp+b.exp sum inside a
// multiply, whose value parts total at most money -- stays above
// -64*money - 256. With money < 2^56 every such quantity is bounded by
// ~2^62+2^8 in magnitude, comfortably inside int64. (A 2^57 bound is NOT
// safe: money = 2^57-1 with 1-p = 1/(2^64-1) needs a stored exponent near
// -2^63-63 and wraps.)
//
// The bound is ~7x (about 2.8 bits) above Algorand's 10^16 microalgo supply.
// Behavior above it is undefined, and SelectF128 does not check it at
// runtime. Consumers should assert their supply invariants against this
// constant in a test, so a future economics change fails loudly there instead
// of silently misrounding consensus.
const SelectF128MaxMoney = uint64(1) << 56

// SelectF128 is a deterministic sortition function. It evaluates both the VRF
// ratio and binomial CDF at f128 precision using software integer arithmetic, so
// its result is bit-reproducible across platforms. money must be below
// SelectF128MaxMoney.
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
// FROZEN-TAIL POLICY. Rounding q=1-p once and then raising it to money can
// scale every PMF term by a common error of about money*2^-129. When that
// error is downward, the accumulated f128 CDF can settle at a plateau below
// 1. A digest ratio above the plateau would never cross another represented
// boundary, and a literal walk would eventually fall through and return
// money after as many as money no-op iterations.
//
// Once an addition leaves the CDF unchanged while the PMF is strictly
// shrinking, every later term is no larger and every later CDF addition is
// also a no-op. SelectF128 therefore promotes that first frozen boundary to 1
// and returns its index. This assigns the unresolved tail to one finite result
// instead of treating the account's entire stake as its selection weight. It
// is a deliberate approximation: affected digests no longer distinguish the
// true binomial quantiles beyond the precision horizon, and the result differs
// from the literal fall-through in the C++ reference loop.
//
// The rule also applies when the digest ratio rounded to exactly 1. The top
// ~2^-129 of digest space does so at f128 precision; only the all-0xff digest
// is exactly 1 under the digest/(2^256-1) mapping. If the accumulated CDF
// rounds to 1 before freezing, the ordinary inclusive boundary wins. If it
// freezes below 1, the promoted freeze index wins. At small money the CDF can
// instead remain live and below 1 through every j < money, in which case the
// ordinary loop legitimately falls through to money, the exact inverse-CDF
// count for ratio 1. TestSelectF128RatioExactlyOne pins all three trajectories.
//
// The frozen sliver is approximately money*2^-129 wide. Summed over online
// accounts, its first-order rate is approximately totalMoney*2^-129 per
// committee selection, independent of how stake is split. Under the protocol
// invariant money <= totalMoney, the binomial mean is at most expectedSize;
// current-parameter regression tests pin the promoted results at
// committee-scale indexes from the base account minimum through the mainnet
// supply ceiling. Callers that violate money <= totalMoney can have a mean and
// selection result larger than expectedSize; this tail policy is not a general
// output cap. Computing pmf(0) with guard bits would narrow the frozen sliver
// and move the promoted indexes; changing that precision policy is therefore
// also a consensus change.
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
