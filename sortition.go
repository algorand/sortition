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

// SelectF128MaxWeightFactor bounds the statistically plausible SelectF128
// result. Outside the frozen-tail sliver documented on SelectF128, the walk
// cannot return money-scale values: each result j needs the f128 CDF to
// strictly increase at j, and the CDF freezes once adding the next PMF term
// no longer moves the accumulated sum. The sum sits just below 1, in the
// binade whose ULP spacing is 2^-128 (128-bit mantissa), so terms under
// ~2^-129 -- half that spacing -- are no-ops. The largest reachable
// pre-freeze index is therefore about the binomial quantile where the PMF
// term falls to ~2^-129. For the committee sizes in current
// go-algorand consensus use that quantile is at most ~5.2*expectedSize
// (at expectedSize=20, today's NumProposers; the multiple shrinks toward
// ~1.2 as committees grow -- a future committee smaller than 20 would need
// this factor re-derived). Inside the sliver the result is DEFINED as
// money, the account's entire stake, and no result strictly between the
// freeze index and money is reachable at all, so any threshold in the gap
// separates the two regimes exactly. TestSelectF128WeightGap pins the
// freeze indexes below this factor across current committee sizes and
// stake scales.
//
// Consumers that treat the result as trusted voting power or as a loop
// bound should reject results above SelectF128MaxWeightFactor*expectedSize.
// The factor must exceed ~5.2 to admit every reachable non-plateau result,
// and in go-algorand it must stay below MinBalance/DownCommitteeSize =
// 100_000/6_000 ~= 16.7 so that a plateau result -- at least the 100,000
// microalgo minimum stake -- exceeds the bound for every committee size.
// Every factor in that window rejects the identical, otherwise-unreachable
// set, so 6 -- the smallest sound integer -- is chosen to keep the most
// headroom under MinBalance as committees grow. Taking the tight end leans
// on two commitments, each enforced by a test. First, committee sizes never
// shrink below today's smallest of 20: reducing them would weaken the
// chain's security assumptions independent of sortition, and go-algorand's
// TestSortitionWeightBound asserts the floor. Second, the walk's precision
// stays as it is: guard bits would raise the freeze quantile toward the
// exact-arithmetic ceiling of ~7.5*expectedSize at expectedSize=20, so any
// plateau-narrowing change must re-derive this factor, and
// TestSelectF128WeightGap fails the moment the indexes cross the bound.
// Rejecting a credential changes what validates and is therefore a
// consensus rule: it must ride the same protocol upgrade gate as the switch
// to SelectF128 itself.
const SelectF128MaxWeightFactor = 6

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
// the full trial count money. A uniform VRF output lands in this interval with
// probability about 2^-129 per credential. The event is possible, but the
// consensus threat model treats it as negligible and assumes the registered
// key and unpredictable seed prevent an adversary from targeting it. This is
// therefore a documented statistical edge rather than a case special-cased by
// the implementation.
//
// The same holds in a wider sliver just below 1.0. pmf(0) = (1-p)^money
// amplifies the 2^-129 rounding of 1-p by up to the trial count, and pmf(0)
// scales every PMF term, so the accumulated CDF settles at a plateau that can
// sit as much as ~money*2^-129 below 1 (~2^-78 at 2e15 microalgos of stake,
// and ~2^-76 at the 10^16-microalgo mainnet supply ceiling). A digest ratio
// between that plateau and 1.0 sits above every boundary without rounding to
// 1.0; the walk detects the frozen CDF and immediately returns money, the same
// result the plain walk would reach after up to money no-op iterations.
//
// These outputs are possible under current go-algorand committee and balance
// bounds; TestSelectF128CurrentConsensusFrozenTail pins examples from the base
// account minimum through the mainnet supply ceiling. For an account with
// stake m, the affected interval is approximately m*2^-129, and summing that
// first-order bound over all online accounts gives approximately
// totalMoney*2^-129 per committee selection, independent of how stake is
// split. The consensus rationale for accepting the edge is probabilistic, not
// impossibility: registered keys and an unpredictable seed are assumed to
// prevent targeting the interval. Within it the count is DEFINED as money
// rather than the exact binomial-tail crossing. Computing pmf(0) with guard
// bits could narrow the interval, at the cost of additional consensus-critical
// arithmetic and audit surface.
//
// Because the frozen CDF makes every count strictly between the freeze index
// and money unreachable, a consumer can reject the sliver exactly rather than
// probabilistically: see SelectF128MaxWeightFactor.
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
