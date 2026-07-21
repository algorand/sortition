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

// SelectF128 is a pure-Go, cgo-free, deterministic equivalent of Select. Compare
// it line-by-line with Select above: the two function bodies are IDENTICAL except
// the final call -- where Select invokes the C++ sortition_binomial_cdf_walk
// (Boost, hardware double), SelectF128 invokes binomialCDFWalkF128 (software
// f128, see f128.go). Both perform the same binomial-CDF walk; SelectF128's result
// is additionally bit-reproducible on every platform/toolchain (no libm, no FMA,
// no hardware floating point).
//
// CONSENSUS / MIGRATION NOTE. SelectF128 is bit-reproducible but NOT bit-identical
// to the Boost-double Select. They agree on the overwhelming majority of inputs
// (see TestF128AgreesWithCurrent) but differ at knife-edge VRF outputs -- ratios
// within ~2^-53 of a CDF boundary, including a VRF whose ratio rounds to exactly
// 1.0, where the gap can exceed 1. Each difference is a different committee
// selection, so swapping Select -> SelectF128 in production is a protocol-gated,
// network-coordinated consensus change, never a drop-in: a node on SelectF128
// while peers run Boost would fork at those inputs. At such edges SelectF128
// returns the correctly-rounded (128-bit) count; the divergence is exactly the
// libm/double last-bit non-determinism that f128 removes.
func SelectF128(money uint64, totalMoney uint64, expectedSize float64, vrfOutput Digest) uint64 {
	binomialN := float64(money)
	binomialP := expectedSize / float64(totalMoney)

	t := &big.Int{}
	t.SetBytes(vrfOutput[:])

	h := big.Float{}
	h.SetPrec(precision)
	h.SetInt(t)

	ratio := big.Float{}
	cratio, _ := ratio.Quo(&h, maxFloat).Float64()

	return binomialCDFWalkF128(binomialN, binomialP, cratio, money)
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
