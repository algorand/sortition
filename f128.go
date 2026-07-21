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
	"math"
	"math/big"
	"math/bits"
)

// f128 is a minimal NON-NEGATIVE binary floating-point value with a 128-bit
// mantissa, used for a fast, allocation-free, deterministic binomial CDF (see
// SelectF128). It is a value type -- arithmetic returns new f128s on the stack,
// never the heap -- and uses only integer ops (math/bits), so it is bit-identical
// on every platform (no hardware FP, no FMA, no libm).
//
//	value = (hi<<64 | lo) * 2^exp
//
// The 128-bit mantissa is normalized so bit 127 (the MSB of hi) is set, or the
// value is zero (hi==lo==0). All sortition quantities (p, 1-p, ratio, pmf, cdf,
// factors) are >= 0, so there is no sign bit. Arithmetic ROUNDS TO NEAREST, TIES
// TO EVEN (matching math/big.Float). Round-to-nearest is required, not merely
// nicer: a VRF near the maximum makes the ratio round to exactly 1.0, and only
// round-to-nearest lets the accumulated cdf reach 1.0 (truncation asymptotes just
// below it and the walk runs to `money`).
type f128 struct {
	hi, lo uint64
	exp    int
}

// f128MantBits is the f128 mantissa width; the big.Float setup constants are
// formed at this same precision so they round to f128 exactly.
const f128MantBits = 128

func (a f128) isZero() bool { return a.hi == 0 && a.lo == 0 }

func shl128(hi, lo uint64, n uint) (uint64, uint64) {
	switch {
	case n == 0:
		return hi, lo
	case n < 64:
		return hi<<n | lo>>(64-n), lo << n
	case n < 128:
		return lo << (n - 64), 0
	default:
		return 0, 0
	}
}

func shr128(hi, lo uint64, n uint) (uint64, uint64) {
	switch {
	case n == 0:
		return hi, lo
	case n < 64:
		return hi >> n, lo>>n | hi<<(64-n)
	case n < 128:
		return 0, hi >> (n - 64)
	default:
		return 0, 0
	}
}

// shr128gs shifts hi:lo right by n (1..128), returning the result plus the round
// bit (the most-significant shifted-out bit, at position n-1) and sticky (any
// lower shifted-out bit). Used for round-to-nearest on exponent alignment.
func shr128gs(hi, lo uint64, n uint) (rhi, rlo uint64, round, sticky bool) {
	rhi, rlo = shr128(hi, lo, n)
	switch {
	case n <= 64:
		round = (lo>>(n-1))&1 != 0
		if n >= 2 {
			sticky = lo&((uint64(1)<<(n-1))-1) != 0
		}
	case n < 128:
		m := n - 64
		round = (hi>>(m-1))&1 != 0
		if m >= 2 {
			sticky = hi&((uint64(1)<<(m-1))-1) != 0
		}
		sticky = sticky || lo != 0
	default: // n == 128
		round = hi&(1<<63) != 0
		sticky = (hi&^(uint64(1)<<63) != 0) || lo != 0
	}
	return rhi, rlo, round, sticky
}

// shl192 shifts a 192-bit value (a2:a1:a0) left by n < 192.
func shl192(a2, a1, a0 uint64, n uint) (uint64, uint64, uint64) {
	switch {
	case n == 0:
		return a2, a1, a0
	case n < 64:
		return a2<<n | a1>>(64-n), a1<<n | a0>>(64-n), a0 << n
	case n < 128:
		n -= 64
		if n == 0 {
			return a1, a0, 0
		}
		return a1<<n | a0>>(64-n), a0 << n, 0
	default:
		return a0 << (n - 128), 0, 0
	}
}

// norm128 normalizes a 128-bit mantissa (shifts MSB to bit 127). No bits are
// dropped, so no rounding is needed; used for exact conversions.
func norm128(hi, lo uint64, exp int) f128 {
	if hi == 0 && lo == 0 {
		return f128{}
	}
	var s int
	if hi != 0 {
		s = bits.LeadingZeros64(hi)
	} else {
		s = 64 + bits.LeadingZeros64(lo)
	}
	if s != 0 {
		hi, lo = shl128(hi, lo, uint(s))
		exp -= s
	}
	return f128{hi, lo, exp}
}

// roundNE rounds the normalized 128-bit mantissa hi:lo to nearest, ties to even,
// given the round bit and sticky of the discarded tail, and renormalizes on
// carry-out. hi:lo must already be normalized (bit 127 set).
func roundNE(hi, lo uint64, roundBit, sticky bool, exp int) f128 {
	if roundBit && (sticky || lo&1 != 0) {
		var c uint64
		lo, c = bits.Add64(lo, 1, 0)
		hi, c = bits.Add64(hi, 0, c)
		if c != 0 { // mantissa overflowed to 2^128 -> renormalize to 2^127
			return f128{1 << 63, 0, exp + 1}
		}
	}
	return f128{hi, lo, exp}
}

// norm192 rounds a 192-bit value (r2:r1:r0) * 2^exp to an f128 (top 128 bits,
// round to nearest even using the remaining bits).
func norm192(r2, r1, r0 uint64, exp int) f128 {
	if r2 == 0 && r1 == 0 && r0 == 0 {
		return f128{}
	}
	var lz int
	switch {
	case r2 != 0:
		lz = bits.LeadingZeros64(r2)
	case r1 != 0:
		lz = 64 + bits.LeadingZeros64(r1)
	default:
		lz = 128 + bits.LeadingZeros64(r0)
	}
	s2, s1, s0 := shl192(r2, r1, r0, uint(lz))
	roundBit := s0&(1<<63) != 0
	sticky := s0&^(uint64(1)<<63) != 0
	return roundNE(s2, s1, roundBit, sticky, exp+64-lz)
}

func f128FromUint64(u uint64) f128 {
	if u == 0 {
		return f128{}
	}
	s := bits.LeadingZeros64(u)
	return f128{u << uint(s), 0, -(s + 64)}
}

func f128FromFloat64(f float64) f128 {
	if f <= 0 {
		return f128{}
	}
	b := math.Float64bits(f)
	mant := b & (1<<52 - 1)
	exp := int((b >> 52) & 0x7ff)
	if exp == 0 { // subnormal: value = mant * 2^-1074
		return norm128(0, mant, -1074)
	}
	// normal: significand (mant|2^52) in [2^52,2^53); MSB (bit 52) -> bit 127.
	return f128{(mant | 1<<52) << 11, 0, exp - 1150}
}

var f128bigMask = new(big.Int).SetUint64(math.MaxUint64)

// f128FromBigFloat converts a (one-time, setup) big.Float constant to f128,
// rounding to nearest even at 128 bits (so it matches a 128-bit big.Float).
func f128FromBigFloat(x *big.Float) f128 {
	if x.Sign() <= 0 {
		return f128{}
	}
	r := new(big.Float).SetPrec(128).Set(x) // round to 128-bit mantissa, nearest-even
	m := new(big.Float).SetPrec(256)
	e := r.MantExp(m)    // r = m * 2^e, m in [0.5,1), 128-bit
	m.SetMantExp(m, 128) // m * 2^128 = exact 128-bit integer
	bi, _ := m.Int(nil)
	lo := new(big.Int).And(bi, f128bigMask).Uint64()
	hi := new(big.Int).Rsh(bi, 64).Uint64()
	return norm128(hi, lo, e-128)
}

// mul returns a*b rounded to nearest even.
func (a f128) mul(b f128) f128 {
	if a.isZero() || b.isZero() {
		return f128{}
	}
	hhHi, hhLo := bits.Mul64(a.hi, b.hi)
	hlHi, hlLo := bits.Mul64(a.hi, b.lo)
	lhHi, lhLo := bits.Mul64(a.lo, b.hi)
	llHi, llLo := bits.Mul64(a.lo, b.lo)
	p0 := llLo
	p1, cA := bits.Add64(llHi, hlLo, 0)
	p1, cB := bits.Add64(p1, lhLo, 0)
	cp1 := cA + cB
	p2, cC := bits.Add64(hhLo, hlHi, 0)
	p2, cD := bits.Add64(p2, lhHi, 0)
	p2, cE := bits.Add64(p2, cp1, 0)
	p3 := hhHi + cC + cD + cE
	if p3&(1<<63) != 0 { // product >= 2^255: mantissa p3:p2, tail p1:p0
		roundBit := p1&(1<<63) != 0
		sticky := (p1&^(uint64(1)<<63) != 0) || p0 != 0
		return roundNE(p3, p2, roundBit, sticky, a.exp+b.exp+128)
	}
	// product in [2^254, 2^255): shift left 1 to normalize
	hi := p3<<1 | p2>>63
	lo := p2<<1 | p1>>63
	roundBit := p1&(1<<62) != 0
	sticky := (p1&^(uint64(3)<<62) != 0) || p0 != 0
	return roundNE(hi, lo, roundBit, sticky, a.exp+b.exp+127)
}

// divU returns a/u for a small unsigned integer u (the per-step denominator j),
// rounded to nearest even.
func (a f128) divU(u uint64) f128 {
	if a.isZero() || u == 0 {
		return f128{}
	}
	// (M/u)*2^64 = q2:q1:q0 (integer part q2:q1, next 64 frac bits q0).
	q2, r := bits.Div64(0, a.hi, u)
	q1, r := bits.Div64(r, a.lo, u)
	q0, rem := bits.Div64(r, 0, u)
	if rem != 0 {
		q0 |= 1 // mark sticky for the division remainder
	}
	return norm192(q2, q1, q0, a.exp-64)
}

// add returns a+b (both non-negative), rounded to nearest even.
func (a f128) add(b f128) f128 {
	if a.isZero() {
		return b
	}
	if b.isZero() {
		return a
	}
	if a.exp < b.exp {
		a, b = b, a
	}
	diff := uint(a.exp - b.exp)
	if diff > 128 {
		return a // b is below the round bit
	}
	bhi, blo, round, sticky := shr128gs(b.hi, b.lo, diff)
	slo, c := bits.Add64(a.lo, blo, 0)
	shi, c2 := bits.Add64(a.hi, bhi, c)
	exp := a.exp
	if c2 != 0 { // carry into bit 128: shift right 1, recompute round/sticky
		sticky = sticky || round
		round = slo&1 != 0
		slo = slo>>1 | shi<<63
		shi = shi>>1 | 1<<63
		exp++
	}
	return roundNE(shi, slo, round, sticky, exp)
}

func (a f128) cmp(b f128) int {
	az, bz := a.isZero(), b.isZero()
	switch {
	case az && bz:
		return 0
	case az:
		return -1
	case bz:
		return 1
	case a.exp != b.exp:
		if a.exp < b.exp {
			return -1
		}
		return 1
	case a.hi != b.hi:
		if a.hi < b.hi {
			return -1
		}
		return 1
	case a.lo != b.lo:
		if a.lo < b.lo {
			return -1
		}
		return 1
	}
	return 0
}

// toBigFloat returns the exact value as a big.Float (debug/inspection only).
func (a f128) toBigFloat() *big.Float {
	hiF := new(big.Float).SetPrec(256).SetUint64(a.hi)
	hiF.SetMantExp(hiF, 64)
	m := new(big.Float).SetPrec(256).Add(hiF, new(big.Float).SetPrec(256).SetUint64(a.lo))
	return m.SetMantExp(m, a.exp)
}

// intPow returns base^e by exponentiation by squaring (integer exponent).
func (base f128) intPow(e uint64) f128 {
	result := f128FromUint64(1)
	b := base
	for e > 0 {
		if e&1 == 1 {
			result = result.mul(b)
		}
		e >>= 1
		if e > 0 {
			b = b.mul(b)
		}
	}
	return result
}

// binomialF128 evaluates the CDF of Binomial(money trials, success probability p)
// in software f128 -- the counterpart of boost::math::binomial_distribution<double>.
// cdf(j) returns P(X <= j). Boost computes that as ibetac(j+1, n-j, p); this
// accumulates the IDENTICAL value as the running sum of the binomial PMF:
//
//	pmf(0) = (1-p)^money
//	pmf(j) = pmf(j-1) * (money-j+1)/j * p/(1-p)
//	cdf(j) = pmf(0) + pmf(1) + ... + pmf(j)
//
// cdf MUST be called with j = 0, 1, 2, ... in increasing order (as the walk
// below does); each call advances the running PMF/CDF using only f128 software
// arithmetic, so cdf(j) is bit-reproducible on every platform.
type binomialF128 struct {
	money uint64
	pq    f128   // p/(1-p), the per-step PMF multiplier
	pmf   f128   // pmf(at): the current term
	cum   f128   // cdf(at) = P(X <= at)
	at    uint64 // index that pmf/cum currently hold
}

// newBinomialF128 constructs the Binomial(money, p) CDF evaluator -- the analogue
// of constructing binomial_distribution<double>(n=money, p). The PMF-recurrence
// constants 1-p and p/(1-p) are formed once in big.Float (exact for the float64
// p) and rounded to f128. Returns nil for the degenerate p >= 1 (all probability
// mass at j == money), which the caller handles.
func newBinomialF128(p float64, money uint64) *binomialF128 {
	pb := new(big.Float).SetPrec(f128MantBits).SetFloat64(p)
	qb := new(big.Float).SetPrec(f128MantBits).Sub(new(big.Float).SetPrec(f128MantBits).SetInt64(1), pb)
	if qb.Sign() <= 0 { // p >= 1
		return nil
	}
	pq := f128FromBigFloat(new(big.Float).SetPrec(f128MantBits).Quo(pb, qb))
	pmf0 := f128FromBigFloat(qb).intPow(money) // (1-p)^money
	return &binomialF128{money: money, pq: pq, pmf: pmf0, cum: pmf0, at: 0}
}

func (b *binomialF128) cdf(j uint64) f128 {
	for b.at < j {
		b.at++
		// pmf(at) = pmf(at-1) * (money-at+1)/at * p/(1-p)
		b.pmf = f128FromUint64(b.money - b.at + 1).divU(b.at).mul(b.pq).mul(b.pmf)
		b.cum = b.cum.add(b.pmf)
	}
	return b.cum
}

// binomialCDFWalkF128 is the pure-Go, deterministic counterpart of the C++
// sortition_binomial_cdf_walk in sortition.cpp. It has the SAME signature and the
// SAME walk -- place the two side by side:
//
//	C++  sortition.cpp                            Go  this function
//	------------------------------------------    -------------------------------------------
//	uint64_t sortition_binomial_cdf_walk(         func binomialCDFWalkF128(
//	    double n, double p, double ratio,             n, p, ratio float64, money uint64) uint64 {
//	    uint64_t money) {
//	  binomial_distribution<double> dist(n, p);     dist := newBinomialF128(p, money)
//	  for (uint64_t j = 0; j < money; j++) {        for j := uint64(0); j < money; j++ {
//	    double boundary = cdf(dist, j);               boundary := dist.cdf(j)
//	    if (ratio <= boundary) {                      if rf.cmp(boundary) <= 0 {
//	      return j;                                     return j
//	    }                                           }
//	  }                                           }
//	  return money;                               return money
//	}                                           }
//
// The only difference is HOW the binomial CDF P(X<=j) is obtained: Boost computes
// cdf(dist, j) = ibetac(j+1, n-j, p) afresh each step in hardware double, whereas
// dist.cdf(j) returns the identical value as the running PMF sum in software f128
// (see binomialF128), making the result bit-reproducible on every platform. (n is
// unused: the trial count is the exact uint64 `money`; Boost needs n only as the
// double argument used to construct dist.)
//
// Precondition: money is within the sortition domain -- at most the total online
// microalgo supply (~10^16 < 2^54). The f128 exponent is a plain int; for money
// in that range the exponent of (1-p)^money cannot overflow it (its magnitude is
// bounded by roughly the mean money*p, a committee size). money far beyond the
// supply (>~2^57) is outside the domain -- Boost's Select cannot evaluate it
// either -- and would eventually overflow the int exponent; behavior is undefined
// there.
func binomialCDFWalkF128(n float64, p float64, ratio float64, money uint64) uint64 {
	_ = n
	dist := newBinomialF128(p, money)
	if dist == nil { // p >= 1: cdf(j)==0 for j<money, cdf(money)==1
		if ratio <= 0 {
			return 0
		}
		return money
	}
	rf := f128FromFloat64(ratio)
	for j := uint64(0); j < money; j++ {
		boundary := dist.cdf(j) // = cdf(dist, j) = P(X <= j)
		if rf.cmp(boundary) <= 0 {
			return j
		}
	}
	return money
}
