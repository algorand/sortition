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
	"encoding/binary"
	"math"
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
// nicer: the ratio is exactly 1.0 for the all-0xff digest (and any digest with
// >= 129 leading one bits rounds there), and only round-to-nearest lets the
// accumulated cdf reach exactly 1.0 -- truncation asymptotes just below it and
// the walk runs to `money` (see TestSelectF128RatioExactlyOne).
type f128 struct {
	hi, lo uint64
	exp    int
}

// f128MantBits is the f128 mantissa width. The big.Float oracle in the test
// forms its constants at this same precision so they round to f128 exactly.
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

// shl256 shifts a 256-bit value (a3:a2:a1:a0) left by n < 256.
func shl256(a3, a2, a1, a0 uint64, n uint) (uint64, uint64, uint64, uint64) {
	words := n / 64
	shift := n % 64
	in := [4]uint64{a3, a2, a1, a0}
	var out [4]uint64
	for i := 0; i < 4; i++ {
		src := i + int(words)
		if src >= len(in) {
			break
		}
		out[i] = in[src] << shift
		if shift != 0 && src+1 < len(in) {
			out[i] |= in[src+1] >> (64 - shift)
		}
	}
	return out[0], out[1], out[2], out[3]
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

// norm192s is norm192 with an incoming sticky bit: extra records that nonzero
// bits were discarded below r0 (used by sub, where aligning the subtrahend can
// shift bits past the 192-bit window). All such bits are far below the result's
// round bit, so they only ever contribute to sticky.
func norm192s(r2, r1, r0 uint64, exp int, extra bool) f128 {
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
	sticky := s0&^(uint64(1)<<63) != 0 || extra
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

// f128FromDigestRatio returns digest/(2^256-1), rounded to nearest-even at
// f128 precision, without reducing the digest to float64 first.
func f128FromDigestRatio(d Digest) f128 {
	w3 := binary.BigEndian.Uint64(d[0:8])
	w2 := binary.BigEndian.Uint64(d[8:16])
	w1 := binary.BigEndian.Uint64(d[16:24])
	w0 := binary.BigEndian.Uint64(d[24:32])

	var leading int
	switch {
	case w3 != 0:
		leading = bits.LeadingZeros64(w3)
	case w2 != 0:
		leading = 64 + bits.LeadingZeros64(w2)
	case w1 != 0:
		leading = 128 + bits.LeadingZeros64(w1)
	case w0 != 0:
		leading = 192 + bits.LeadingZeros64(w0)
	default:
		return f128{}
	}

	n3, n2, n1, n0 := shl256(w3, w2, w1, w0, uint(leading))
	roundBit := n1&(uint64(1)<<63) != 0
	sticky := n1&^(uint64(1)<<63) != 0 || n0 != 0

	// Dividing by 2^256 places the binary point directly after the digest.
	// The real denominator is one smaller, so the exact ratio is slightly
	// larger. That correction is at most one discarded-bit unit and changes
	// rounding only when the 2^256 quotient is exactly halfway.
	if roundBit && !sticky {
		sticky = true
	}
	return roundNE(n3, n2, roundBit, sticky, -128-leading)
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

// divStep is one digit step of Knuth's Algorithm D for a normalized (top bit
// set) 128-bit divisor v1:v0: it divides the 192-bit value uHi:uMid:uLo by
// v1:v0, where the running-remainder prefix uHi:uMid is already < v1:v0, and
// returns the 64-bit quotient digit q and the new 128-bit remainder rHi:rLo.
func divStep(uHi, uMid, uLo, v1, v0 uint64) (q, rHi, rLo uint64) {
	// qhat = min((uHi:uMid)/v1, 2^64-1); Div64 requires uHi < v1, so cap when
	// uHi == v1 (the running remainder guarantees uHi never exceeds v1).
	var qhat, rhat uint64
	refine := true
	if uHi >= v1 {
		qhat = ^uint64(0)
		var c uint64
		rhat, c = bits.Add64(uMid, v1, 0) // rhat = (v1:uMid) - qhat*v1 = uMid + v1
		if c != 0 {
			refine = false // rhat >= 2^64: the refine test is already false
		}
	} else {
		qhat, rhat = bits.Div64(uHi, uMid, v1)
	}
	// Lower qhat (over-estimated by at most 2) until qhat*v0 <= rhat:uLo.
	for refine {
		hi, lo := bits.Mul64(qhat, v0)
		if hi > rhat || (hi == rhat && lo > uLo) {
			qhat--
			var c uint64
			rhat, c = bits.Add64(rhat, v1, 0)
			if c != 0 {
				break
			}
			continue
		}
		break
	}
	// u - qhat*(v1:v0), a 192-bit subtraction.
	p1hi, p1lo := bits.Mul64(qhat, v1)
	p0hi, p0lo := bits.Mul64(qhat, v0)
	prodMid, c := bits.Add64(p1lo, p0hi, 0)
	prodHi := p1hi + c
	sLo, br := bits.Sub64(uLo, p0lo, 0)
	sMid, br := bits.Sub64(uMid, prodMid, br)
	_, br = bits.Sub64(uHi, prodHi, br)
	q = qhat
	if br != 0 { // qhat was 1 too large: add the divisor back
		q--
		sLo, c = bits.Add64(sLo, v0, 0)
		sMid, _ = bits.Add64(sMid, v1, c)
	}
	return q, sMid, sLo
}

// div returns a/b rounded to nearest even, for NON-NEGATIVE operands (b != 0).
// It divides the 256-bit a.hi:a.lo:0:0 by the normalized 128-bit b.hi:b.lo via
// Knuth long division. Since both mantissas are in [2^127,2^128), the ratio is
// in (0.5,2), so the 129-bit integer quotient Q is either already normalized
// (Q < 2^128) or one bit wide (Q >= 2^128); the division remainder supplies the
// bits below Q for round-to-nearest-even. Unlike divU (small integer divisor),
// this handles a full f128 divisor; it is used once per setup, not in the walk.
func (a f128) div(b f128) f128 {
	if a.isZero() || b.isZero() {
		return f128{}
	}
	v1, v0 := b.hi, b.lo
	// Long-divide [a.hi, a.lo, 0, 0] by v1:v0, most-significant limb first.
	var remHi, remLo uint64
	var q1, q0 uint64 // the two low quotient digits; the top two are 0 and {0,1}
	var q2 uint64
	digit, remHi, remLo := divStep(remHi, remLo, a.hi, v1, v0) // = 0
	_ = digit
	q2, remHi, remLo = divStep(remHi, remLo, a.lo, v1, v0) // in {0,1}
	q1, remHi, remLo = divStep(remHi, remLo, 0, v1, v0)
	q0, remHi, remLo = divStep(remHi, remLo, 0, v1, v0)

	expq := a.exp - b.exp - 128
	if q2 != 0 { // Q in [2^128,2^129): mantissa = Q>>1, dropped low bit is round
		mantHi := q2<<63 | q1>>1
		mantLo := q1<<63 | q0>>1
		round := q0&1 != 0
		sticky := remHi != 0 || remLo != 0
		return roundNE(mantHi, mantLo, round, sticky, expq+1)
	}
	// Q in [2^127,2^128): already normalized; round/sticky come from rem/b, i.e.
	// round = (2*rem >= b), sticky = the leftover after that comparison.
	dblLo := remLo << 1
	dblHi := remHi<<1 | remLo>>63
	carry := remHi >> 63
	var round, sticky bool
	if carry != 0 || dblHi > v1 || (dblHi == v1 && dblLo >= v0) {
		round = true
		sLo, br := bits.Sub64(dblLo, v0, 0)
		sHi, _ := bits.Sub64(dblHi, v1, br)
		sticky = sHi != 0 || sLo != 0
	} else {
		sticky = remHi != 0 || remLo != 0
	}
	return roundNE(q1, q0, round, sticky, expq)
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

// sub returns a-b, rounded to nearest even, for NON-NEGATIVE operands with
// a >= b (f128 is unsigned; a < b returns zero). The subtraction is performed
// exactly in a 192-bit field -- a occupies a.hi:a.lo:0, giving 64 guard bits
// below a's ulp -- so it is correct even under catastrophic cancellation
// (a ~= b). b is aligned into that field by an arithmetic right shift; any bits
// pushed below bit 0 (only possible when b << a, i.e. no cancellation) are
// folded into sticky. When such bits exist, one field-ulp is borrowed first so
// the residual (field-ulp - discarded) rounds correctly as pure sticky.
func (a f128) sub(b f128) f128 {
	if b.isZero() {
		return a
	}
	if a.cmp(b) <= 0 {
		return f128{}
	}
	diff := uint(a.exp - b.exp)
	var b2, b1, b0 uint64
	var sticky bool
	switch {
	case diff == 0:
		b2, b1, b0 = b.hi, b.lo, 0
	case diff < 64:
		b2 = b.hi >> diff
		b1 = b.hi<<(64-diff) | b.lo>>diff
		b0 = b.lo << (64 - diff)
	case diff == 64:
		b2, b1, b0 = 0, b.hi, b.lo
	case diff < 128:
		s := diff - 64
		b1 = b.hi >> s
		b0 = b.hi<<(64-s) | b.lo>>s
		sticky = b.lo<<(64-s) != 0
	case diff == 128:
		b0 = b.hi
		sticky = b.lo != 0
	case diff < 192:
		s := diff - 128
		b0 = b.hi >> s
		sticky = b.hi<<(64-s) != 0 || b.lo != 0
	default:
		sticky = true // b nonzero, entirely below the field
	}
	if sticky { // borrow one field-ulp so the discarded remainder is pure sticky
		var brw uint64
		b0, brw = bits.Add64(b0, 1, 0)
		b1, brw = bits.Add64(b1, 0, brw)
		b2 += brw
	}
	// a >= b, so the 192-bit subtraction A - B never borrows out of the top.
	r0, brw := bits.Sub64(0, b0, 0)
	r1, brw := bits.Sub64(a.lo, b1, brw)
	r2, _ := bits.Sub64(a.hi, b2, brw)
	return norm192s(r2, r1, r0, a.exp-64, sticky)
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
// constants 1-p and p/(1-p) are formed once, entirely in f128 (no big.Float, no
// heap): both are round-to-nearest-even f128 operations, bit-identical to the
// 128-bit big.Float oracle (1-p is additionally exact for any p >= ~2^-76,
// which covers every realistic sortition probability). Returns nil for the
// degenerate p >= 1 (all probability mass at j == money), which the caller
// handles.
func newBinomialF128(p float64, money uint64) *binomialF128 {
	pf := f128FromFloat64(p)
	if pf.cmp(f128FromUint64(1)) >= 0 { // p >= 1
		return nil
	}
	qf := f128FromUint64(1).sub(pf) // 1-p
	pq := pf.div(qf)                // p/(1-p)
	pmf0 := qf.intPow(money)        // (1-p)^money
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
// sortition_binomial_cdf_walk in sortition.cpp. It performs the same boundary
// walk, using an f128 ratio and f128 CDF values:
//
//	C++  sortition.cpp                            Go  this function
//	------------------------------------------    -------------------------------------------
//	uint64_t sortition_binomial_cdf_walk(         func binomialCDFWalkF128(
//	    double n, double p, double ratio,             p float64, ratio f128, money uint64) uint64 {
//	    uint64_t money) {
//	  binomial_distribution<double> dist(n, p);     dist := newBinomialF128(p, money)
//	  for (uint64_t j = 0; j < money; j++) {        for j := uint64(0); j < money; j++ {
//	    double boundary = cdf(dist, j);               boundary := dist.cdf(j)
//	    if (ratio <= boundary) {                      if ratio.cmp(boundary) <= 0 {
//	      return j;                                     return j
//	    }                                           }
//	  }                                           }
//	  return money;                               return money
//	}                                           }
//
// Boost computes cdf(dist, j) = ibetac(j+1, n-j, p) afresh each step in hardware
// double, whereas dist.cdf(j) returns the same mathematical value as a running
// PMF sum in software f128 (see binomialF128). The f128 path also receives the
// digest ratio directly at f128 precision.
//
// Precondition: money is within the sortition domain -- at most the total online
// microalgo supply (~10^16 < 2^54). The f128 exponent is a plain int; for money
// in that range the exponent of (1-p)^money cannot overflow it (its magnitude is
// bounded by roughly the mean money*p, a committee size). money far beyond the
// supply (>~2^57) is outside the domain -- Boost's Select cannot evaluate it
// either -- and would eventually overflow the int exponent; behavior is undefined
// there.
func binomialCDFWalkF128(p float64, ratio f128, money uint64) uint64 {
	dist := newBinomialF128(p, money)
	if dist == nil { // p >= 1: cdf(j)==0 for j<money, cdf(money)==1
		if ratio.isZero() {
			return 0
		}
		return money
	}
	for j := uint64(0); j < money; j++ {
		boundary := dist.cdf(j) // = cdf(dist, j) = P(X <= j)
		if ratio.cmp(boundary) <= 0 {
			return j
		}
	}
	return money
}
