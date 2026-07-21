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
	"math/big"
	"math/rand"
	"testing"
)

var integerOracleOne = big.NewInt(1)

// floorLog2Rat returns floor(log2(num/den)) for positive integers. It uses
// integer comparison only; in particular, the exact-integer oracle below does
// not ask big.Float to decide normalization or rounding.
func floorLog2Rat(num, den *big.Int) int {
	e := num.BitLen() - den.BitLen()
	if e >= 0 {
		if num.Cmp(new(big.Int).Lsh(new(big.Int).Set(den), uint(e))) < 0 {
			e--
		}
	} else if new(big.Int).Lsh(new(big.Int).Set(num), uint(-e)).Cmp(den) < 0 {
		e--
	}
	return e
}

// roundExactRatToF128 rounds (num/den)*2^exp2 to f128 using only big.Int
// quotient/remainder arithmetic. num must be non-negative and den positive.
func roundExactRatToF128(num, den *big.Int, exp2 int64) f128 {
	if num.Sign() == 0 {
		return f128{}
	}
	if num.Sign() < 0 || den.Sign() <= 0 {
		panic("roundExactRatToF128 requires num >= 0 and den > 0")
	}

	e := floorLog2Rat(num, den)
	n, d := new(big.Int).Set(num), new(big.Int).Set(den)
	if shift := 127 - e; shift >= 0 {
		n.Lsh(n, uint(shift))
	} else {
		d.Lsh(d, uint(-shift))
	}

	mant, rem := new(big.Int), new(big.Int)
	mant.QuoRem(n, d, rem)
	twiceRem := new(big.Int).Lsh(new(big.Int).Set(rem), 1)
	if cmp := twiceRem.Cmp(d); cmp > 0 || (cmp == 0 && mant.Bit(0) != 0) {
		mant.Add(mant, integerOracleOne)
	}

	outExp := exp2 + int64(e) - 127
	if mant.BitLen() == 129 { // rounded 2^128: renormalize to 2^127
		mant.Rsh(mant, 1)
		outExp++
	}
	if mant.BitLen() != 128 {
		panic("integer oracle produced a non-normalized mantissa")
	}
	lo := mant.Uint64()
	hi := new(big.Int).Rsh(new(big.Int).Set(mant), 64).Uint64()
	return f128{hi: hi, lo: lo, exp: outExp}
}

func f128MantissaInt(x f128) *big.Int {
	m := new(big.Int).SetUint64(x.hi)
	m.Lsh(m, 64)
	return m.Add(m, new(big.Int).SetUint64(x.lo))
}

func exactMulF128(a, b f128) f128 {
	if a.isZero() || b.isZero() {
		return f128{}
	}
	n := new(big.Int).Mul(f128MantissaInt(a), f128MantissaInt(b))
	return roundExactRatToF128(n, integerOracleOne, a.exp+b.exp)
}

func exactAddF128(a, b f128) f128 {
	if a.isZero() {
		return b
	}
	if b.isZero() {
		return a
	}
	baseExp := a.exp
	if b.exp < baseExp {
		baseExp = b.exp
	}
	an := f128MantissaInt(a)
	bn := f128MantissaInt(b)
	an.Lsh(an, uint(a.exp-baseExp))
	bn.Lsh(bn, uint(b.exp-baseExp))
	return roundExactRatToF128(new(big.Int).Add(an, bn), integerOracleOne, baseExp)
}

func exactDivF128(a, b f128) f128 {
	if a.isZero() || b.isZero() {
		return f128{}
	}
	return roundExactRatToF128(f128MantissaInt(a), f128MantissaInt(b), a.exp-b.exp)
}

func exactDivUF128(a f128, u uint64) f128 {
	if a.isZero() || u == 0 {
		return f128{}
	}
	return roundExactRatToF128(f128MantissaInt(a), new(big.Int).SetUint64(u), a.exp)
}

func assertCanonicalF128(t *testing.T, name string, x f128) {
	t.Helper()
	if x.isZero() {
		if x.exp != 0 {
			t.Fatalf("%s: zero has nonzero exponent: %+v", name, x)
		}
		return
	}
	if x.hi&(uint64(1)<<63) == 0 {
		t.Fatalf("%s: mantissa is not normalized: %+v", name, x)
	}
}

func assertExactF128(t *testing.T, name string, got, want f128) {
	t.Helper()
	assertCanonicalF128(t, name, got)
	if got != want {
		t.Fatalf("%s: got %+v, exact-integer RNE oracle wants %+v", name, got, want)
	}
}

// TestF128OpsExactIntegerOracle validates the public arithmetic contract with
// an oracle structurally independent of math/big.Float. Exact values are
// represented as integer ratios; normalization and ties-to-even are decided
// from quotient and remainder.
func TestF128OpsExactIntegerOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	gaps := [...]int64{0, 1, -1, 63, -63, 64, -64, 65, -65, 127, -127, 128, -128, 129, -129, 511, -511, 2000, -2000}
	for i := 0; i < 100_000; i++ {
		aexp := int64(rng.Intn(4001) - 2000)
		bexp := aexp - gaps[i%len(gaps)]
		a := f128{hi: rng.Uint64() | 1<<63, lo: rng.Uint64(), exp: aexp}
		b := f128{hi: rng.Uint64() | 1<<63, lo: rng.Uint64(), exp: bexp}
		if i%997 == 0 {
			a = f128{}
		}
		if i%991 == 0 {
			b = f128{}
		}

		assertExactF128(t, "mul", a.mul(b), exactMulF128(a, b))
		assertExactF128(t, "add", a.add(b), exactAddF128(a, b))
		assertExactF128(t, "div", a.div(b), exactDivF128(a, b))

		var u uint64
		switch i % 8 {
		case 0:
			u = 0
		case 1:
			u = 1
		case 2:
			u = SelectF128MaxMoney - 1
		case 3:
			u = 1<<62 | rng.Uint64()&((1<<62)-1)
		case 4:
			u = 1<<63 | rng.Uint64()&((1<<63)-1)
		case 5:
			u = ^uint64(0)
		default:
			u = rng.Uint64()
		}
		assertExactF128(t, "divU", a.divU(u), exactDivUF128(a, u))
	}
}

// TestF128ConversionsExactIntegerOracle independently checks the two exact
// input conversions. The existing big.Float matrix remains useful; this test
// makes its rounding oracle non-circular.
func TestF128ConversionsExactIntegerOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	den := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), integerOracleOne)
	for i := 0; i < 20_000; i++ {
		u := rng.Uint64()
		assertExactF128(t, "fromUint64", f128FromUint64(u),
			roundExactRatToF128(new(big.Int).SetUint64(u), integerOracleOne, 0))

		var d Digest
		_, _ = rng.Read(d[:])
		if i == 0 {
			d = Digest{}
		} else if i == 1 {
			for j := range d {
				d[j] = 0xff
			}
		}
		assertExactF128(t, "digestRatio", f128FromDigestRatio(d),
			roundExactRatToF128(new(big.Int).SetBytes(d[:]), den, 0))
	}
}

func words128(x *big.Int) (uint64, uint64) {
	lo := x.Uint64()
	hi := new(big.Int).Rsh(new(big.Int).Set(x), 64).Uint64()
	return hi, lo
}

// TestDivStepExactCertificate checks the long-division digit directly rather
// than only through the final rounded f128 quotient. Under divStep's prefix
// precondition, the returned digit and remainder must be the Euclidean
// quotient and remainder of the 192-by-128-bit division.
func TestDivStepExactCertificate(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	check := func(v1, v0 uint64, prefix *big.Int, uLo uint64) {
		t.Helper()
		v := new(big.Int).SetUint64(v1)
		v.Lsh(v, 64).Add(v, new(big.Int).SetUint64(v0))
		if prefix.Sign() < 0 || prefix.Cmp(v) >= 0 {
			t.Fatal("invalid divStep test prefix")
		}
		uHi, uMid := words128(prefix)
		q, rHi, rLo := divStep(uHi, uMid, uLo, v1, v0)

		u := new(big.Int).Lsh(new(big.Int).Set(prefix), 64)
		u.Add(u, new(big.Int).SetUint64(uLo))
		wantQ, wantR := new(big.Int), new(big.Int)
		wantQ.QuoRem(u, v, wantR)
		if wantQ.BitLen() > 64 || q != wantQ.Uint64() {
			t.Fatalf("divStep quotient: got %#x, want %s", q, wantQ.Text(16))
		}
		wantRHi, wantRLo := words128(wantR)
		if rHi != wantRHi || rLo != wantRLo {
			t.Fatalf("divStep remainder: got %#x:%#x, want %#x:%#x", rHi, rLo, wantRHi, wantRLo)
		}

		// Check the certificate separately from comparison with QuoRem.
		gotR := new(big.Int).SetUint64(rHi)
		gotR.Lsh(gotR, 64).Add(gotR, new(big.Int).SetUint64(rLo))
		reconstructed := new(big.Int).Mul(new(big.Int).SetUint64(q), v)
		reconstructed.Add(reconstructed, gotR)
		if reconstructed.Cmp(u) != 0 || gotR.Sign() < 0 || gotR.Cmp(v) >= 0 {
			t.Fatalf("invalid division certificate: U=%x q=%x V=%x R=%x", u, q, v, gotR)
		}
	}

	for i := 0; i < 200_000; i++ {
		v1 := rng.Uint64() | 1<<63
		v0 := rng.Uint64()
		if i%16 == 0 {
			v0 |= 1 // V-1 then has uHi == v1 and exercises the qhat cap
		}
		v := new(big.Int).SetUint64(v1)
		v.Lsh(v, 64).Add(v, new(big.Int).SetUint64(v0))
		prefix := new(big.Int).SetUint64(rng.Uint64())
		prefix.Lsh(prefix, 64).Add(prefix, new(big.Int).SetUint64(rng.Uint64()))
		prefix.Mod(prefix, v)
		if i%16 == 0 {
			prefix.Sub(v, integerOracleOne)
		}
		check(v1, v0, prefix, rng.Uint64())
	}

	// qhat cap with rhat carry: uHi == v1, uMid+v1 overflows, and uMid < v0.
	v1, v0 := uint64(1)<<63, ^uint64(0)
	prefix := new(big.Int).SetUint64(v1)
	prefix.Lsh(prefix, 64).Add(prefix, new(big.Int).SetUint64(1<<63))
	check(v1, v0, prefix, 0)
}
