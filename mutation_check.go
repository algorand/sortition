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

//go:build ignore

// mutation_check applies curated single-site mutants to f128.go and its test
// oracles, then verifies the fast suite kills each one. Every mutant models a
// plausible implementation or harness bug (flipped rounding masks, dropped
// carries, wrong initialization, corrupt recurrence/boundary logic); an
// UNEXPECTED survivor is a test gap and must be answered with a new test.
// Mutants marked equivalent encode a proof of why the mutation cannot change
// behavior; they are expected to survive and document that analysis.
//
// Run from the repo root:
//
//	go run mutation_check.go
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

type mutant struct {
	name       string
	target     string // defaults to f128.go; test-oracle mutants name their file
	old, new   string
	equivalent string // non-empty: why this mutant is expected to survive
}

var mutants = []mutant{
	{name: "mul-roundbit-hi", old: "roundBit := p1&(1<<63) != 0", new: "roundBit := p1&(1<<62) != 0"},
	{name: "mul-sticky-hi-drop-p0", old: "sticky := (p1&^(uint64(1)<<63) != 0) || p0 != 0", new: "sticky := p1&^(uint64(1)<<63) != 0"},
	{name: "mul-carry-or", old: "cp1 := cA + cB", new: "cp1 := cA | cB"},
	{name: "mul-exp-hi", old: "roundNE(p3, p2, roundBit, sticky, a.exp+b.exp+128)", new: "roundNE(p3, p2, roundBit, sticky, a.exp+b.exp+129)"},
	{name: "mul-roundbit-lo", old: "roundBit := p1&(1<<62) != 0", new: "roundBit := p1&(1<<61) != 0"},
	{
		name: "mul-sticky-lo-mask", old: "sticky := (p1&^(uint64(3)<<62) != 0) || p0 != 0", new: "sticky := (p1&^(uint64(1)<<62) != 0) || p0 != 0",
		equivalent: "p1 bit 63 is also the mantissa's low bit in this branch, so whenever the leaked sticky could matter (round bit set) the mantissa is odd and ties round up anyway; clearing bit 63 is necessary only cosmetically",
	},
	{name: "roundNE-invert-tie", old: "if roundBit && (sticky || lo&1 != 0) {", new: "if roundBit && (sticky || lo&1 == 0) {"},
	{name: "roundNE-carry-exp", old: "return f128{1 << 63, 0, exp + 1}", new: "return f128{1 << 63, 0, exp}"},
	{name: "gs-roundbit-pos", old: "round = (lo>>(n-1))&1 != 0", new: "round = (lo>>n)&1 != 0"},
	{name: "gs-sticky-guard", old: "if n >= 2 {", new: "if n >= 3 {"},
	{name: "gs-128-drop-lo", old: "sticky = (hi&^(uint64(1)<<63) != 0) || lo != 0", new: "sticky = hi&^(uint64(1)<<63) != 0"},
	{name: "add-gap-boundary", old: "if a.exp-b.exp > 128 {", new: "if a.exp-b.exp > 127 {"},
	{name: "add-carry-drop-sticky", old: "sticky = sticky || round", new: "sticky = round"},
	{name: "add-carry-drop-round", old: "round = slo&1 != 0", new: "round = false"},
	{name: "divU-drop-rem-sticky", old: "sticky := o1&^(uint64(1)<<63) != 0 || o0 != 0 || rem != 0", new: "sticky := o1&^(uint64(1)<<63) != 0 || o0 != 0 || rem > rem"},
	{name: "divU-roundbit", old: "round := o1&(uint64(1)<<63) != 0", new: "round := o1&(uint64(1)<<62) != 0"},
	{name: "divU-exp", old: "roundNE(o3, o2, round, sticky, a.exp-int64(lz))", new: "roundNE(o3, o2, round, sticky, a.exp-int64(lz)-1)"},
	{name: "ratio-drop-halfway", old: "if roundBit && !sticky {", new: "if roundBit && sticky {"},
	{name: "ratio-exp", old: "roundNE(n3, n2, roundBit, sticky, -128-int64(leading))", new: "roundNE(n3, n2, roundBit, sticky, -129-int64(leading))"},
	{
		name: "ratio-sticky-drop-n0", old: "sticky := n1&^(uint64(1)<<63) != 0 || n0 != 0", new: "sticky := n1&^(uint64(1)<<63) != 0 || n0 > n0",
		equivalent: "the halfway denominator-correction turns any roundBit-without-sticky into sticky, so n0's contribution is recreated exactly when it could matter, and sticky is irrelevant when roundBit is clear",
	},
	{name: "digest-little-endian", old: "w3 := binary.BigEndian.Uint64(d[0:8])", new: "w3 := binary.LittleEndian.Uint64(d[0:8])"},
	{name: "intpow-invert-bit", old: "if e&1 == 1 {", new: "if e&1 == 0 {"},
	{name: "intpow-wrong-square", old: "b = b.mul(b)", new: "b = b.mul(base)"},
	{name: "distribution-degenerate-strict", old: "if expectedSize >= totalMoney { // p >= 1", new: "if expectedSize > totalMoney { // p >= 1"},
	{
		name: "distribution-q-numerator",
		old:  "qf := f128FromUint64(totalMoney - expectedSize).div(f128FromUint64(totalMoney))   // 1-p",
		new:  "qf := f128FromUint64(expectedSize).div(f128FromUint64(totalMoney))                // 1-p",
	},
	{
		name: "distribution-pq-denominator",
		old:  "pq := f128FromUint64(expectedSize).div(f128FromUint64(totalMoney - expectedSize)) // p/(1-p)",
		new:  "pq := f128FromUint64(expectedSize).div(f128FromUint64(totalMoney))                // p/(1-p)",
	},
	{name: "distribution-pmf0-power", old: "pmf0 := qf.intPow(money)", new: "pmf0 := qf.intPow(money + 1)"},
	{name: "distribution-cum-zero", old: "pmf: pmf0, cum: pmf0, at: 0", new: "pmf: pmf0, cum: f128{}, at: 0"},
	{name: "cdf-skip-index", old: "b.at++", new: "b.at += 2"},
	{name: "cdf-recurrence-off-by-one", old: "f128FromUint64(b.money - b.at + 1)", new: "f128FromUint64(b.money - b.at)"},
	{
		name: "walk-start-at-one",
		old:  "for j := uint64(0); j < money; j++ {\n\t\tboundary := dist.cdf(j)",
		new:  "for j := uint64(1); j < money; j++ {\n\t\tboundary := dist.cdf(j)",
	},
	{name: "walk-strict-compare", old: "P(X <= j)\n\t\tif ratio.cmp(boundary) <= 0 {", new: "P(X <= j)\n\t\tif ratio.cmp(boundary) < 0 {"},
	{
		name: "frozen-branch-return-money",
		old:  "monotonicity in the digest.\n\t\t\treturn j",
		new:  "monotonicity in the digest.\n\t\t\treturn money",
	},
	{
		name: "frozen-branch-off-by-one",
		old:  "monotonicity in the digest.\n\t\t\treturn j",
		new:  "monotonicity in the digest.\n\t\t\treturn j + 1",
	},
	{name: "freeze-fire-on-change", old: "if b.cum.cmp(cumPrev) == 0 && b.pmf.cmp(pmfPrev) < 0 {", new: "if b.cum.cmp(cumPrev) != 0 && b.pmf.cmp(pmfPrev) < 0 {"},
	{
		name: "freeze-nonstrict-pmf", old: "b.pmf.cmp(pmfPrev) < 0 {", new: "b.pmf.cmp(pmfPrev) <= 0 {",
		equivalent: "the mutant widens the trigger to equal-pmf no-ops, but no such step exists, so the observable freeze index is unchanged: pmf(k) == pmf(k-1) needs the rounded step multiplier within ~an ulp of 1, which the strictly decreasing factor satisfies only adjacent to the PMF mode, where the term is within ulps of the distribution's maximum and cum <= (k+1)*pmf, giving pmf/cum >= 1/(k+1) > 2^-57 for money < 2^56 -- far above the ~2^-129 no-op threshold, so the add always moves cum there",
	},
	{name: "divstep-cap-boundary", old: "if uHi >= v1 {", new: "if uHi > v1 {"},
	{
		name: "divstep-refine-nonstrict", old: "if hi > rhat || (hi == rhat && lo > uLo) {", new: "if hi > rhat || (hi == rhat && lo >= uLo) {",
		equivalent: "the extra decrement fires only on exact quotient digits, whose zero-padded tail makes the remaining walk produce capped all-ones digits ending with remainder exactly V, and the final round-up carries the representation back to Q; the tie-sensitive case would need an odd exact 129-bit quotient, impossible since the dividend's 2^128 factor cannot come from M_b",
	},
	{name: "div-mantissa-shift", old: "mantHi := q2<<63 | q1>>1", new: "mantHi := q2<<62 | q1>>1"},
	{name: "div-roundbit-q2", old: "round := q0&1 != 0", new: "round := false"},
	{
		name: "div-tie-nonstrict", old: "if carry != 0 || dblHi > v1 || (dblHi == v1 && dblLo >= v0) {", new: "if carry != 0 || dblHi > v1 || (dblHi == v1 && dblLo > v0) {",
		equivalent: "exact ties are impossible in div: 2*rem == M_b requires 2^129 to divide M_b*(2q+1), and M_b < 2^128 with 2q+1 odd cannot supply it, so the >= vs > distinction is unreachable",
	},
	{name: "norm128-direction", old: "exp -= int64(s)", new: "exp += int64(s)"},
	{name: "fromuint64-exp", old: "return f128{u << uint(s), 0, -int64(s) - 64}", new: "return f128{u << uint(s), 0, -int64(s) - 63}"},
	{name: "shl256-fill-shift", old: "out[i] |= in[src+1] >> (64 - shift)", new: "out[i] |= in[src+1] >> (63 - shift)"},
	{name: "cmp-lo-invert", old: "if a.lo < b.lo {", new: "if a.lo > b.lo {"},

	// Test the tests: corrupt each major in-process oracle independently. The
	// exact-integer, exact-CDF, high-precision, Arb, metamorphic, and production
	// layers should make these harness defects observable rather than allowing
	// all references to agree with one another accidentally.
	{
		name:   "big-oracle-pmf0-power",
		target: "f128_test.go",
		old:    "pmf := bigIntPow(q, money, prec) // (1-p)^money",
		new:    "pmf := bigIntPow(q, money+1, prec) // (1-p)^money",
	},
	{
		name:   "big-oracle-recurrence-off-by-one",
		target: "f128_test.go",
		old:    "new(big.Float).SetPrec(prec).SetUint64(money-j+1),",
		new:    "new(big.Float).SetPrec(prec).SetUint64(money-j),",
	},
	{
		name:   "big-oracle-ignore-freeze",
		target: "f128_test.go",
		old:    "if cdf.Cmp(cdfPrev) == 0 && pmf.Cmp(pmfPrev) < 0 {\n\t\t\treturn j\n\t\t}",
		new:    "if cdf.Cmp(cdfPrev) == 0 && pmf.Cmp(pmfPrev) < 0 {\n\t\t\treturn money\n\t\t}",
	},
	{
		name:   "digest-oracle-denominator",
		target: "f128_test.go",
		old:    "denominator.Sub(denominator, big.NewInt(1))",
		new:    "denominator.Sub(denominator, big.NewInt(2))",
	},
	{
		name:   "highprec-oracle-recurrence-off-by-one",
		target: "f128_exact_test.go",
		old:    "new(big.Float).SetPrec(prec).SetUint64(money-j+1),",
		new:    "new(big.Float).SetPrec(prec).SetUint64(money-j),",
	},
	{
		name:   "frozen-sliver-invert",
		target: "f128_exact_test.go",
		old:    "return gap.Cmp(bound) < 0",
		new:    "return gap.Cmp(bound) > 0",
	},
	{
		name:   "exact-cdf-binomial-off-by-one",
		target: "f128_exact_test.go",
		old:    "term := new(big.Int).Binomial(int64(money), int64(i))",
		new:    "term := new(big.Int).Binomial(int64(money-1), int64(i))",
	},
	{
		name:   "exact-select-strict-boundary",
		target: "f128_exact_test.go",
		old:    "if lhs.Cmp(rhs) <= 0 {",
		new:    "if lhs.Cmp(rhs) < 0 {",
	},
	{
		name:   "straddle-wrong-ulp",
		target: "f128_exact_test.go",
		old:    "step := new(big.Int).Rsh(den, uint(128-c.MantExp(nil)))",
		new:    "step := new(big.Int).Rsh(den, uint(127-c.MantExp(nil)))",
	},
}

func main() {
	const defaultTarget = "f128.go"
	originals := make(map[string][]byte)
	for i := range mutants {
		if mutants[i].target == "" {
			mutants[i].target = defaultTarget
		}
		if _, ok := originals[mutants[i].target]; ok {
			continue
		}
		orig, err := os.ReadFile(mutants[i].target)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		originals[mutants[i].target] = orig
	}
	restore := func(target string) {
		if err := os.WriteFile(target, originals[target], 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "restore %s: %v\n", target, err)
			os.Exit(1)
		}
	}
	restoreAll := func() {
		for target := range originals {
			restore(target)
		}
	}
	defer restoreAll()

	killArgs := []string{
		"test", "-count=1",
		"-run", "TestF128Digest|TestF128Primitive|TestF128OpsExact|TestF128ConversionsExact|TestDivStepExact|TestDivUVsBig|TestDivVsBig|TestSelectF128|TestSelectHighPrecision|TestRapid|Fuzz",
		"-rapid.checks=2000",
		// killed mutants fail the rapid tests by design; don't litter
		// testdata/rapid with failfiles for them
		"-rapid.nofailfile",
	}

	killed, expectedSurvivors := 0, 0
	var unexpected []string
	for _, m := range mutants {
		src := string(originals[m.target])
		if c := strings.Count(src, m.old); c != 1 {
			restoreAll()
			fmt.Fprintf(os.Stderr, "mutant %s: target string occurs %d times in %s, need exactly 1\n", m.name, c, m.target)
			os.Exit(1)
		}
		if err := os.WriteFile(m.target, []byte(strings.Replace(src, m.old, m.new, 1)), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		out, testErr := exec.Command("go", killArgs...).CombinedOutput()
		restore(m.target)
		if strings.Contains(string(out), "[build failed]") {
			fmt.Fprintf(os.Stderr, "mutant %s: does not compile, fix the table\n%s\n", m.name, out)
			os.Exit(1)
		}
		switch {
		case testErr != nil && m.equivalent == "":
			killed++
			fmt.Printf("KILLED     %s\n", m.name)
		case testErr != nil && m.equivalent != "":
			fmt.Printf("UNEXPECTED KILL of equivalent mutant %s: the equivalence proof is wrong or the code changed\n", m.name)
			unexpected = append(unexpected, m.name+" (equivalent mutant was killed)")
		case testErr == nil && m.equivalent != "":
			expectedSurvivors++
			fmt.Printf("SURVIVED   %s (expected: %s)\n", m.name, m.equivalent)
		default:
			fmt.Printf("SURVIVED   %s  <-- TEST GAP\n", m.name)
			unexpected = append(unexpected, m.name)
		}
	}
	fmt.Printf("\n%d mutants: %d killed, %d expected-equivalent survivors, %d UNEXPECTED\n",
		len(mutants), killed, expectedSurvivors, len(unexpected))
	if len(unexpected) > 0 {
		for _, n := range unexpected {
			fmt.Printf("  unexpected: %s\n", n)
		}
		os.Exit(1)
	}
}
