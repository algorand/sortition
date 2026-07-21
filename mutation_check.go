//go:build ignore

// mutation_check applies curated single-site mutants to f128.go and verifies
// the fast test suite kills each one. Every mutant models a plausible
// implementation bug (flipped rounding masks, dropped carries or sticky
// terms, off-by-one boundaries, inverted ties); an UNEXPECTED survivor is a
// test gap and must be answered with a new test. Mutants marked equivalent
// encode a proof of why the mutation cannot change behavior; they are
// expected to survive and exist to document that analysis.
//
// Run from the repo root:
//
//	go run mutation_check.go
//
// See ~/ga/docs/2026-07-21-sortition-f128-testing-blind-spots.md, section T2.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

type mutant struct {
	name       string
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
	{name: "cdf-recurrence-off-by-one", old: "f128FromUint64(b.money - b.at + 1)", new: "f128FromUint64(b.money - b.at)"},
	{name: "walk-strict-compare", old: "P(X <= j)\n\t\tif ratio.cmp(boundary) <= 0 {", new: "P(X <= j)\n\t\tif ratio.cmp(boundary) < 0 {"},
	{name: "freeze-fire-on-change", old: "if b.cum.cmp(cumPrev) == 0 && b.pmf.cmp(pmfPrev) < 0 {", new: "if b.cum.cmp(cumPrev) != 0 && b.pmf.cmp(pmfPrev) < 0 {"},
	{
		name: "freeze-nonstrict-pmf", old: "b.pmf.cmp(pmfPrev) < 0 {", new: "b.pmf.cmp(pmfPrev) <= 0 {",
		equivalent: "an equal pmf whose add was a no-op still freezes cum forever: the step factor is non-increasing, so later pmfs stay <= this one and later adds stay no-ops",
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
}

func main() {
	const target = "f128.go"
	orig, err := os.ReadFile(target)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	restore := func() { os.WriteFile(target, orig, 0o644) }
	defer restore()

	killArgs := []string{
		"test", "-count=1",
		"-run", "TestF128Digest|TestF128Primitive|TestDivUVsBig|TestDivVsBig|TestSelectF128|TestRapid|Fuzz",
		"-rapid.checks=2000",
		// killed mutants fail the rapid tests by design; don't litter
		// testdata/rapid with failfiles for them
		"-rapid.nofailfile",
	}

	src := string(orig)
	killed, expectedSurvivors := 0, 0
	var unexpected []string
	for _, m := range mutants {
		if c := strings.Count(src, m.old); c != 1 {
			restore()
			fmt.Fprintf(os.Stderr, "mutant %s: target string occurs %d times, need exactly 1\n", m.name, c)
			os.Exit(1)
		}
		if err := os.WriteFile(target, []byte(strings.Replace(src, m.old, m.new, 1)), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		out, testErr := exec.Command("go", killArgs...).CombinedOutput()
		restore()
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
