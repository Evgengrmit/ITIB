package main

import (
	"fmt"
	"math"
	"math/bits"
)

var AllVariables = GetVariables()
var Set = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
var MinSet = []int{2, 7, 9, 15}

func Combinations(set []int, n int) (subsets [][]int) {
	length := uint(len(set))
	if n > len(set) {
		n = len(set)
	}
	for subsetBits := 1; subsetBits < (1 << length); subsetBits++ {
		if n > 0 && bits.OnesCount(uint(subsetBits)) != n {
			continue
		}

		var subset []int

		for object := uint(0); object < length; object++ {
			if (subsetBits>>object)&1 == 1 {
				subset = append(subset, set[object])
			}
		}
		subsets = append(subsets, subset)
	}
	return subsets
}

func GetVariables() [16][5]int {
	var variables [16][5]int
	for i := 0; i < 16; i++ {
		variables[i] = [5]int{1, (i / 8) % 2, (i / 4) % 2, (i / 2) % 2, (i / 1) % 2}
	}
	return variables
}

func Equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func GetFastSet(n int) {
	minEpoch := math.MaxInt
	var minValues []int
	subSetN := Combinations(Set, n)
	for _, v := range subSetN {
		nw := NewNeuralNetwork(2, 0.3, Function)
		epoch, ok := nw.StudyWithSet(v)

		if epoch < minEpoch && ok {
			minEpoch = epoch
			minValues = v
		}
	}
	if minValues == nil {
		fmt.Println("cannot study on this small variables n:", n)
		return
	}
	fmt.Println("fast set of size:", n, "epochs:", minEpoch, "variables:", minValues)

}

func FindMinSet() {
	for i := 15; i > 1; i-- {
		GetFastSet(i)
	}

}
