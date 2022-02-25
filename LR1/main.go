package main

import (
	"fmt"
	"strings"
)

var Function = []int{0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1}

func main() {
	fmt.Println(strings.Repeat("-", 30), "FA 1st type", strings.Repeat("-", 30))
	nw := NewNeuralNetwork(1, 0.3, Function)
	nw.Study(Set)
	fmt.Println(strings.Repeat("-", 30), "FA 2nd type", strings.Repeat("-", 30))
	nw = NewNeuralNetwork(2, 0.3, Function)
	nw.Study(Set)
	FindMinSet()
	nw.Study(MinSet)

}
