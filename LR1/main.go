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
	Plot(nw.epochsArr, nw.errorsArr, "1.png")
	fmt.Println(strings.Repeat("-", 30), "FA 2nd type", strings.Repeat("-", 30))
	nw = NewNeuralNetwork(2, 0.3, Function)
	nw.Study(Set)
	Plot(nw.epochsArr, nw.errorsArr, "2.png")
	FindMinSet()
	nw = NewNeuralNetwork(2, 0.3, Function)
	nw.Study(MinSet)
	Plot(nw.epochsArr, nw.errorsArr, "3.png")

}
