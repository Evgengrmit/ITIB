package main

import (
	"fmt"
	"strings"
)

var Function = []int{0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1}

func main() {
	fmt.Println(strings.Repeat("-", 30), "FA 1st type", strings.Repeat("-", 30))
	{
		nw := NewNeuralNetwork(1, 0.3, Function)
		nw.Study(Set)
		fmt.Println(nw.Weights, nw.Run())
		Plot(nw.epochsArr, nw.errorsArr, "1.png")
	}
	fmt.Println("\n\n", strings.Repeat("-", 30), "FA 2nd type", strings.Repeat("-", 30))
	{

		nw := NewNeuralNetwork(2, 0.3, Function)
		nw.Study(Set)
		Plot(nw.epochsArr, nw.errorsArr, "2.png")
	}
	fmt.Println("\n\n", strings.Repeat("-", 25), "Get smallest values", strings.Repeat("-", 25))
	{
		FindMinSet()
	}
	fmt.Println("\n\n", strings.Repeat("-", 25), "FA smallest values", strings.Repeat("-", 25))
	{
		nw := NewNeuralNetwork(2, 0.3, Function)
		nw.Study(MinSet)
		Plot(nw.epochsArr, nw.errorsArr, "3.png")
	}

}
