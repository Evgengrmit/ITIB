package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

var Set = []int{0, 1, 2, 3}
var Function = []int{1, 0, 0, 1}
var AllVariables = GetVariables()

type NeuralNetwork struct {
	funcType  int
	nu        float64
	Weights   [3]float64
	boolFunc  []int
	epochsArr []int
	errorsArr []int
}

func NewNeuralNetwork(funcType int, nu float64, boolFunc []int) *NeuralNetwork {
	return &NeuralNetwork{funcType: funcType, nu: nu, Weights: [3]float64{0, 0, 0}, boolFunc: boolFunc}
}

func (n *NeuralNetwork) weightsToString() string {
	weights := make([]string, 3, 3)
	for i := 0; i < 3; i++ {
		if n.Weights[i] < 0 {
			weights[i] = fmt.Sprintf("%.3f", n.Weights[i])
		} else {
			weights[i] = fmt.Sprintf("%.4f", n.Weights[i])
		}
	}
	return strings.Join(weights, ", ")
}

func (n NeuralNetwork) ThresholdFunction(net float64) int {
	if net >= 0 {
		return 1
	}
	return 0
}

func (n NeuralNetwork) ActivationFunction(net float64) int {
	if 0.5*(net/(1+math.Abs(net))+1) >= 0.5 {
		return 1
	}
	return 0
}

func (n NeuralNetwork) DiffActivationFunction(net float64) float64 {
	return 0.5 * (1 / math.Pow(1+math.Abs(net), 2))
}

func (n *NeuralNetwork) CalculateNet(x [3]int) float64 {
	net := 0.0
	for i := 0; i < 3; i++ {
		net += n.Weights[i] * float64(x[i])
	}
	fmt.Println(net)
	return net
}

func (n *NeuralNetwork) WeightsCorrection(sigma, dfdnet float64, x [3]int) {
	for i := 0; i < 3; i++ {
		n.Weights[i] += n.nu * sigma * dfdnet * float64(x[i])
	}
	fmt.Println(n.Weights)
}

func (n *NeuralNetwork) Study(variablesVector []int) {
	fmt.Printf("%5s %18s %39s %5s", "EPOCH", "FUNCTION", "WEIGHTS", "ERROR\n")
	fmt.Println(strings.Repeat("_", 5), strings.Repeat("_", 18),
		strings.Repeat("_", 39), strings.Repeat("_", 5))
	epoch := 0

	for k := 0; k < 3; k++ {
		err := 0
		valuesVector := ""
		var predictedY int
		weightsString := n.weightsToString()
		for _, j := range variablesVector {
			x := AllVariables[j]
			net := n.CalculateNet(x)
			if n.funcType == 1 {
				predictedY = n.ThresholdFunction(net)
			} else if n.funcType == 2 {
				predictedY = n.ActivationFunction(net)
			}
			if predictedY != n.boolFunc[j] {
				err += 1
			}
			valuesVector += strconv.Itoa(predictedY)
		}
		fmt.Printf("%5d %18s %39s %5d\n", epoch, valuesVector, weightsString, err)
		n.errorsArr = append(n.errorsArr, err)
		n.epochsArr = append(n.errorsArr, epoch)

		for _, j := range variablesVector {
			x := AllVariables[j]
			net := n.CalculateNet(x)
			if n.funcType == 1 {
				predictedY = n.ThresholdFunction(net)
			} else if n.funcType == 2 {
				predictedY = n.ActivationFunction(net)
			}
			var dfdnet float64 = 1
			sigma := float64(n.boolFunc[j] - predictedY)
			n.CalculateNet(x)
			if n.funcType == 2 {
				dfdnet = n.DiffActivationFunction(net)
			}
			n.WeightsCorrection(sigma, dfdnet, x)
		}
		if err == 0 {
			return
		}

		epoch += 1
	}

}

func (n NeuralNetwork) Run() bool {
	predicted := make([]int, 0, 4)
	for i := 0; i < 4; i++ {
		x := AllVariables[i]
		net := n.CalculateNet(x)
		if n.funcType == 1 {
			predicted = append(predicted, n.ThresholdFunction(net))
		} else if n.funcType == 2 {
			predicted = append(predicted, n.ActivationFunction(net))
		}
	}
	return Equal(Function, predicted)
}

func (n *NeuralNetwork) StudyWithSet(variablesVector []int) (int, bool) {
	epoch := 0
	for {
		err := 0
		predictedY := 0
		for _, j := range variablesVector {
			x := AllVariables[j]
			net := n.CalculateNet(x)
			if n.funcType == 1 {
				predictedY = n.ThresholdFunction(net)
			} else if n.funcType == 2 {
				predictedY = n.ActivationFunction(net)
			}
			if predictedY != n.boolFunc[j] {
				err += 1
			}
			var dfdnet float64 = 1
			sigma := float64(n.boolFunc[j] - predictedY)
			n.CalculateNet(x)
			if n.funcType == 2 {
				net := n.CalculateNet(x)
				dfdnet = n.DiffActivationFunction(net)
			}
			n.WeightsCorrection(sigma, dfdnet, x)
		}

		if err == 0 {
			break
		}
		epoch += 1
	}
	if n.Run() {

		return epoch, true
	}
	return 0, false

}

func GetVariables() [4][3]int {
	var variables [4][3]int
	for i := 0; i < 4; i++ {
		variables[i] = [3]int{1, (i / 2) % 2, (i / 1) % 2}
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

func main() {
	fmt.Println(strings.Repeat("-", 30), "FA 1st type", strings.Repeat("-", 30))
	{
		nw := NewNeuralNetwork(1, 1, Function)
		nw.Study(Set)
		fmt.Println(nw.Weights, nw.Run())

	}
	fmt.Println("\n\n", strings.Repeat("-", 30), "FA 2nd type", strings.Repeat("-", 30))
	{

		nw := NewNeuralNetwork(2, 1, Function)
		nw.Study(Set)

	}

}
