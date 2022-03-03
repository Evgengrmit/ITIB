package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

type NeuralNetwork struct {
	funcType  int
	nu        float64
	Weights   [5]float64
	boolFunc  []int
	epochsArr []int
	errorsArr []int
}

func NewNeuralNetwork(funcType int, nu float64, boolFunc []int) *NeuralNetwork {
	return &NeuralNetwork{funcType: funcType, nu: nu, Weights: [5]float64{0, 0, 0, 0, 0}, boolFunc: boolFunc}
}

func (n *NeuralNetwork) weightsToString() string {
	weights := make([]string, 5, 5)
	for i := 0; i < 5; i++ {
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

func (n *NeuralNetwork) CalculateNet(x [5]int) float64 {
	net := 0.0
	for i := 0; i < 5; i++ {
		net += n.Weights[i] * float64(x[i])
	}
	return net
}

func (n *NeuralNetwork) WeightsCorrection(sigma, dfdnet float64, x [5]int) {
	for i := 0; i < 5; i++ {
		n.Weights[i] += n.nu * sigma * dfdnet * float64(x[i])
	}
}

func (n *NeuralNetwork) Study(variablesVector []int) {
	fmt.Printf("%5s %18s %39s %5s", "EPOCH", "FUNCTION", "WEIGHTS", "ERROR\n")
	fmt.Println(strings.Repeat("_", 5), strings.Repeat("_", 18),
		strings.Repeat("_", 39), strings.Repeat("_", 5))
	epoch := 0

	for {
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
	predicted := make([]int, 0, 16)
	for i := 0; i < 16; i++ {
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
