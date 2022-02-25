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
	net       float64
	weights   [5]float64
	boolFunc  []int
	errorsArr [][2]int
}

func NewNeuralNetwork(funcType int, nu float64, boolFunc []int) *NeuralNetwork {
	return &NeuralNetwork{funcType: funcType, nu: nu, net: 0, weights: [5]float64{0, 0, 0, 0, 0}, boolFunc: boolFunc}
}

func (n NeuralNetwork) weightsToString() string {
	weights := make([]string, 5, 5)
	for i := 0; i < 5; i++ {
		if n.weights[i] < 0 {
			weights[i] = fmt.Sprintf("%.3f", n.weights[i])
		} else {
			weights[i] = fmt.Sprintf("%.4f", n.weights[i])
		}
	}
	return strings.Join(weights, ", ")
}

func (n NeuralNetwork) ThresholdFunction() int {
	if n.net >= 0 {
		return 1
	}
	return 0
}

func (n NeuralNetwork) ActivationFunction() float64 {
	return 0.5 * (n.net/(1+math.Abs(n.net)) + 1)
}

func (n NeuralNetwork) DiffActivationFunction() float64 {
	return 0.5 * (1 / math.Pow(1+math.Abs(n.net), 2))
}

func (n *NeuralNetwork) CalculateNet(x [5]int) {
	n.net = 0
	for i := 0; i < 5; i++ {
		n.net += n.weights[i] * float64(x[i])
	}
}

func (n *NeuralNetwork) WeightsCorrection(sigma, dfdnet float64, x [5]int) {
	for i := 0; i < 5; i++ {
		n.weights[i] = n.weights[i] + n.nu*sigma*dfdnet*float64(x[i])
	}
}

func (n *NeuralNetwork) Study(variablesVector []int) {
	fmt.Printf("%5s %18s %39s %5s", "EPOCH", "FUNCTION", "WEIGHTS", "ERROR\n")
	fmt.Println(strings.Repeat("_", 5), strings.Repeat("_", 18),
		strings.Repeat("_", 39), strings.Repeat("_", 5))
	epoch := 0
	for {
		err := 0
		predictedY := 0
		valuesVector := ""
		for _, j := range variablesVector {
			x := AllVariables[j]
			if n.funcType == 1 {
				n.CalculateNet(x)
				if n.ThresholdFunction() == 1 {
					predictedY = 1
				} else {
					predictedY = 0
				}
			} else if n.funcType == 2 {
				n.CalculateNet(x)
				if n.ActivationFunction() >= 0.5 {
					predictedY = 1
				} else {
					predictedY = 0
				}
			}
			if predictedY != n.boolFunc[j] {
				err += 1
			}
			valuesVector += strconv.Itoa(predictedY)
			var dfdnet float64 = 1
			sigma := float64(n.boolFunc[j] - predictedY)
			n.CalculateNet(x)
			if n.funcType == 2 {
				dfdnet = n.DiffActivationFunction()
			}
			n.WeightsCorrection(sigma, dfdnet, x)
		}
		weightsString := n.weightsToString()
		n.errorsArr = append(n.errorsArr, [2]int{epoch, err})
		fmt.Printf("%5d %18s %39s %5d\n", epoch, valuesVector, weightsString, err)

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
		if n.funcType == 1 {
			n.CalculateNet(x)
			if n.ThresholdFunction() == 1 {
				predicted = append(predicted, 1)
			} else {
				predicted = append(predicted, 0)
			}
		} else if n.funcType == 2 {
			n.CalculateNet(x)
			if n.ActivationFunction() >= 0.5 {
				predicted = append(predicted, 1)
			} else {
				predicted = append(predicted, 0)
			}
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
			if n.funcType == 1 {
				n.CalculateNet(x)
				if n.ThresholdFunction() == 1 {
					predictedY = 1
				} else {
					predictedY = 0
				}
			} else if n.funcType == 2 {
				n.CalculateNet(x)
				if n.ActivationFunction() >= 0.5 {
					predictedY = 1
				} else {
					predictedY = 0
				}
			}
			if predictedY != n.boolFunc[j] {
				err += 1
			}
			var dfdnet float64 = 1
			sigma := float64(n.boolFunc[j] - predictedY)
			n.CalculateNet(x)
			if n.funcType == 2 {
				dfdnet = n.DiffActivationFunction()
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
