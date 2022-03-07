package main

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

func getStringsFormat(m []float64) string {
	strF := make([]string, 0, len(m))
	for _, n := range m {
		strF = append(strF, fmt.Sprintf("%f", n))
	}
	return strings.Join(strF, ", ")
}

func NewNeuron(M int, a, b, eta float64) *Neuron {
	windowSize := int(math.Abs(a) + math.Abs(b))
	n := 20
	epochs := make([]int, 0, M)
	epsilons := make([]float64, 0, M)
	return &Neuron{
		epochsNumber: M,
		a:            a,
		b:            b,
		eta:          eta,
		n:            n,
		windowSize:   windowSize,
		epochs:       epochs,
		epsilons:     epsilons,
	}
}

type Neuron struct {
	epochsNumber  int
	a             float64
	b             float64
	eta           float64
	learnFunction []float64
	mainFunction  []float64
	n             int
	windowSize    int
	Weights       []float64
	Epsilon       float64
	delta         float64
	epochs        []int
	epsilons      []float64
}

func standardError(xReal, xPred []float64) (float64, error) {
	if len(xReal) != len(xPred) {
		return 0., errors.New("dimensions of the vectors are not equal")
	}
	err := 0.0
	for i := 0; i < len(xReal); i++ {
		err += math.Pow(xReal[i]-xPred[i], 2)
	}
	err = math.Sqrt(err)
	return err, nil
}
func targetFunction(t float64) float64 {
	return 0.5*math.Exp(0.5*math.Cos(0.5*t)) + math.Sin(0.5*t)
}
func (n Neuron) net(x []float64) float64 {
	var net float64
	for i := 0; i < n.windowSize; i++ {
		net += n.Weights[i+1] * x[i]
	}
	return net
}

func (n Neuron) getT(a, b float64) []float64 {
	vectorT := make([]float64, 0, n.n)
	dt := (b - a) / float64(n.n)
	for i := 0; i < n.n; i++ {
		vectorT = append(vectorT, a+float64(i)*dt)
	}
	return vectorT
}

func (n Neuron) getX(a, b float64) []float64 {
	vectorX := make([]float64, 0, n.n)
	vectorT := n.getT(a, b)
	for _, t := range vectorT {
		vectorX = append(vectorX, targetFunction(t))
	}
	return vectorX
}

func (n *Neuron) TrainingMode() {
	n.learnFunction = make([]float64, n.n)
	n.mainFunction = n.getX(n.a, n.b)
	n.Weights = make([]float64, n.windowSize+1)
	for k := 0; k < n.epochsNumber; k++ {
		for q := 0; q < n.windowSize; q++ {
			n.learnFunction[q] = n.mainFunction[q]
		}
		for i := n.windowSize; i < n.n; i++ {
			n.learnFunction[i] = n.net(n.mainFunction[i-n.windowSize : i])
			n.delta = n.mainFunction[i] - n.learnFunction[i]

			for j := 0; j < n.windowSize; j++ {
				n.Weights[j+1] += n.eta * n.delta * n.mainFunction[i-n.windowSize+j]
			}
		}
		n.Epsilon, _ = standardError(n.mainFunction, n.learnFunction)
		n.epochs = append(n.epochs, k)
		n.epsilons = append(n.epsilons, n.Epsilon)

	}
}

func (n *Neuron) WorkingMode() {
	vecFT := n.getT(n.a, n.b)
	vecFX := n.getX(n.a, n.b)
	vectorT := n.getT(n.b, 2*n.b-n.a)
	vectorX := n.getX(n.b, 2*n.b-n.a)
	testFunction := make([]float64, 28)
	tF := make([]float64, n.n)
	for i := n.n - n.windowSize; i < n.n; i++ {
		testFunction[i-(n.n-n.windowSize)] = n.learnFunction[i]
	}
	for j := n.windowSize; j < len(vectorX)+n.windowSize; j++ {
		vectorTestFunction := make([]float64, n.windowSize)
		for k := 0; k < n.windowSize; k++ {
			vectorTestFunction[k] = testFunction[k+j-n.windowSize]
		}
		testFunction[j] = n.net(vectorTestFunction)
	}
	for i := n.windowSize; i < len(testFunction); i++ {
		tF[i-n.windowSize] = testFunction[i]
	}
	fmt.Println("vecFT:", getStringsFormat(vecFT))
	fmt.Println("vecFX:", getStringsFormat(vecFX))
	fmt.Println("vectorT:", getStringsFormat(vectorT))
	fmt.Println("vectorX:", getStringsFormat(vectorX))
	fmt.Println("tF:", getStringsFormat(tF))
}

func main() {
	obj := NewNeuron(10, -5, 3, 0.115)
	obj.TrainingMode()
	obj.WorkingMode()
	fmt.Println(obj.Epsilon)
	obj = NewNeuron(465, -5, 3, 0.115)
	obj.TrainingMode()
	obj.WorkingMode()
	fmt.Println(obj.Epsilon)
	fmt.Println(obj.Weights)
}
