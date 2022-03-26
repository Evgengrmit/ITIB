package main

import (
	"fmt"
	"math"
)

var Set = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
var MinSet = []int{2, 7, 9, 15}
var Function = [16]int{0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1}

type RBFNeuron struct {
	net       float64
	j         int
	fiV       []float64
	u         []float64
	n         float64
	c         [][4]int
	functionY [16]int
	functionX [][4]int
}

func NewRBFNeuron(n float64) *RBFNeuron {
	return &RBFNeuron{
		n: n,
	}
}

func (n RBFNeuron) thresholdFunction() int {
	if n.net >= 0 {
		return 1
	}
	return 0
}

func (n *RBFNeuron) calculateNet() {
	net := 0.
	for i := 0; i < len(n.u); i++ {
		net += n.fiV[i] * n.u[i]
	}
	n.net = net
}
func (n *RBFNeuron) jCount() {
	n.j = min(count(n.functionY, 1), count(n.functionY, 0))
	c := make([][4]int, 0)

	for i := 0; i < len(n.functionY); i++ {
		if count(n.functionY, 1) < count(n.functionY, 0) {
			if n.functionY[i] == 1 {
				c = append(c, n.functionX[i])
			}
		} else {
			if n.functionY[i] == 0 {
				c = append(c, n.functionX[i])
			}
		}
		n.c = c
	}
}

func (n *RBFNeuron) initialize() {
	X := make([][4]int, 0)
	for _, i := range Set {
		x := [4]int{(i / 8) % 2, (i / 4) % 2, (i / 2) % 2, (i / 1) % 2}
		X = append(X, x)
	}
	n.functionX = X
	n.functionY = Function

}
func (n *RBFNeuron) fi(x [4]int, c [][4]int) []float64 {
	fiY := make([][]int, 0)
	for _, cc := range c {
		cur := make([]int, 0)
		for i := 0; i < len(x); i++ {
			cur = append(cur, x[i]-cc[i])
		}
		fiY = append(fiY, cur)
	}
	fi := make([]float64, 0)
	fi = append(fi, 1)
	for _, v := range fiY {
		fi = append(fi, math.Exp(float64(sumInt(v))))
	}
	return fi

}

func (n *RBFNeuron) Study() {
	n.initialize()
	n.jCount()
	n.u = make([]float64, len(n.c)+1)
	sumError := make([]int, 0)
	y := [16]int{}
	f := [16]int{}
	era := 0
	out := ""
	for {
		for i := 0; i < len(n.functionY); i++ {
			n.fiV = n.fi(n.functionX[i], n.c)
			n.calculateNet()
			y[i] = n.thresholdFunction()
			f[i] = n.functionY[i]
			sigma := float64(f[i] - y[i])
			for j := 0; j < len(n.u); j++ {
				n.u[j] += n.n * sigma * n.fiV[j]
			}
		}
		errors := 0
		for k := 0; k < len(n.functionX); k++ {
			errors += int(math.Abs(float64(f[k] - y[k])))
		}
		sumError = append(sumError, errors)
		out += fmt.Sprintf("%d y: %v Целевой вектор f:%v, w:%v Error = %d \n", era, y, f, n.u, errors)
		era++
		if era > 1000 || errors == 0 {
			break
		}

	}
	fmt.Printf("Выборка: %v Функция обучена правильно!\n %s", n.functionX, out)
}

func sumFloat(m []float64) float64 {
	summa := 0.
	ch := make(chan float64)
	go func() {
		for _, v := range m {
			ch <- float64(v * v)
		}
		close(ch)
	}()
	for v := range ch {
		summa += v
	}
	return -1 * summa
}

func sumInt(m []int) int {
	summa := 0
	ch := make(chan int)
	go func() {
		for _, v := range m {
			ch <- v * v
		}
		close(ch)
	}()
	for v := range ch {
		summa += v
	}
	return -1 * summa
}

func count(slice [16]int, value int) int {
	dict := make(map[int]int)
	for _, v := range slice {
		dict[v]++
	}
	return dict[value]
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

func main() {
	obj := NewRBFNeuron(0.3)
	obj.Study()
}
