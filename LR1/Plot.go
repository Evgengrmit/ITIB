package main

import (
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"math/rand"
)

func Plot(epochs, errs []int, name string) {
	rand.Seed(int64(0))

	p := plot.New()

	p.Title.Text = "Graph"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "errors"
	pts := make(plotter.XYs, len(errs))
	for i := range pts {
		if i == 0 {
			pts[i].X = float64(epochs[i])
		} else {
			pts[i].X = pts[i-1].X + float64(epochs[i-1])
		}
		pts[i].Y = float64(errs[i])

	}
	err := plotutil.AddLinePoints(p, "First", pts)
	if err != nil {
		panic(err)
	}

	if err := p.Save(4*vg.Inch, 4*vg.Inch, name); err != nil {
		panic(err)
	}
}
