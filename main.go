package main

import (
	"errors"
	"image/color"
	"log"
	"math/rand"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	screenWidth  = 480
	screenHeight = 480
)

type Perceptron struct {
	weights      []float32
	learningRate float32
}

type Game struct {
	perceptron *Perceptron
	points     [][3]float32 // x, y, class
}

func (p *Perceptron) Initialize(inputSize int, learningRate float32) {
	p.weights = make([]float32, inputSize)
	p.learningRate = learningRate

	for i := range p.weights {
		p.weights[i] = rand.Float32()*2 - 1
	}
}

func (p *Perceptron) Activation(sum float32) float32 {
	if sum >= 0 {
		return 1.0
	}
	return 0.0
}

func (p *Perceptron) Predict(inputs []float32) (float32, error) {
	if len(inputs) != len(p.weights) {
		return -1, errors.New("input size does not match number of weights")
	}
	sum := float32(0)
	for i, input := range inputs {
		sum += input * p.weights[i]
	}
	return p.Activation(sum), nil
}

func (p *Perceptron) Train(inputs []float32, target float32) error {
	prediction, err := p.Predict(inputs)
	if err != nil {
		return err
	}
	targetDifference := target - prediction
	for i := range p.weights {
		p.weights[i] += p.learningRate * targetDifference * inputs[i]
	}
	return nil
}

func (g *Game) Update() error {
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	centerX := float32(screenWidth / 2)
	centerY := float32(screenHeight / 2) // Center of the screen
	scale := float32(15)

	// Draw x-axis
	for x := 0; x < screenWidth; x++ {
		vector.DrawFilledRect(screen, float32(x), centerY, 1, 1, color.RGBA{255, 255, 255, 255}, false)
	}

	// Draw y-axis
	for y := 0; y < screenHeight; y++ {
		vector.DrawFilledRect(screen, centerX, float32(y), 1, 1, color.RGBA{255, 255, 255, 255}, false)
	}

	// Draw points
	for _, pt := range g.points {
		// Transform pt[0] and pt[1] to screen space correctly accounting for positive and negative values
		x := centerX + pt[0]*scale
		y := centerY - pt[1]*scale                    // Notice the '-' sign to flip the y values correctly for graphical rendering
		clr := color.RGBA{R: 0, G: 0, B: 255, A: 255} // Default blue
		if pt[2] == 1.0 {
			clr = color.RGBA{R: 255, G: 0, B: 0, A: 255} // Red for class 1
		}
		vector.DrawFilledRect(screen, x, y, 3, 3, clr, false)
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func main() {
	game := &Game{perceptron: &Perceptron{}}
	game.perceptron.Initialize(2, 0.1)

	// Generate and classify points
	for i := 0; i < 100; i++ {
		x := rand.Float32()*20 - 10
		y := rand.Float32()*20 - 10
		class, _ := game.perceptron.Predict([]float32{x, y})
		game.points = append(game.points, [3]float32{x, y, class})
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Perceptron Visualization")
	if err := ebiten.RunGame(game); err != nil {
		log.Fatal(err)
	}
}
