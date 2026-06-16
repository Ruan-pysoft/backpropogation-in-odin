package backprop

import "core:c"
import "core:fmt"
import "core:math"
import "core:math/rand"

loss :: proc(expected, actual: f32) -> f32 {
	return (actual - expected)*(actual - expected)
}
dloss :: proc(expected, actual: f32) -> f32 {
	return 2*(actual - expected)
}

activation :: proc(input: f32) -> f32 {
	return 1 / (1 + math.exp(-input))
}
dactivation :: proc(input: f32) -> f32 {
	act := activation(input)
	return act*(1 - act)
}

sin :: proc(x: f32) -> f32 { return math.sin(x) }
dsin :: proc(x: f32) -> f32 { return math.cos(x) }

ActivationFunction :: struct {
	func, dfunc: proc(f32) -> f32,
}

LossFunction :: struct {
	func, dfunc: proc(y: f32, o: f32) -> f32,
}

LayerType :: enum {
	input,
	hidden,
}

LayerPointer :: struct($size: uint) {
	layer: rawptr,
	type: LayerType,
}

InputLayer :: struct($size: uint) {
	nodes: [size]f32,
}

layer_pointer_il :: proc(il: ^InputLayer($size)) -> LayerPointer(size) {
	return { il, .input }
}

il_load :: proc(il: ^InputLayer($size), input: [size]f32) {
	il.nodes = input
}

HiddenLayer :: struct($size, $prevsize: uint) {
	nodes: [size]f32,
	dnodes: [size]f32,
	prev_layer: LayerPointer(prevsize),
	weights: [size][prevsize]f32,
	act: ActivationFunction,
}

layer_pointer_hl :: proc(hl: ^HiddenLayer($size, $prevsize)) -> LayerPointer(size) {
	return { hl, .hidden }
}

hl_init_weights :: proc(hl: ^HiddenLayer($size, $prevsize)) {
	for i in 0..<size {
		for j in 0..<prevsize {
			hl.weights[i][j] = rand.float32()*2 - 1
		}
	}
}

hl_propogate :: proc(hl: ^HiddenLayer($size, $prevsize)) {
	for i in 0..<size {
		acc: f32 = 0

		for j in 0..<prevsize {
			acc += lp_get(hl.prev_layer, j)*hl.weights[i][j]
		}

		hl.nodes[i] = hl.act.func(acc)
		hl.dnodes[i] = hl.act.dfunc(acc)
	}
}

hl_backprop :: proc(hl: ^HiddenLayer($size, $prevsize), next_deltas_sums: [size]f32, eta: f32) -> (deltas_sums: [prevsize]f32) {
	deltas_sums = [prevsize]f32{}

	for i in 0..<size {
		delta := next_deltas_sums[i] * hl.dnodes[i]

		for j in 0..<prevsize {
			deltas_sums[j] += hl.weights[i][j]*delta

			o := lp_get(hl.prev_layer, j)
			change := -eta * o * delta

			hl.weights[i][j] += change
		}
	}

	return deltas_sums
}

OutputLayer :: struct($size, $prevsize: uint) {
	nodes: [size]f32,
	dnodes: [size]f32,
	prev_layer: LayerPointer(prevsize),
	weights: [size][prevsize]f32,
	act: ActivationFunction,
	loss: LossFunction,
}

layer_pointer_ol :: proc(ol: ^OutputLayer($size, $prevsize)) -> LayerPointer(size) {
	return { ol, .input }
}

ol_init_weights :: proc(ol: ^OutputLayer($size, $prevsize)) {
	for i in 0..<size {
		for j in 0..<prevsize {
			ol.weights[i][j] = rand.float32()*2 - 1
		}
	}
}

ol_propogate :: proc(ol: ^OutputLayer($size, $prevsize)) {
	for i in 0..<size {
		acc: f32 = 0

		for j in 0..<prevsize {
			acc += lp_get(ol.prev_layer, j)*ol.weights[i][j]
		}

		ol.nodes[i] = ol.act.func(acc)
		ol.dnodes[i] = ol.act.func(acc)
	}
}

ol_backprop :: proc(ol: ^OutputLayer($size, $prevsize), expected: [size]f32, eta: f32) -> (deltas_sums: [prevsize]f32) {
	deltas_sums = [prevsize]f32{}

	for i in 0..<size {
		delta := ol.loss.dfunc(expected[i], ol.nodes[i]) * ol.dnodes[i]

		for j in 0..<prevsize {
			deltas_sums[j] += ol.weights[i][j]*delta

			o := lp_get(ol.prev_layer, j)
			change := -eta * o * delta

			ol.weights[i][j] += change
		}
	}

	return deltas_sums
}

ol_get_error :: proc(ol: OutputLayer($size, $prevsize), expected: [size]f32) -> (error: f32) {
	error = 0

	for i in 0..<size {
		error += ol.loss.func(expected[i], ol.nodes[i])
	}

	return
}

layer_pointer :: proc{
	layer_pointer_il,
	layer_pointer_hl,
	layer_pointer_ol,
}

lp_get :: proc(lp: LayerPointer($size), idx: uint) -> f32 {
	assert(idx < size)

	switch lp.type {
	case .input:
		layer: ^InputLayer(size) = auto_cast lp.layer
		return layer.nodes[idx]
	case .hidden:
		layer_h: ^HiddenLayer(size, 2) = auto_cast lp.layer
		return layer_h.nodes[idx]
	}

	assert(false, "expected switch statement to be exhaustive")
	return 0
}

NeuralNet :: struct($input_size, $pre_output_size, $output_size: uint) {
	layer_input: InputLayer(input_size),
	layer_output: OutputLayer(output_size, pre_output_size),
}

input_size :: 2
output_size :: 3

Network :: struct {
	layer_input: ^InputLayer(input_size),
	layer_hidden1: ^HiddenLayer(4, input_size),
	layer_hidden2: ^HiddenLayer(8, 4),
	layer_hidden3: ^HiddenLayer(8, 8),
	layer_hidden4: ^HiddenLayer(16, 8),
	layer_hidden5: ^HiddenLayer(4, 16),
	layer_output: ^OutputLayer(output_size, 4),

	eta: f32,
}

network_new :: proc() -> (result: Network) {
	result = {
		new(InputLayer(input_size)),
		new(HiddenLayer(4, input_size)),
		new(HiddenLayer(8, 4)),
		new(HiddenLayer(8, 8)),
		new(HiddenLayer(16, 8)),
		new(HiddenLayer(4, 16)),
		new(OutputLayer(output_size, 4)),

		0.01,
	}

	result.layer_input^ = {
		[input_size]f32{},
	}

	result.layer_hidden1^ = {
		[4]f32{},
		[4]f32{},
		layer_pointer(result.layer_input),
		[4][input_size]f32{},
		{ sin, dsin },
	}
	hl_init_weights(result.layer_hidden1)
	result.layer_hidden2^ = {
		[8]f32{},
		[8]f32{},
		layer_pointer(result.layer_hidden1),
		[8][4]f32{},
		{ sin, dsin },
	}
	hl_init_weights(result.layer_hidden2)
	result.layer_hidden3^ = {
		[8]f32{},
		[8]f32{},
		layer_pointer(result.layer_hidden2),
		[8][8]f32{},
		{ activation, dactivation },
	}
	hl_init_weights(result.layer_hidden3)
	result.layer_hidden4^ = {
		[16]f32{},
		[16]f32{},
		layer_pointer(result.layer_hidden3),
		[16][8]f32{},
		{ activation, dactivation },
	}
	hl_init_weights(result.layer_hidden4)
	result.layer_hidden5^ = {
		[4]f32{},
		[4]f32{},
		layer_pointer(result.layer_hidden4),
		[4][16]f32{},
		{ activation, dactivation },
	}
	hl_init_weights(result.layer_hidden5)

	result.layer_output^ = {
		[output_size]f32{},
		[output_size]f32{},
		layer_pointer(result.layer_hidden5),
		[output_size][4]f32{},
		{ activation, dactivation },
		{ loss, dloss },
	}
	ol_init_weights(result.layer_output)

	return
}

network_free :: proc(n: ^Network) {
}

network_propogate :: proc(n: ^Network, input: [input_size]f32) {
	il_load(n.layer_input, input)
	hl_propogate(n.layer_hidden1)
	hl_propogate(n.layer_hidden2)
	hl_propogate(n.layer_hidden3)
	hl_propogate(n.layer_hidden4)
	hl_propogate(n.layer_hidden5)
	ol_propogate(n.layer_output)
}

network_error :: proc(n: Network, expected: [output_size]f32) -> f32 {
	return ol_get_error(n.layer_output^, expected)
}

network_backprop :: proc(n: ^Network, expected: [output_size]f32) {
	dss0 := ol_backprop(n.layer_output, expected, n.eta)
	dss1 := hl_backprop(n.layer_hidden5, dss0, n.eta)
	dss2 := hl_backprop(n.layer_hidden4, dss1, n.eta)
	dss3 := hl_backprop(n.layer_hidden3, dss2, n.eta)
	dss4 := hl_backprop(n.layer_hidden2, dss3, n.eta)
	dss5 := hl_backprop(n.layer_hidden1, dss4, n.eta)
}

network: Network
img: Image(.rgb)

main :: proc() {
	rand.reset(42)

	network = network_new()

	img = image_load("GoblinFace.jpg", .rgb)
	defer image_free(&img)
	if (!img.valid) {
		fmt.println(stbi_failure_reason())
		return
	}

	generations :: 80
	print_every :: 8

	fmt.println("Starting network training!")
	{
		error: f32 = 0

		for y in 0..<img.height {
			for x in 0..<img.width {
				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				network_propogate(&network, [?]f32{ xnorm, ynorm })

				pixel := image_get(img, x, y)

				error += network_error(network, [output_size]f32 {
					f32(pixel.r)/255,
					f32(pixel.g)/255,
					f32(pixel.b)/255,
				})
			}
		}

		fmt.printf("Before training, error is: {}\n", error)
	}

	for g in 0..<generations {
		for y in 0..<img.height {
			for x in 0..<img.width {
				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				network_propogate(&network, [?]f32{ xnorm, ynorm })

				pixel := image_get(img, x, y)

				network_backprop(&network, [output_size]f32 {
					f32(pixel.r)/255,
					f32(pixel.g)/255,
					f32(pixel.b)/255,
				})
			}
		}

		if g == generations/4 {
			network.eta /= 10
		}
		if g == generations/2 {
			network.eta /= 16
		}
		if g == 3*generations/4 {
			network.eta /= 64
		}

		if (g+1) % print_every == 0 {
			error: f32 = 0

			for y in 0..<img.height {
				for x in 0..<img.width {
					xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
					network_propogate(&network, [?]f32{ xnorm, ynorm })

					pixel := image_get(img, x, y)

					error += network_error(network, [output_size]f32 {
						f32(pixel.r)/255,
						f32(pixel.g)/255,
						f32(pixel.b)/255,
					})
				}
			}

			fmt.printf("After {} generations of training, error is: {}\n", g+1, error)
		} else {
			fmt.printf("Generation {} complete...\n", g+1)
		}
	}

	fmt.printf("{}x{}\n", img.width, img.height)
	fmt.printf("Top-left pixel: {}\n", image_get(img, 0, 0))

	image_modify(img, proc(x, y: uint, pixel: PixelRgb) -> PixelRgb {
		xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
		network_propogate(&network, [?]f32{ xnorm, ynorm })
		//fmt.printf("output for {},{} = {}\n", x, y, network.layer_output.nodes)
		return {
			u8(network.layer_output.nodes[0]*255),
			u8(network.layer_output.nodes[1]*255),
			u8(network.layer_output.nodes[2]*255),
		}
	})

	image_write_jpg(img, "output.jpg", 75)
}
