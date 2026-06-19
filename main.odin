package backprop

import "core:c"
import "core:fmt"
import "core:math"
import "core:math/rand"

ActivationFunction :: struct {
	func, dfunc: proc(f32) -> f32,
}

linear_act: ActivationFunction: {
	proc(x: f32) -> f32 { return x },
	proc(f32) -> f32 { return 1 },
}

logistic_act: ActivationFunction: {
	proc(x: f32) -> f32 { return 1 / (1 + math.exp(-x)) },
	proc(x: f32) -> f32 {
		act := logistic_act.func(x)
		return act*(1 - act)
	},
}

sin_act: ActivationFunction: {
	proc(x: f32) -> f32 { return math.sin(x) },
	proc(x: f32) -> f32 { return math.cos(x) },
}

adj_sin_act: ActivationFunction: {
	proc(x: f32) -> f32 { return math.abs(x) * math.sin(math.PI*x / 2) },
	proc(x: f32) -> f32 {
		x := math.abs(x)
		x = math.PI*x / 2
		return math.sin(x) + x*math.cos(x)
	},
}

/* appears I got the derivative wrong...
cutoff_sin_act: ActivationFunction: {
	proc(x: f32) -> f32 {
		if -1 <= x || x <= 1 {
			return math.sin(math.PI*x / 2)
		} else if x < -1 {
			return (x+1)/64 - 1
		} else {
			return (x-1)/64 + 1
		}
	},
	proc(x: f32) -> f32 {
		if -1 <= x || x <= 1 {
			return (math.PI/2)*math.cos(math.PI*x / 2)
		} else {
			return 1/64
		}
	},
}
*/

// TODO: support softmax output layer?

tanh_act: ActivationFunction: {
	proc(x: f32) -> f32 { return math.tanh(x) },
	proc(x: f32) -> f32 {
		t := math.tanh(x)
		return 1 - t*t
	},
}

relu_act: ActivationFunction: {
	proc(x: f32) -> f32 { return math.max(0, x) },
	proc(x: f32) -> f32 { return 1 if x >= 0 else 0 },
}

leaky_relu_act: ActivationFunction: {
	proc(x: f32) -> f32 { return x if x > 0 else x/32 },
	proc(x: f32) -> f32 { return 1 if x > 0 else 1/32 },
}

softplus_act: ActivationFunction: {
	//proc(x: f32) -> f32 { return math.log(1 + math.exp(x)) },
	proc(x: f32) -> f32 { return math.LN2 if x == 0 else x/(1 - math.exp(-x/math.LN2)) },
	proc(x: f32) -> f32 { return 1/(1 + math.exp(-x)) },
}

swish_act: ActivationFunction: {
	proc(x: f32) -> f32 { return x*logistic_act.func(x) },
	proc(x: f32) -> f32 { return logistic_act.func(x) + x*logistic_act.dfunc(x) },
	/*proc(x: f32) -> f32 {
		coshx2 := math.cosh(x/2)
		return (x + math.sinh(x))/(4*coshx2*coshx2) + 1/2
	}*/
}

mish_act: ActivationFunction: {
	proc(x: f32) -> f32 { return x*math.tanh(swish_act.func(x)) },
	proc(x: f32) -> f32 {
		swish_val := swish_act.func(x)
		return (0 +
			tanh_act.func(swish_val) +
			x*tanh_act.dfunc(swish_val)*swish_act.dfunc(x) +
		0)
	}
}

LossFunction :: struct {
	func, dfunc: proc(y, o: f32) -> f32,
}

quad_loss: LossFunction: {
	proc(y, o: f32) -> f32 { return (o-y)*(o-y) },
	proc(y, o: f32) -> f32 { return 2*(o-y) },
}

mad_loss: LossFunction: {
	proc(y, o: f32) -> f32 { return math.abs(o-y) },
	proc(y, o: f32) -> f32 { return -1 if o-y < 0 else 1 },
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
	biases: [size]f32,
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

		hl.biases[i] = rand.float32()*2 - 1
	}
}

hl_propogate :: proc(hl: ^HiddenLayer($size, $prevsize)) {
	for i in 0..<size {
		acc: f32 = hl.biases[i]

		for j in 0..<prevsize {
			acc += lp_get(hl.prev_layer, j)*hl.weights[i][j]
		}

		hl.nodes[i] = hl.act.func(acc)
		hl.dnodes[i] = hl.act.dfunc(acc)
	}
}

hl_backprop :: proc(hl: ^HiddenLayer($size, $prevsize), next_deltas_sums: [size]f32, eta: f32) -> (deltas_sums: [prevsize]f32) {
	// Also I realise the comments are criptic,
	// but I basically just implemented wikipedia's algorithm while I was zonked out on sleep deprivations
	// so I'm sitting here with a piece of paper deriving how all this works
	// and then writing a comment to document it somewhat
	// ...unfortunately I can't include a diagram in the comments

	// In this function:
	// For each weight, we find the partial derivative of the loss function with respect to that weight
	// This partial derivative works out to:
	//   (dL/dw) = (dL/dy)(dy/dw) = (dL/dy)(df(...)/dw) = (dL/dy)(xf'(...))
	// where L is the loss function,
	// y is the output of the current node,
	// f is the activation function,
	// ... is sum(w*x) + b where b is the bias,
	// and x is the input associated with the current weight

	// Then for the input associated with each weight,
	// we want to calculate (dL/dx) which becomes (dL/dy) for that backprop function
	//   (dL/dx) = (dL/dy)(dy/dx) = (dL/dy)(df(...)/dx) = (dL/dy)(wf'(...))

	deltas_sums = [prevsize]f32{}

	for i in 0..<size {
		// Adjusting the inputs of node i

		// (dL/dy) = next_deltas_sums[i]
		// f'(...) = hl.dnodes[i]

		// d = (dL/dy) * f'(...)
		delta := next_deltas_sums[i] * hl.dnodes[i]

		for j in 0..<prevsize {
			// now for each weight, we can find its partial derivative as
			//   xd
			// where x is its associated input
			// and d is the variable `delta`.

			// we can also find (dL/dx) = wd
			// (partial derivative through this node; still needs to be added to partials through the other nodes)
			// and add that to deltas_sums[j]

			// w = hl.weights[i][j]
			w := hl.weights[i][j]
			// x = lp_get(hl.prev_layer, j)
			x := lp_get(hl.prev_layer, j)

			deltas_sums[j] += w*delta

			partial_derivative := x*delta

			hl.weights[i][j] += -eta*partial_derivative
		}

		// for the bias, its partial derivative is simply d
		// since (where b is the bias):
		//   (dL/db) = (dL/dy)(dy/db) = (dL/dy)(df(...)/db) = (dL/dy)f'(...) = d

		hl.biases[i] += -eta*delta
	}

	return deltas_sums
}

OutputLayer :: struct($size, $prevsize: uint) {
	nodes: [size]f32,
	dnodes: [size]f32,
	prev_layer: LayerPointer(prevsize),
	weights: [size][prevsize]f32,
	biases: [size]f32,
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

		ol.biases[i] = rand.float32()*2 - 1
	}
}

ol_propogate :: proc(ol: ^OutputLayer($size, $prevsize)) {
	for i in 0..<size {
		acc: f32 = ol.biases[i]

		for j in 0..<prevsize {
			acc += lp_get(ol.prev_layer, j)*ol.weights[i][j]
		}

		ol.nodes[i] = ol.act.func(acc)
		ol.dnodes[i] = ol.act.dfunc(acc)
	}
}

ol_backprop :: proc(ol: ^OutputLayer($size, $prevsize), expected: [size]f32, eta: f32) -> (deltas_sums: [prevsize]f32) {
	// see comments in the hl_backprop function

	deltas_sums = [prevsize]f32{}

	for i in 0..<size {
		// (dL/dy) = ol.loss.dfunc(expected[i], ol.nodes[i])

		// d = (dL/dy) * f'(...)
		delta := ol.loss.dfunc(expected[i], ol.nodes[i]) * ol.dnodes[i]

		for j in 0..<prevsize {
			w := ol.weights[i][j]
			x := lp_get(ol.prev_layer, j)

			deltas_sums[j] += w*delta

			partial_derivative := x*delta

			ol.weights[i][j] += -eta*partial_derivative
		}

		ol.biases[i] += -eta*delta
	}

	return deltas_sums
}

ol_get_error :: proc(ol: OutputLayer($size, $prevsize), expected: [size]f32) -> (error: f32) {
	error = 0

	for i in 0..<size {
		error += ol.loss.func(expected[i], ol.nodes[i])
	}

	error /= f32(size)

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
	layer_hidden1: ^HiddenLayer(8, input_size),
	layer_hidden2: ^HiddenLayer(4, 8),
	layer_hidden3: ^HiddenLayer(16, 4),
	layer_hidden4: ^HiddenLayer(32, 16),
	layer_hidden5: ^HiddenLayer(8, 32),
	layer_hidden6: ^HiddenLayer(4, 8),
	layer_output: ^OutputLayer(output_size, 4),

	eta: f32,
}

network_new :: proc() -> (result: Network) {
	result = {
		new(InputLayer(input_size)),
		new(HiddenLayer(8, input_size)),
		new(HiddenLayer(4, 8)),
		new(HiddenLayer(16, 4)),
		new(HiddenLayer(32, 16)),
		new(HiddenLayer(8, 32)),
		new(HiddenLayer(4, 8)),
		new(OutputLayer(output_size, 4)),

		0.0625,
	}

	result.layer_input^ = {
		[input_size]f32{},
	}

	result.layer_hidden1^ = {
		[8]f32{},
		[8]f32{},
		layer_pointer(result.layer_input),
		[8][input_size]f32{},
		0,
		sin_act,
	}
	hl_init_weights(result.layer_hidden1)
	result.layer_hidden2^ = {
		[4]f32{},
		[4]f32{},
		layer_pointer(result.layer_hidden1),
		[4][8]f32{},
		0,
		logistic_act,
	}
	hl_init_weights(result.layer_hidden2)
	result.layer_hidden3^ = {
		[16]f32{},
		[16]f32{},
		layer_pointer(result.layer_hidden2),
		[16][4]f32{},
		0,
		sin_act,
	}
	hl_init_weights(result.layer_hidden3)
	result.layer_hidden4^ = {
		[32]f32{},
		[32]f32{},
		layer_pointer(result.layer_hidden3),
		[32][16]f32{},
		0,
		logistic_act,
	}
	hl_init_weights(result.layer_hidden4)
	result.layer_hidden5^ = {
		[8]f32{},
		[8]f32{},
		layer_pointer(result.layer_hidden4),
		[8][32]f32{},
		0,
		tanh_act,
	}
	hl_init_weights(result.layer_hidden5)
	result.layer_hidden6^ = {
		[4]f32{},
		[4]f32{},
		layer_pointer(result.layer_hidden5),
		[4][8]f32{},
		0,
		tanh_act,
	}
	hl_init_weights(result.layer_hidden6)

	result.layer_output^ = {
		[output_size]f32{},
		[output_size]f32{},
		layer_pointer(result.layer_hidden6),
		[output_size][4]f32{},
		0,
		logistic_act,
		quad_loss,
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
	hl_propogate(n.layer_hidden6)
	ol_propogate(n.layer_output)
}

network_error :: proc(n: Network, expected: [output_size]f32) -> f32 {
	return ol_get_error(n.layer_output^, expected)
}

network_backprop :: proc(n: ^Network, expected: [output_size]f32) {
	dssO := ol_backprop(n.layer_output, expected, n.eta)
	dss6 := hl_backprop(n.layer_hidden6, dssO, n.eta)
	dss5 := hl_backprop(n.layer_hidden5, dss6, n.eta)
	dss4 := hl_backprop(n.layer_hidden4, dss5, n.eta)
	dss3 := hl_backprop(n.layer_hidden3, dss4, n.eta)
	dss2 := hl_backprop(n.layer_hidden2, dss3, n.eta)
	dss1 := hl_backprop(n.layer_hidden1, dss2, n.eta)
}

network: Network
img: Image(.rgb_alpha)

main :: proc() {
	ok: bool

	rand.reset(42)

	network = network_new()

	//img, ok = image_load("GoblinFace_small.jpg", .rgb)
	img, ok = image_load("test.png", .rgb_alpha)
	//img, ok = image_load("grass_block_side.png", .rgb_alpha)
	defer image_free(&img)
	if !ok {
		fmt.println(stbi_failure_reason())
		return
	}

	generations :: 10_000
	print_every :: 250

	fmt.println("Starting network training!")
	{
		error: f32 = 0

		for y in 0..<img.height {
			for x in 0..<img.width {
				pixel := image_get(img, x, y)
				if pixel.alpha != 255 { continue }

				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				network_propogate(&network, [?]f32{ xnorm, ynorm })

				error += network_error(network, [output_size]f32 {
					f32(pixel.r)/255,
					f32(pixel.g)/255,
					f32(pixel.b)/255,
				})
			}
		}

		error /= f32(img.width*img.height)

		fmt.printf("Before training, avg. error is: {}\n", error)
	}

	error_history := new([generations]f32)
	eta_history := new([generations]f32)
	error_max: f32 = 0
	eta_max: f32 = 0

	for g in 0..<generations {
		if g+1 == 1 {
			network.eta = 1/16.0
		}
		if g+1 == 5_000 {
			network.eta = 1/32.0
		}
		/*if g+1 == 8_000 {
			network.eta = 3/64.0
		}*/
		if g+1 == 9_000 {
			network.eta -= network.eta/4
		}
		if g+1 == 9_500 {
			network.eta = 1/64.0
		}
		if g+1 == 10_500 {
			network.eta = 1/64.0
		}
		if g+1 == 20_000 {
			network.eta = 1/128.0
		}
		if g+1 == 50_000 {
			network.eta = 1/256.0
		}
		if g+1 == 75_000 {
			network.eta = 1/512.0
		}

		eta_history[g] = network.eta
		eta_max = math.max(eta_max, network.eta)

		for y in 0..<img.height {
			for x in 0..<img.width {
				pixel := image_get(img, x, y)
				if pixel.alpha != 255 { continue }

				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				network_propogate(&network, [?]f32{ xnorm, ynorm })

				network_backprop(&network, [output_size]f32 {
					f32(pixel.r)/255,
					f32(pixel.g)/255,
					f32(pixel.b)/255,
				})
			}
		}

		error: f32 = 0

		for y in 0..<img.height {
			for x in 0..<img.width {
				pixel := image_get(img, x, y)
				if pixel.alpha != 255 { continue }

				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				network_propogate(&network, [?]f32{ xnorm, ynorm })

				error += network_error(network, [output_size]f32 {
					f32(pixel.r)/255,
					f32(pixel.g)/255,
					f32(pixel.b)/255,
				})
			}
		}

		error /= f32(img.width*img.height)

		error_history[g] = error
		error_max = max(error_max, error)

		if (g+1) % print_every == 0 {
			fmt.printf("After {} generations of training, avg error is: {}\n", g+1, error)
		} else {
			//fmt.printf("Generation {} complete...\n", g+1)
		}
	}

	fmt.printf("{}x{}\n", img.width, img.height)
	fmt.printf("Top-left pixel: {}\n", image_get(img, 0, 0))

	image_modify(img, proc(x, y: uint, pixel: PixelRgbAlpha) -> PixelRgbAlpha {
		if pixel.alpha != 255 { return pixel }
		xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
		network_propogate(&network, [?]f32{ xnorm, ynorm })
		//fmt.printf("output for {},{} = {}\n", x, y, network.layer_output.nodes)
		return {
			u8(network.layer_output.nodes[0]*255),
			u8(network.layer_output.nodes[1]*255),
			u8(network.layer_output.nodes[2]*255),
			255,
		}
	})

	image_write_png(img, "output.png")

	upscaled_img, err := image_new(.rgb, 1024, 1024)
	assert(err == nil)

	for y in 0..<upscaled_img.height {
		for x in 0..<upscaled_img.width {
			network_propogate(&network, [?]f32{
				f32(x)/f32(upscaled_img.width),
				f32(y)/f32(upscaled_img.height),
			})

			image_set(upscaled_img, x, y, PixelRgb {
				u8(network.layer_output.nodes[0]*255),
				u8(network.layer_output.nodes[1]*255),
				u8(network.layer_output.nodes[2]*255),
			})
		}
	}

	image_write_png(upscaled_img, "upscaled.png")

	extended_inner_size :: 256
	extended_total_size :: extended_inner_size*7
	extension_size :: (extended_total_size-extended_inner_size)/2
	extended_data := new([extended_total_size*extended_total_size*3]u8)

	min_x: f32 = 0
	max_x: f32 = 0
	for y in 0..<extended_total_size {
		for x in 0..<extended_total_size {
			if (
				(x - extension_size == -1 && -1 <= y - extension_size && y - extension_size <= extended_inner_size) ||
				(x - extension_size == extended_inner_size && -1 <= y - extension_size && y - extension_size <= extended_inner_size) ||
				(y - extension_size == -1 && -1 <= x - extension_size && x - extension_size <= extended_inner_size) ||
				(y - extension_size == extended_inner_size && -1 <= x - extension_size && x - extension_size <= extended_inner_size)
			) {
				extended_data[3*(y*extended_total_size + x) + 0] = 0
				extended_data[3*(y*extended_total_size + x) + 1] = 0
				extended_data[3*(y*extended_total_size + x) + 2] = 0

				continue
			}

			coords := [?]f32{
				f32(x - extension_size) / f32(extended_inner_size),
				f32(y - extension_size) / f32(extended_inner_size),
			}
			min_x = math.min(min_x, coords[0])
			max_x = math.max(max_x, coords[0])
			network_propogate(&network, coords)

			extended_data[3*(y*extended_total_size + x) + 0] = u8(network.layer_output.nodes[0]*255)
			extended_data[3*(y*extended_total_size + x) + 1] = u8(network.layer_output.nodes[1]*255)
			extended_data[3*(y*extended_total_size + x) + 2] = u8(network.layer_output.nodes[2]*255)
		}
	}

	fmt.printf("For extended, x is in [{}, {}]\n", min_x, max_x)

	stbi_write_png("extended.png", extended_total_size, extended_total_size, 3, extended_data, 3*extended_total_size)

	/*
	error_graph_height :: 1024
	error_graph_data := new([error_graph_height*generations*3]u8)

	line_extend :: 3

	for y in 0..<error_graph_height {
		curr_height := error_graph_height - y - 1
		for x in 0..<generations {
			idx := (y*generations + x)*3

			error := error_history[x]
			error_height := int(f32(error_graph_height)*error/error_max)
			error_min_height := error_height - line_extend
			error_max_height := error_height + line_extend

			eta := eta_history[x]
			eta_height := int(f32(error_graph_height)*eta/eta_max)
			eta_min_height := eta_height - line_extend
			eta_max_height := eta_height + line_extend
			eta_changed := x != 0 && eta != eta_history[x-1]

			if error_min_height <= curr_height && curr_height <= error_max_height {
				error_graph_data[idx + 0] = 0
				error_graph_data[idx + 1] = 0
				error_graph_data[idx + 2] = 0
			} else if eta_min_height <= curr_height && curr_height <= eta_max_height {
				error_graph_data[idx + 0] = 0 if !eta_changed else 255
				error_graph_data[idx + 1] = 0
				error_graph_data[idx + 2] = 255
			} else {
				error_graph_data[idx + 0] = 255
				error_graph_data[idx + 1] = 255 if !eta_changed else 0
				error_graph_data[idx + 2] = 255 if !eta_changed else 0
			}
		}
	}

	stbi_write_png("error_graph.png", generations, error_graph_height, 3, error_graph_data, generations*3)
	*/
}
