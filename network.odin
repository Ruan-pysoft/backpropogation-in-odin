package backprop

import "base:intrinsics"
import "base:runtime"
import "core:math/rand"
import "core:mem"

FloatType :: f32
rand_float :: proc(gen := context.random_generator) -> FloatType {
	return rand.float32(gen)*2 - 1
}

Layer :: struct {
	size: uint,
	activations: [^]FloatType,
	partials: [^]FloatType,
}

layer_new :: proc(size: uint) -> (layer: Layer, err: runtime.Allocator_Error) {
	activations, partials: rawptr

	activations, err = mem.alloc(int(size_of(FloatType)*size), align_of(FloatType))
	if err != nil do return {}, err

	partials, err = mem.alloc(int(size_of(FloatType)*size), align_of(FloatType))
	if err != nil {
		free(activations)
		return {}, err
	}

	return {
		size,
		auto_cast activations,
		auto_cast partials,
	}, nil
}
layer_delete :: proc(layer: ^Layer) {
	layer.size = 0
	free(layer.activations)
	free(layer.partials)
}

Parameters :: struct {
	size, prev_size: uint,
	weights: Matrix(FloatType),
	biases: [^]FloatType,
}

params_new :: proc(size, prev_size: uint) -> (params: Parameters, err: runtime.Allocator_Error) {
	weights: Matrix(FloatType)
	biases: rawptr

	weights, err = mat_zero(FloatType, size, prev_size)
	if err != nil do return {}, err

	biases, err = mem.alloc(int(size_of(FloatType)*size), align_of(FloatType))
	if err != nil {
		mat_delete(&weights)
		return {}, err
	}

	return {
		size, prev_size,
		weights,
		auto_cast biases,
	}, nil
}
params_delete :: proc(params: ^Parameters) {
	params.size = 0
	params.prev_size = 0
	mat_delete(&params.weights)
	free(params.biases)
	params.biases = nil
}

params_init_random :: proc(params: Parameters, gen := context.random_generator) {
	for i in 0..<params.size {
		for j in 0..<params.prev_size {
			mat_set(params.weights, i, j, rand_float(gen))
		}
		params.biases[i] = rand_float(gen)
	}
}

SimpleNetwork :: struct($Layers: uint)
/*where Layers > 1*/ {
	layers: /*#soa*/[Layers]Layer,
	params: /*#soa*/[Layers-1]Parameters,
	act_funcs: /*#soa*/[Layers-1]BasicActFunc,
	loss_func: BasicLossFunc,
}

simple_net_new :: proc($Layers: uint, topology: [Layers]uint, act_funcs: [Layers-1]BasicActFunc, loss_func: BasicLossFunc) -> (net: SimpleNetwork(Layers), err: runtime.Allocator_Error)
where Layers > 1 {
	for i in 0..<Layers {
		assert(topology[i] > 0)

		net.layers[i], err = layer_new(topology[i])
		if err != nil {
			for j in 0..<i {
				layer_delete(&net.layers[j])
			}
		}
	}

	for i in 0..<Layers-1 {
		net.params[i], err = params_new(topology[i+1], topology[i])
		if err != nil {
			for j in 0..<Layers {
				layer_delete(&net.layers[j])
			}
			for j in 0..<i {
				params_delete(&net.params[j])
			}
		}
	}

	net.act_funcs = act_funcs
	net.loss_func = loss_func

	return net, nil
}
simple_net_delete :: proc(net: ^SimpleNetwork($Layers)) {
	for i in 0..<Layers {
		layer_delete(&net.layers[i])
	}
	for i in 0..<Layers-1 {
		params_delete(&net.params[i])
	}
}

simple_net_init_random :: proc(net: SimpleNetwork($Layers), gen := context.random_generator) {
	for i in 0..<Layers-1 {
		params_init_random(net.params[i])
	}
}

@(private="file")
simple_net_layer_propogate :: proc(layer, prev_layer: Layer, params: Parameters, act_func: BasicActFunc) {
	assert(layer.size == params.size)
	assert(prev_layer.size == params.prev_size)

	for i in 0..<params.size {
		acc: FloatType = params.biases[i]

		for j in 0..<params.prev_size {
			acc += prev_layer.activations[j]*mat_get(params.weights, i, j)
		}

		layer.activations[i] = act_func.func(acc)
		layer.partials[i] = act_func.dfunc(acc)
	}
}
simple_net_propogate :: proc(net: SimpleNetwork($Layers), input: []FloatType) {
	assert(uint(len(input)) == net.layers[0].size)
	for i in 0..<net.layers[0].size {
		net.layers[0].activations[i] = input[i]
	}

	#unroll for i in 1..<Layers {
		simple_net_layer_propogate(net.layers[i], net.layers[i-1], net.params[i-1], net.act_funcs[i-1])
	}
}

@(private="file")
simple_net_layer_backprop_immediate :: proc(layer, prev_layer: Layer, params: Parameters, output_partials: [dynamic]FloatType, eta: FloatType) -> (input_partials: [dynamic]FloatType) {
	// Also I realise the comments are cryptic,
	// but I basically just implemented wikipedia's algorithm while I was zonked out on sleep deprivations
	// so I'm sitting here with a piece of paper deriving how all this works
	// and then writing a comment to document it somewhat
	// ...unfortunately I can't include a diagram in the comments

	// NOTATION:
	//   L       :: the loss function
	//   f       :: the layer's activation function
	//   f'      :: the derivative of the layer's activation function
	//   y_i     :: the output of node i
	//     y     :: the output of the current node
	//   x_j     :: the output of node j on the previous layer
	//   w_ji    :: the weight from node j on the previous layer to node i
	//     w_j   :: the weight from node j on the previous layer to the current node
	//   b_i     :: the bias of node i
	//     b     :: the bias of the current node
	//   act_i   :: the input to the activation function for node i
	//     act   :: the input to the activation function for the current node
	//   (dA/dB) :: the partial derivative of A with respect to B

	// NOTE:
	//   act = x_1w_1 + x_2w_2 + ... + x_nw_n + b
	//   y = f(act)

	// PARAMETERS:
	//   layer, prev_layer :: self-explanatory
	//   params :: list of weights and biases between prev_layer and layer
	//   output_partials :: list of (dL/dy)
	//   eta :: the learning rate
	// RETURN VALUE:
	//   input_partials :: list of (dL/dx), becomes (dL/dy) for the previous layer

	// In this function:
	// For each weight and each bias, we find the partial derivative of the loss function with respect to that weight/bias.
	// This gives the direction of change for that weight/bias to increase the loss function as fast as possible.
	// To decrease the loss function as fast as possible, we must adjust it in the opposite direction, scaled by the learning rate.
	// So we are to find (dL/dw) and (dL/db)

	// Finding (dL/dw):
	//   (dL/dw) = (dL/dy)(dy/dw)
	//           = (dL/dy)(df(act)/dw)
	//           = (dL/dy)(df(act)/dact)(dact/dw)
	//           = (dL/dy)f'(act)x = (dL/dy)(xf'(act))
	// Finding (dL/db):
	//   (dL/db) = (dL/dy)(dy/db)
	//           = (dL/dy)(df(act)/db)
	//           = (dL/dy)(df(act)/dact)(dact/db)
	//           = (dL/dy)f'(act)1 = (dL/dy)f'(act)
	// Finding (dL/dx):
	//   (dL/dx) = Sum{i} (dL/dy_i)(dy_i/dx)
	//           = Sum{i} (dL/dy_i)(df(act_i)/dx)
	//           = Sum{i} (dL/dy_i)(df(act_i)/dact_i)(dact_i/dx)
	//           = Sum{i} (dL/dy_i)f'(act_i)w
	//           = Sum{i} (dL/dy_i)(wf'(act_i))

	assert(layer.size == params.size)
	assert(prev_layer.size == params.prev_size)
	assert(uint(len(output_partials)) == layer.size)

	input_partials = make([dynamic]FloatType, prev_layer.size)

	for i in 0..<params.size {
		// d = (dL/dy)f'(act)
		delta := output_partials[i] * layer.partials[i]

		for j in 0..<params.prev_size {
			// (dL/dx)_j = d * w
			input_partials[j] += delta*mat_get(params.weights, i, j)

			// (dL/dw) = d * x
			weight_partial := delta*prev_layer.activations[j]

			mat_set(params.weights, i, j,
				mat_get(params.weights, i, j) - eta*weight_partial
			)
		}

		// (dL/db) = d
		params.biases[i] += -eta*delta
	}

	return input_partials
}
simple_net_backprop_immediate :: proc(net: SimpleNetwork($Layers), expected: []FloatType, eta: FloatType) {
	assert(uint(len(expected)) == net.layers[Layers-1].size)

	transferred_partials: [dynamic]FloatType

	reserve(&transferred_partials, net.layers[Layers-1].size)
	for i in 0..<net.layers[Layers-1].size {
		append(&transferred_partials, net.loss_func.dfunc(
			expected[i],
			net.layers[Layers-1].activations[i]
		))
	}

	// for i := Layers-1; i > 0; i -= 1 {
	#unroll for rev_i in 1..<Layers {
		i := Layers-rev_i
		partials := simple_net_layer_backprop_immediate(net.layers[i], net.layers[i-1], net.params[i-1], transferred_partials, eta)
		delete(transferred_partials)
		transferred_partials = partials
	}

	delete(transferred_partials)
}

SimpleNetworkGradient :: struct($Layers: uint) {
	gradients: /*#soa*/[Layers-1]Parameters,
	samples_count: int,
}

simple_net_grad_new :: proc(for_net: SimpleNetwork($Layers)) -> (grad: SimpleNetworkGradient(Layers), err: runtime.Allocator_Error) {
	for i in 0..<Layers-1 {
		grad.gradients[i], err = params_new(for_net.params[i].size, for_net.params[i].prev_size)
		if err != nil {
			for j in 0..<i {
				params_delete(&grad.gradients[j])
			}
		}
	}

	return grad, nil
}
simple_net_grad_delete :: proc(grad: ^SimpleNetworkGradient($Layers)) {
	for i in 0..<Layers-1 {
		params_delete(&grad.gradients[i])
	}
}
simple_net_grad_compute_actual :: proc(grad: ^SimpleNetworkGradient($Layers)) {
	assert(grad.samples_count > 0)

	for i in 0..<Layers-1 {
		for r in 0..<grad.gradients[i].weights.rows {
			for c in 0..<grad.gradients[i].weights.cols {
				mat_set(
					grad.gradients[i].weights, r, c,
					mat_get(grad.gradients[i].weights, r, c) / FloatType(grad.samples_count),
				)
			}
		}
		for j in 0..<grad.gradients[i].size {
			grad.gradients[i].biases[j] /= FloatType(grad.samples_count)
		}
	}

	grad.samples_count = -1
}

@(private="file")
simple_net_layer_backprop_partial :: proc(layer, prev_layer: Layer, params: Parameters, output_partials: [dynamic]FloatType, eta: FloatType, params_gradient: ^Parameters) -> (input_partials: [dynamic]FloatType) {
	// see comments in simple_net_layer_backprop_immediate

	assert(layer.size == params.size)
	assert(prev_layer.size == params.prev_size)
	assert(uint(len(output_partials)) == layer.size)
	assert(params_gradient.size == params.size)
	assert(params_gradient.prev_size == params.prev_size)

	input_partials = make([dynamic]FloatType, prev_layer.size)

	for i in 0..<params.size {
		// d = (dL/dy)f'(act)
		delta := output_partials[i] * layer.partials[i]

		for j in 0..<params.prev_size {
			// (dL/dx)_j = d * w
			input_partials[j] += delta*mat_get(params.weights, i, j)

			// (dL/dw) = d * x
			weight_partial := delta*prev_layer.activations[j]

			mat_set(params_gradient.weights, i, j,
				mat_get(params_gradient.weights, i, j) + eta*weight_partial
			)
		}

		// (dL/db) = d
		params_gradient.biases[i] += eta*delta
	}

	return input_partials
}
simple_net_backprop_partial :: proc(net: SimpleNetwork($Layers), expected: []FloatType, eta: FloatType, gradient: ^SimpleNetworkGradient(Layers)) {
	assert(uint(len(expected)) == net.layers[Layers-1].size)
	assert(gradient.samples_count >= 0)

	gradient.samples_count += 1

	transferred_partials: [dynamic]FloatType

	reserve(&transferred_partials, net.layers[Layers-1].size)
	for i in 0..<net.layers[Layers-1].size {
		append(&transferred_partials, net.loss_func.dfunc(
			expected[i],
			net.layers[Layers-1].activations[i]
		))
	}

	// for i := Layers-1; i > 0; i -= 1 {
	#unroll for rev_i in 1..<Layers {
		i := Layers-rev_i
		partials := simple_net_layer_backprop_partial(net.layers[i], net.layers[i-1], net.params[i-1], transferred_partials, eta, &gradient.gradients[i-1])
		delete(transferred_partials)
		transferred_partials = partials
	}

	delete(transferred_partials)
}
TrainingDataPoint :: struct($InputSize, $OutputSize: uint) {
	X: [InputSize]FloatType,
	Y: [OutputSize]FloatType,
}
import "core:fmt"
simple_net_backprop :: proc(net: SimpleNetwork($Layers), training_set: []TrainingDataPoint($InputSize, $OutputSize), eta: FloatType, $ComputeError: bool) -> (avg_err: FloatType) {
	assert(InputSize == net.layers[0].size)
	assert(OutputSize == net.layers[Layers-1].size)

	gradient, err := simple_net_grad_new(net)
	if err != nil {
		fmt.panicf("Failed allocating gradient structure during backpropogation: {}", err)
	}
	defer simple_net_grad_delete(&gradient)

	for &point in training_set {
		simple_net_propogate(net, point.X[:])
		when ComputeError do avg_err += simple_net_get_error(net, point.Y[:])
		simple_net_backprop_partial(net, point.Y[:], eta, &gradient)
	}

	when ComputeError do avg_err /= FloatType(len(training_set))

	simple_net_grad_compute_actual(&gradient)

	for i in 0..<Layers-1 {
		mat_sub(net.params[i].weights, net.params[i].weights, gradient.gradients[i].weights)

		for j in 0..<net.params[i].size {
			net.params[i].biases[j] -= gradient.gradients[i].biases[j]
		}
	}

	return
}

simple_net_get_error :: proc(net: SimpleNetwork($Layers), expected: []FloatType) -> FloatType {
	assert(uint(len(expected)) == net.layers[Layers-1].size)

	error: FloatType = 0

	for i in 0..<net.layers[Layers-1].size {
		error += net.loss_func.func(
			expected[i],
			net.layers[Layers-1].activations[i],
		)
	}

	return error
}
