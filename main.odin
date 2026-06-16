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

input_size :: 2
hidden_size :: 4
output_size :: 3

Network :: struct {
	layer_input: [input_size]f32, // x, y
	layer_hidden: [hidden_size]f32,
	layer_output: [output_size]f32, // r, g, b

	weights_hidden: matrix[input_size, hidden_size]f32,
	weights_output: matrix[hidden_size, output_size]f32,

	eta: f32,
}

network_new :: proc() -> (result: Network) {
	result = {
		[input_size]f32{},
		[hidden_size]f32{},
		[output_size]f32{},

		matrix[input_size, hidden_size]f32{},
		matrix[hidden_size, output_size]f32{},

		0.01,
	}

	for row in 0..<input_size {
		for col in 0..<hidden_size {
			result.weights_hidden[row, col] = rand.float32()*2 - 1
		}
	}
	for row in 0..<hidden_size {
		for col in 0..<output_size {
			result.weights_output[row, col] = rand.float32()*2 - 1
		}
	}

	return
}

network_permute :: proc(n: ^Network) {
	for idx_h in 0..<hidden_size {
		acc: f32 = 0
		for idx_i in 0..<input_size {
			acc += n.layer_input[idx_i]*n.weights_hidden[idx_i, idx_h]
		}
		n.layer_hidden[idx_h] = activation(acc)
	}

	for idx_o in 0..<output_size {
		acc: f32 = 0
		for idx_h in 0..<hidden_size {
			acc += n.layer_hidden[idx_h]*n.weights_output[idx_h, idx_o]
		}
		n.layer_output[idx_o] = activation(acc)
	}
}

network_error :: proc(n: Network, expected: [output_size]f32) -> (error: f32) {
	error = 0

	for i in 0..<output_size {
		error += loss(expected[i], n.layer_output[i])
	}

	return
}

network_backprop :: proc(n: ^Network, expected: [output_size]f32) {
	deltas := [output_size]f32{}

	for idx_o in 0..<output_size {
		net: f32 = 0
		for idx_h in 0..<hidden_size {
			net += n.layer_hidden[idx_h]*n.weights_output[idx_h, idx_o]
		}
		delta := dloss(expected[idx_o], n.layer_output[idx_o]) * dactivation(net)
		deltas[idx_o] = delta
		for idx_h in 0..<hidden_size {
			o := n.layer_hidden[idx_h]
			change := -n.eta * o * delta

			n.weights_output[idx_h, idx_o] += change
		}
	}

	for idx_h in 0..<hidden_size {
		net: f32 = 0
		for idx_i in 0..<input_size {
			net += n.layer_input[idx_i]*n.weights_hidden[idx_i, idx_h]
		}
		next_deltas_sum: f32 = 0
		for idx_o in 0..<output_size {
			next_deltas_sum += n.weights_output[idx_h, idx_o]*deltas[idx_o]
		}
		delta := next_deltas_sum * dactivation(net)
		for idx_i in 0..<input_size {
			o := n.layer_input[idx_i]
			change := -n.eta * o * delta

			n.weights_hidden[idx_i, idx_h] += change
		}
	}
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

	for g in 0..<generations {
		for y in 0..<img.height {
			for x in 0..<img.width {
				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				network.layer_input = [?]f32{ xnorm, ynorm }
				network_permute(&network)

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
					network.layer_input = [?]f32{ xnorm, ynorm }
					network_permute(&network)

					pixel := image_get(img, x, y)

					error += network_error(network, [output_size]f32 {
						f32(pixel.r)/255,
						f32(pixel.g)/255,
						f32(pixel.b)/255,
					})
				}
			}

			fmt.printf("After {} generations of training, error is: {}\n", g+1, error)
		}
	}

	fmt.printf("{}x{}\n", img.width, img.height)
	fmt.printf("Top-left pixel: {}\n", image_get(img, 0, 0))

	image_modify(img, proc(x, y: uint, pixel: PixelRgb) -> PixelRgb {
		xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
		network.layer_input = [?]f32{ xnorm, ynorm }
		network_permute(&network)
		return {
			u8(network.layer_output[0]*255),
			u8(network.layer_output[1]*255),
			u8(network.layer_output[2]*255),
		}
	})

	image_write_jpg(img, "output.jpg", 75)
}
