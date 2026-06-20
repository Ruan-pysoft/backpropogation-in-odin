package backprop

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:math"
import "core:math/rand"

input_size :: 2
output_size :: 3

NetworkLayers :: 8
network: SimpleNetwork(NetworkLayers)
img: Image(.rgb_alpha)

main :: proc() {
	ok: bool
	alloc_err: runtime.Allocator_Error

	rand.reset(42)

	network, alloc_err = simple_net_new(
		Layers = 8,
		topology = [?]uint {
			input_size,
			8, 4, 16, 32, 8, 4,
			output_size,
		},
		act_funcs = [?]BasicActFunc {
			sin_act,
			logistic_act,
			sin_act,
			logistic_act,
			tanh_act,
			tanh_act,
			logistic_act,
		},
		loss_func = quad_loss,
	)
	if alloc_err != nil {
		fmt.printf("Error allocating neural network: {}\n", alloc_err)
		return
	}
	defer simple_net_delete(&network)

	simple_net_init_random(network)

	eta: FloatType = 0.0625

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
				if pixel.a != 255 { continue }

				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				simple_net_propogate(network, []f32{ xnorm, ynorm })

				error += simple_net_get_error(network, []f32 {
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
			eta = 1/16.0
		}
		if g+1 == 5_000 {
			eta = 1/32.0
		}
		/*if g+1 == 8_000 {
			eta = 3/64.0
		}*/
		if g+1 == 9_000 {
			eta -= eta/4
		}
		if g+1 == 9_500 {
			eta = 1/64.0
		}
		if g+1 == 10_500 {
			eta = 1/64.0
		}
		if g+1 == 20_000 {
			eta = 1/128.0
		}
		if g+1 == 50_000 {
			eta = 1/256.0
		}
		if g+1 == 75_000 {
			eta = 1/512.0
		}

		eta_history[g] = eta
		eta_max = math.max(eta_max, eta)

		for y in 0..<img.height {
			for x in 0..<img.width {
				pixel := image_get(img, x, y)
				if pixel.a != 255 { continue }

				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				simple_net_propogate(network, []f32{ xnorm, ynorm })

				simple_net_backprop(network, []f32 {
					f32(pixel.r)/255,
					f32(pixel.g)/255,
					f32(pixel.b)/255,
				}, eta)
			}
		}

		error: f32 = 0

		for y in 0..<img.height {
			for x in 0..<img.width {
				pixel := image_get(img, x, y)
				if pixel.a != 255 { continue }

				xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
				simple_net_propogate(network, []f32{ xnorm, ynorm })

				error += simple_net_get_error(network, []f32 {
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
		if pixel.a != 255 { return pixel }
		xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)
		simple_net_propogate(network, []f32{ xnorm, ynorm })
		//fmt.printf("output for {},{} = {}\n", x, y, network.layer_output.nodes)
		return {
			u8(network.layers[NetworkLayers-1].activations[0]*255),
			u8(network.layers[NetworkLayers-1].activations[1]*255),
			u8(network.layers[NetworkLayers-1].activations[2]*255),
			255,
		}
	})

	image_write_png(img, "output.png")

	upscaled_img, err := image_new(.rgb, 1024, 1024)
	assert(err == nil)

	for y in 0..<upscaled_img.height {
		for x in 0..<upscaled_img.width {
			simple_net_propogate(network, []f32{
				f32(x)/f32(upscaled_img.width),
				f32(y)/f32(upscaled_img.height),
			})

			image_set(upscaled_img, x, y, PixelRgb {
				u8(network.layers[NetworkLayers-1].activations[0]*255),
				u8(network.layers[NetworkLayers-1].activations[1]*255),
				u8(network.layers[NetworkLayers-1].activations[2]*255),
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
			simple_net_propogate(network, coords[:])

			extended_data[3*(y*extended_total_size + x) + 0] = u8(network.layers[NetworkLayers-1].activations[0]*255)
			extended_data[3*(y*extended_total_size + x) + 1] = u8(network.layers[NetworkLayers-1].activations[1]*255)
			extended_data[3*(y*extended_total_size + x) + 2] = u8(network.layers[NetworkLayers-1].activations[2]*255)
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
