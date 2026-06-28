package backprop

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:math"
import "core:math/rand"

main :: proc() {
	upscale_texture()
}

upscale_texture :: proc() {
	ImageFile :: "my_eye.png"
	TargetSize :: [?]uint { 64, 64 }

	ok: bool
	img_grey: Image(.grey)
	img_grey_alpha: Image(.grey_alpha)
	img_rgb: Image(.rgb)
	img_rgb_alpha: Image(.rgb_alpha)
	format: StbiFormat
	img_grey, img_grey_alpha, img_rgb, img_rgb_alpha, format, ok = image_load_scuffed(ImageFile)
	if !ok do fmt.panicf("Failed loading image!")
	defer {
		image_free(&img_grey)
		image_free(&img_grey_alpha)
		image_free(&img_rgb)
		image_free(&img_rgb_alpha)
	}

	fmt.printf("Image format: {}\n", format)

	// NOTE: probably a logic error here; probably even a greyscale, fully opaque png image would have format .rgb_alpha

	switch format {
	case .grey:
		upscaled := texture_upscaler(img_grey, TargetSize.x, TargetSize.y)
		image_write_png(upscaled, "output.png")
	case .grey_alpha:
		upscaled := texture_upscaler(img_grey_alpha, TargetSize.x, TargetSize.y)
		image_write_png(upscaled, "output.png")
	case .rgb:
		upscaled := texture_upscaler(img_rgb, TargetSize.x, TargetSize.y)
		image_write_png(upscaled, "output.png")
	case .rgb_alpha:
		upscaled := texture_upscaler(img_rgb_alpha, TargetSize.x, TargetSize.y)
		image_write_png(upscaled, "output.png")
	}
}

texture_upscaler :: proc(texture: Image($format), target_w, target_h: uint) -> Image(format) {
	IsGrey :: format == .grey || format == .grey_alpha
	IsAlpha :: format == .grey_alpha || format == .rgb_alpha

	eta: FloatType
	generations :: 100_000
	print_every :: 2_500

	net_texture, net_cutout: SimpleNetwork(8)
	alloc_err: runtime.Allocator_Error

	rand.reset(42)

	when IsGrey {
		OutputSize :: 1
	} else {
		OutputSize :: 3
	}

	training_set: [dynamic]TrainingDataPoint(2, OutputSize)
	reserve(&training_set, texture.width*texture.height)
	when IsAlpha {
		training_set_cutout := make([]TrainingDataPoint(2, 1), texture.width*texture.height)
	}

	for y in 0..<texture.height {
		for x in 0..<texture.width {
			pixel := image_get(texture, x, y)
			// TODO: figure out how this should be best calculated
			xnorm, ynorm := f32(x) / f32(texture.width-1), f32(y) / f32(texture.height-1)

			when IsAlpha {
				idx := y*texture.width + x

				when IsGrey {
					training_set_cutout[idx] = {
						[?]f32{ xnorm, ynorm },
						[?]f32{ f32(pixel[1]) / 255 },
					}

					if pixel[1] == 0 do continue
				} else {
					training_set_cutout[idx] = {
						[?]f32{ xnorm, ynorm },
						[?]f32{ f32(pixel.a) / 255 },
					}

					if pixel.a == 0 do continue
				}
			}

			when IsGrey {
				append(&training_set, TrainingDataPoint(2, OutputSize) {
					[?]f32{ xnorm, ynorm },
					[?]f32{ f32(pixel[0])/255, },
				})
			} else {
				append(&training_set, TrainingDataPoint(2, OutputSize) {
					[?]f32{ xnorm, ynorm },
					[?]f32{
						f32(pixel.r)/255,
						f32(pixel.g)/255,
						f32(pixel.b)/255,
					},
				})
			}
		}
	}

	net_texture, alloc_err = simple_net_new(
		Layers = 8,
		topology = [?]uint {
			2,
			8, 8, 16, 32, 8, 4,
			OutputSize,
		},
		act_funcs = [?]BasicActFunc {
			sin_act,
			tanh_act,
			adj_sin_act,
			tanh_act,
			tanh_act,
			softplus_act,
			logistic_act,
		},
		loss_func = quad_loss,
	)
	if alloc_err != nil {
		fmt.panicf("Error allocating neural network: {}\n", alloc_err)
	}
	defer simple_net_delete(&net_texture)

	simple_net_init_random(net_texture)

	when format == .grey_alpha || format == .rgb_alpha {
		net_cutout, alloc_err = simple_net_new(
			Layers = 8,
			topology = [?]uint {
				2,
				8, 8, 16, 32, 8, 4,
				1,
			},
			act_funcs = [?]BasicActFunc {
				sin_act,
				tanh_act,
				adj_sin_act,
				tanh_act,
				tanh_act,
				softplus_act,
				logistic_act,
			},
			loss_func = quad_loss,
		)
		if alloc_err != nil {
			fmt.panicf("Error allocating neural network: {}\n", alloc_err)
		}
		defer simple_net_delete(&net_cutout)

		simple_net_init_random(net_cutout)
	}

	early_error_avg: f32 = 0

	for g in 0..<generations {
		if g+1 == 1 {
			eta = 1/8.0
		}

		error := simple_net_backprop(net_texture, training_set[:], eta, true)

		if g+1 > 2_500 && g+1 <= 5_000 {
			early_error_avg += error/2_500
		}
		if eta > 1/16.0 && g+1 > 5_000 && error <= early_error_avg/6 {
			fmt.println("Adjusted eta downwards!")
			eta = 1/16.0
		}
		if eta > 1/64.0 && g+1 > 5_000 && error <= early_error_avg/32 {
			fmt.println("Adjusted eta downwards!")
			eta = 1/64.0
		}

		if (g+1) % print_every == 0 {
			fmt.printf("After {} generations of training, avg error is: {}\n", g+1, error)
		}
	}

	when IsAlpha {
		fmt.println("Training alpha cutout...")

		early_error_avg = 0

		for g in 0..<generations/10 {
			if g+1 == 1 {
				eta = 1/8.0
			}

			error := simple_net_backprop(net_cutout, training_set_cutout[:], eta, true)

			if g+1 > 250 && g+1 <= 500 {
				early_error_avg += error/250
			}
			if eta > 1/16.0 && g+1 > 500 && error <= early_error_avg/6 {
				fmt.println("Adjusted eta downwards!")
				eta = 1/16.0
			}
			if eta > 1/64.0 && g+1 > 500 && error <= early_error_avg/32 {
				fmt.println("Adjusted eta downwards!")
				eta = 1/64.0
			}

			if (g+1) % print_every == 0 {
				fmt.printf("After {} generations of training, avg error is: {}\n", g+1, error)
			}
		}
	}

	upscaled_img, err := image_new(format, target_w, target_h)
	assert(err == nil)

	for y in 0..<upscaled_img.height {
		for x in 0..<upscaled_img.width {
			// TODO: figure out how this should be best calculated
			xnorm, ynorm := f32(x) / f32(upscaled_img.width-1), f32(y) / f32(upscaled_img.height-1)
			
			simple_net_propogate(net_texture, []f32{ xnorm, ynorm })
			when IsAlpha {
				simple_net_propogate(net_cutout, []f32{ xnorm, ynorm })
			}

			// TODO: pixels will never have a value of 255; can we correct for this?
			// I mean, theoretically I can do *256?
			when IsGrey {
				grey := u8(net_texture.layers[7].activations[0]*255)
			} else {
				r := u8(net_texture.layers[7].activations[0]*255)
				g := u8(net_texture.layers[7].activations[1]*255)
				b := u8(net_texture.layers[7].activations[2]*255)
			}
			alpha := u8(net_cutout.layers[7].activations[0]*256)

			when format == .grey {
				pixel := PixelGrey { grey }
			} else when format == .grey_alpha {
				pixel := PixelGreyAlpha { grey, alpha }
			} else when format == .rgb {
				pixel := PixelRgb { r, g, b }
			} else {
				pixel := PixelRgbAlpha { r, g, b, alpha }
			}
			image_set(upscaled_img, x, y, pixel)
		}
	}

	return upscaled_img
}

adder_network_tanh :: proc($Bits: uint)
where Bits <= 8
{
	network: SimpleNetwork(4)
	ok: bool
	alloc_err: runtime.Allocator_Error

	network, alloc_err = simple_net_new(
		Layers = 4,
		topology = [?]uint { Bits*2, Bits*4, Bits*2, Bits+1 },
		act_funcs = [?]BasicActFunc {
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

	eta: FloatType: 1/16.0

	Supremum :: 1<<Bits
	training_set := make([]TrainingDataPoint(Bits*2, Bits+1), Supremum*Supremum)
	for a in 0..<Supremum {
		for b in 0..<Supremum {
			c := a+b

			for bit in 0..<Bits {
				bit_a := (a>>bit)&1
				bit_b := (b>>bit)&1
				bit_c := (c>>bit)&1

				training_set[a*Supremum + b].X[bit] = f32(bit_a)
				training_set[a*Supremum + b].X[bit + Bits] = f32(bit_b)
				training_set[a*Supremum + b].Y[bit] = f32(bit_c)
			}

			bit_c := (c>>Bits)&1
			training_set[a*Supremum + b].Y[Bits] = f32(bit_c)
		}
	}

	generations :: 100_000
	print_every :: 10_000
	min_err: f32 = 0
	max_err: f32 = 0
	avg_err: f32 = 0

	rand.reset(42)

	simple_net_init_random(network)

	for g in 0..<generations {
		error := simple_net_backprop(network, training_set, eta, true)

		switch (g+1)%print_every {
		case 0:
			avg_err /= f32(print_every)
			fmt.printf("Error after gen {}: {}-{} ~{}\n", g+1, min_err, max_err, avg_err)
		case 1:
			min_err = error
			max_err = error
			avg_err = error
		case:
			min_err = min(min_err, error)
			max_err = max(max_err, error)
			avg_err += error
		}
	}

	fmt.println("Results:")
	for &point in training_set {
		simple_net_propogate(network, point.X[:])
		error := simple_net_get_error(network, point.Y[:])
		output := network.layers[3].activations[:Bits]
		carry := network.layers[3].activations[Bits]
		fmt.printf("{} + {} = {} {} (err: {})\n", point.X[:Bits], point.X[Bits:], output, carry, error)
	}
}

adder_network_sin :: proc($Bits: uint)
where Bits <= 8
{
	network: SimpleNetwork(4)
	ok: bool
	alloc_err: runtime.Allocator_Error

	network, alloc_err = simple_net_new(
		Layers = 4,
		topology = [?]uint { Bits*2, Bits*2, Bits, Bits+1 },
		act_funcs = [?]BasicActFunc {
			sin_act,
			sin_act,
			logistic_act,
		},
		loss_func = quad_loss,
	)
	if alloc_err != nil {
		fmt.printf("Error allocating neural network: {}\n", alloc_err)
		return
	}
	defer simple_net_delete(&network)

	eta: FloatType: 1/16.0

	Supremum :: 1<<Bits
	training_set := make([]TrainingDataPoint(Bits*2, Bits+1), Supremum*Supremum)
	for a in 0..<Supremum {
		for b in 0..<Supremum {
			c := a+b

			for bit in 0..<Bits {
				bit_a := (a>>bit)&1
				bit_b := (b>>bit)&1
				bit_c := (c>>bit)&1

				training_set[a*Supremum + b].X[bit] = f32(bit_a)
				training_set[a*Supremum + b].X[bit + Bits] = f32(bit_b)
				training_set[a*Supremum + b].Y[bit] = f32(bit_c)
			}

			bit_c := (c>>Bits)&1
			training_set[a*Supremum + b].Y[Bits] = f32(bit_c)
		}
	}

	generations :: 100_000
	print_every :: 10_000
	min_err: f32 = 0
	max_err: f32 = 0
	avg_err: f32 = 0

	rand.reset(42)

	simple_net_init_random(network)

	for g in 0..<generations {
		error := simple_net_backprop(network, training_set, eta, true)

		switch (g+1)%print_every {
		case 0:
			avg_err /= f32(print_every)
			fmt.printf("Error after gen {}: {}-{} ~{}\n", g+1, min_err, max_err, avg_err)
		case 1:
			min_err = error
			max_err = error
			avg_err = error
		case:
			min_err = min(min_err, error)
			max_err = max(max_err, error)
			avg_err += error
		}
	}

	fmt.println("Results:")
	for &point in training_set {
		simple_net_propogate(network, point.X[:])
		error := simple_net_get_error(network, point.Y[:])
		output := network.layers[3].activations[:Bits]
		carry := network.layers[3].activations[Bits]
		fmt.printf("{} + {} = {} {} (err: {})\n", point.X[:Bits], point.X[Bits:], output, carry, error)
	}
}

xor_network :: proc() {
	network: SimpleNetwork(3)
	ok: bool
	alloc_err: runtime.Allocator_Error

	{
		fmt.println("Traditional network with sigmoid activation and two hidden neurons:")
		network, alloc_err = simple_net_new(
			Layers = 3,
			topology = [?]uint { 2, 2, 1 },
			act_funcs = [?]BasicActFunc {
				logistic_act,
				logistic_act,
			},
			loss_func = quad_loss,
		)
		if alloc_err != nil {
			fmt.printf("Error allocating neural network: {}\n", alloc_err)
			return
		}
		defer simple_net_delete(&network)

		xor_network_train(network)
	}
	{
		fmt.println("Traditional network with tanh activation and two hidden neurons:")
		network, alloc_err = simple_net_new(
			Layers = 3,
			topology = [?]uint { 2, 2, 1 },
			act_funcs = [?]BasicActFunc {
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

		xor_network_train(network)
	}
	{
		fmt.println("Traditional network with relu activation and two hidden neurons:")
		network, alloc_err = simple_net_new(
			Layers = 3,
			topology = [?]uint { 2, 2, 1 },
			act_funcs = [?]BasicActFunc {
				relu_act,
				logistic_act,
			},
			loss_func = quad_loss,
		)
		if alloc_err != nil {
			fmt.printf("Error allocating neural network: {}\n", alloc_err)
			return
		}
		defer simple_net_delete(&network)

		xor_network_train(network)
	}
	{
		fmt.println("Traditional network with softplus activation and two hidden neurons:")
		network, alloc_err = simple_net_new(
			Layers = 3,
			topology = [?]uint { 2, 2, 1 },
			act_funcs = [?]BasicActFunc {
				softplus_act,
				logistic_act,
			},
			loss_func = quad_loss,
		)
		if alloc_err != nil {
			fmt.printf("Error allocating neural network: {}\n", alloc_err)
			return
		}
		defer simple_net_delete(&network)

		xor_network_train(network)
	}
	{
		fmt.println("Traditional network with softplus activation and one hidden neuron:")
		network, alloc_err = simple_net_new(
			Layers = 3,
			topology = [?]uint { 2, 1, 1 },
			act_funcs = [?]BasicActFunc {
				softplus_act,
				logistic_act,
			},
			loss_func = quad_loss,
		)
		if alloc_err != nil {
			fmt.printf("Error allocating neural network: {}\n", alloc_err)
			return
		}
		defer simple_net_delete(&network)

		xor_network_train(network)
	}
	{
		fmt.println("Traditional network with sin activation and two hidden neurons:")
		network, alloc_err = simple_net_new(
			Layers = 3,
			topology = [?]uint { 2, 2, 1 },
			act_funcs = [?]BasicActFunc {
				sin_act,
				logistic_act,
			},
			loss_func = quad_loss,
		)
		if alloc_err != nil {
			fmt.printf("Error allocating neural network: {}\n", alloc_err)
			return
		}
		defer simple_net_delete(&network)

		xor_network_train(network)
	}
	{
		fmt.println("Traditional network with sin activation and one hidden neuron:")
		network, alloc_err = simple_net_new(
			Layers = 3,
			topology = [?]uint { 2, 1, 1 },
			act_funcs = [?]BasicActFunc {
				sin_act,
				logistic_act,
			},
			loss_func = quad_loss,
		)
		if alloc_err != nil {
			fmt.printf("Error allocating neural network: {}\n", alloc_err)
			return
		}
		defer simple_net_delete(&network)

		xor_network_train(network)
	}
}

xor_network_train :: proc(net: SimpleNetwork(3)) {
	eta: FloatType: 1/16.0

	training_set := []TrainingDataPoint(2, 1) {
		{ [2]f32{ 0, 0 }, [1]f32{ 0 } },
		{ [2]f32{ 0, 1 }, [1]f32{ 1 } },
		{ [2]f32{ 1, 0 }, [1]f32{ 1 } },
		{ [2]f32{ 1, 1 }, [1]f32{ 0 } },
	}

	generations :: 100_000
	print_every :: 10_000
	min_err: f32 = 0
	max_err: f32 = 0
	avg_err: f32 = 0

	rand.reset(42)

	simple_net_init_random(net)

	for g in 0..<generations {
		error := simple_net_backprop(net, training_set, eta, true)

		switch (g+1)%print_every {
		case 0:
			avg_err /= f32(print_every)
			fmt.printf("Error after gen {}: {}-{} ~{}\n", g+1, min_err, max_err, avg_err)
		case 1:
			min_err = error
			max_err = error
			avg_err = error
		case:
			min_err = min(min_err, error)
			max_err = max(max_err, error)
			avg_err += error
		}
	}

	fmt.println("Results:")
	for &point in training_set {
		simple_net_propogate(net, point.X[:])
		error := simple_net_get_error(net, point.Y[:])
		fmt.printf("  {} {} -> {} (err: {})\n", point.X[0], point.X[1], net.layers[2].activations[0], error)
	}

	fmt.println()
}

input_size :: 2
output_size :: 3

NetworkLayers :: 8
network: SimpleNetwork(NetworkLayers)
img: Image(.rgb_alpha)

train_and_upscale :: proc() {
	ok: bool
	alloc_err: runtime.Allocator_Error

	rand.reset(42)

	network, alloc_err = simple_net_new(
		Layers = 8,
		topology = [?]uint {
			input_size,
			8, 8, 16, 32, 8, 4,
			output_size,
		},
		act_funcs = [?]BasicActFunc {
			sin_act,
			tanh_act,
			adj_sin_act,
			tanh_act,
			tanh_act,
			softplus_act,
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

	//img, ok = image_load("yogurt.jpeg", .rgb_alpha)
	img, ok = image_load("my_eye.png", .rgb_alpha)
	//img, ok = image_load("grass_block_side.png", .rgb_alpha)
	defer image_free(&img)
	if !ok {
		fmt.println(stbi_failure_reason())
		return
	}

	training_set: [dynamic]TrainingDataPoint(2, 3)

	reserve(&training_set, img.width*img.height)

	for y in 0..<img.height {
		for x in 0..<img.width {
			pixel := image_get(img, x, y)
			if pixel.a != 255 { continue }

			xnorm, ynorm := f32(x) / f32(img.width), f32(y) / f32(img.height)

			rnorm, gnorm, bnorm := f32(pixel.r)/255, f32(pixel.g)/255, f32(pixel.b)/255

			append(&training_set, TrainingDataPoint(2, 3) {
				X = [?]f32{ xnorm, ynorm },
				Y = [?]f32{ rnorm, gnorm, bnorm },
			})
		}
	}

	generations :: 100_000
	print_every :: 2500

	graph_height, graph_length :: 1024, 8192
	graph_bucket_size :: generations/graph_length when generations%graph_length == 0 else 1 + generations/graph_length
	error_max_buckets := [graph_length]FloatType{}
	error_min_buckets := [graph_length]FloatType{}
	error_max := f32(0)
	eta_max_buckets := [graph_length]FloatType{}
	eta_min_buckets := [graph_length]FloatType{}
	eta_changed_buckets := [graph_length]bool{}
	eta_max := f32(0)

	fmt.println("Starting network training!")
	{
		error: f32 = 0

		for &point in training_set {
			simple_net_propogate(network, point.X[:])
			error += simple_net_get_error(network, point.Y[:])
		}

		error /= f32(len(training_set))

		fmt.printf("Before training, avg. error is: {}\n", error)
	}

	early_error_avg: f32 = 0

	for g in 0..<generations {
		prev_eta := eta

		if g+1 == 1 {
			eta = 1/8.0
		}

		error := simple_net_backprop(network, training_set[:], eta, true)

		if g+1 > 2_500 && g+1 <= 5_000 {
			early_error_avg += error/2_500
		}
		if eta > 1/16.0 && g+1 > 5_000 && error <= early_error_avg/6 {
			fmt.println("Adjusted eta downwards!")
			eta = 1/16.0
		}
		if eta > 1/64.0 && g+1 > 5_000 && error <= early_error_avg/32 {
			fmt.println("Adjusted eta downwards!")
			eta = 1/64.0
		}

		graph_idx := g/graph_bucket_size
		prev_graph_idx := (g-1)/graph_bucket_size

		if graph_idx != prev_graph_idx {
			error_max_buckets[graph_idx] = error
			error_min_buckets[graph_idx] = error

			eta_max_buckets[graph_idx] = eta
			eta_min_buckets[graph_idx] = eta
		} else {
			error_max_buckets[graph_idx] = max(
				error_max_buckets[graph_idx],
				error,
			)
			error_min_buckets[graph_idx] = min(
				error_min_buckets[graph_idx],
				error,
			)

			eta_max_buckets[graph_idx] = max(
				eta_max_buckets[graph_idx],
				eta,
			)
			eta_min_buckets[graph_idx] = min(
				eta_min_buckets[graph_idx],
				eta,
			)
		}

		eta_changed_buckets[graph_idx] = eta != prev_eta

		error_max = max(error_max, error)
		eta_max = max(eta_max, eta)

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

	graph, graph_err := image_new(.rgb, graph_length, graph_height)
	assert(graph_err == nil)

	line_extend :: 3

	for y in 0..<graph_height {
		curr_height := graph_height - y - 1
		for x in 0..<graph_length {
			error_min_height := int(f32(graph_height)*error_min_buckets[x]/error_max)
			error_max_height := int(f32(graph_height)*error_max_buckets[x]/error_max)

			eta_avg := (eta_min_buckets[x]+eta_max_buckets[x])/2
			eta_height := int(f32(graph_height)*eta_avg/eta_max)
			eta_min_height := eta_height - line_extend
			eta_max_height := eta_height + line_extend

			if error_min_height <= curr_height && curr_height <= error_max_height {
				image_set(graph, uint(x), uint(y), PixelRgb { 0, 0, 0 })
			} else if eta_min_height <= curr_height && curr_height <= eta_max_height {
				image_set(graph, uint(x), uint(y), PixelRgb {
					0 if !eta_changed_buckets[x] else 255,
					0,
					255,
				})
			} else {
				image_set(graph, uint(x), uint(y), PixelRgb {
					255,
					255 if !eta_changed_buckets[x] else 0,
					255,
				})
			}
		}
	}

	image_write_png(graph, "error_graph.png")
}
