package backprop

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:math"
import "core:math/rand"

main :: proc() {
	Prefix :: "/home/ruan/Documents/texturepack/"
	BlockPrefix :: Prefix + "mysimpleresourcepack/assets/minecraft/textures/block/"
	AiBlockPrefix :: Prefix + "ai_upscaled_pack/assets/minecraft/textures/block/"

	fmt.println("=== Birch Door ===")
	upscale_texture(Prefix + "birch_door.png", Prefix + "birch_door_big.png", { 64, 128 }, wraparound = false)
	fmt.println("=== Birch Leaves ===")
	upscale_texture(BlockPrefix + "birch_leaves.png", AiBlockPrefix + "birch_leaves.png", { 64, 64 })
	fmt.println("=== Birch Log ===")
	upscale_texture(BlockPrefix + "birch_log.png", AiBlockPrefix + "birch_log.png", { 64, 64 })
	fmt.println("=== Birch Log Top ===")
	upscale_texture(BlockPrefix + "birch_log_top.png", AiBlockPrefix + "birch_log_top.png", { 64, 64 })
	fmt.println("=== Birch Log Planks ===")
	upscale_texture(BlockPrefix + "birch_planks.png", AiBlockPrefix + "birch_planks.png", { 64, 64 })
	fmt.println("=== Birch Sapling ===")
	upscale_texture(BlockPrefix + "birch_sapling.png", AiBlockPrefix + "birch_sapling.png", { 64, 64 }, wraparound = false)
	fmt.println("=== Birch Trapdoor ===")
	upscale_texture(BlockPrefix + "birch_trapdoor.png", AiBlockPrefix + "birch_trapdoor.png", { 64, 64 }, wraparound = false)
	fmt.println("=== Brown Mushroom Block ===")
	upscale_texture(BlockPrefix + "brown_mushroom_block.png", AiBlockPrefix + "brown_mushroom_block.png", { 64, 64 })
	fmt.println("=== Brown Mushroom ===")
	upscale_texture(BlockPrefix + "brown_mushroom.png", AiBlockPrefix + "brown_mushroom.png", { 64, 64 }, wraparound = false)
	fmt.println("=== Coarse Dirt ===")
	upscale_texture(BlockPrefix + "coarse_dirt.png", AiBlockPrefix + "coarse_dirt.png", { 64, 64 })
	fmt.println("=== Dirt Path Side ===")
	upscale_texture(BlockPrefix + "dirt_path_side.png", AiBlockPrefix + "dirt_path_side.png", { 64, 64 }, wraparound = false)
	fmt.println("=== Dirt ===")
	upscale_texture(BlockPrefix + "dirt.png", AiBlockPrefix + "dirt.png", { 64, 64 })
	fmt.println("=== Grass Block Side Overlay ===")
	upscale_texture(BlockPrefix + "grass_block_side_overlay.png", AiBlockPrefix + "grass_block_side_overlay.png", { 64, 64 }, wraparound = false)
	fmt.println("=== Grass Block Side ===")
	upscale_texture(BlockPrefix + "grass_block_side.png", AiBlockPrefix + "grass_block_side.png", { 64, 64 }, wraparound = false)
	fmt.println("=== Grass Block Top ===")
	upscale_texture(BlockPrefix + "grass_block_top.png", AiBlockPrefix + "grass_block_top.png", { 64, 64 })
	fmt.println("=== Mushroom Block Inside ===")
	upscale_texture(BlockPrefix + "mushroom_block_inside.png", AiBlockPrefix + "mushroom_block_inside.png", { 64, 64 })
	fmt.println("=== Mushroom Stem ===")
	upscale_texture(BlockPrefix + "mushroom_stem.png", AiBlockPrefix + "mushroom_stem.png", { 64, 64 })
	fmt.println("=== Mycelium Side ===")
	upscale_texture(BlockPrefix + "mycelium_side.png", AiBlockPrefix + "mycelium_side.png", { 64, 64 })
	fmt.println("=== Mycelium Top ===")
	upscale_texture(BlockPrefix + "mycelium_top.png", AiBlockPrefix + "mycelium_top.png", { 64, 64 })
	fmt.println("=== Podzol Side ===")
	upscale_texture(BlockPrefix + "podzol_side.png", AiBlockPrefix + "podzol_side.png", { 64, 64 }, wraparound = false)
	fmt.println("=== Podzol Top ===")
	upscale_texture(BlockPrefix + "podzol_top.png", AiBlockPrefix + "podzol_top.png", { 64, 64 })
	fmt.println("=== Red Mushroom Block ===")
	upscale_texture(BlockPrefix + "red_mushroom_block.png", AiBlockPrefix + "red_mushroom_block.png", { 64, 64 })
	fmt.println("=== Red Mushroom ===")
	upscale_texture(BlockPrefix + "red_mushroom.png", AiBlockPrefix + "red_mushroom.png", { 64, 64 }, wraparound = false)
	fmt.println("=== Rooted Dirt ===")
	upscale_texture(BlockPrefix + "rooted_dirt.png", AiBlockPrefix + "rooted_dirt.png", { 64, 64 })
	fmt.println("=== Stripped Birch Log ===")
	upscale_texture(BlockPrefix + "stripped_birch_log.png", AiBlockPrefix + "stripped_birch_log.png", { 64, 64 })
	fmt.println("=== Stripped Birch Log Top ===")
	upscale_texture(BlockPrefix + "stripped_birch_log_top.png", AiBlockPrefix + "stripped_birch_log_top.png", { 64, 64 })
}

normalize_coords :: proc(coords, size: [2]$T) -> [2]f32 {
	fsize := cast([2]f32)size
	return (cast([2]f32)coords)/fsize + 1/(2*fsize)
}
denormalize_coords :: proc(coords: [2]f32, size: [2]$T) -> [2]T {
	fsize := cast([2]f32)size
	return cast([2]T)( (coords - 1/(2*fsize)) * fsize )
}

upscale_texture :: proc(image: string, output_to := "output.png", target_size := [2]uint{ 64, 64 }, wraparound := true) {
	fmt.printf("{} -> {} ({})\n", image, output_to, target_size)
	ok: bool
	img_grey: Image(.grey)
	img_grey_alpha: Image(.grey_alpha)
	img_rgb: Image(.rgb)
	img_rgb_alpha: Image(.rgb_alpha)
	format: StbiFormat
	img_grey, img_grey_alpha, img_rgb, img_rgb_alpha, format, ok = image_load_scuffed(image)
	if !ok do fmt.panicf("Failed loading image! {}", stbi_failure_reason())
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
		upscaled := texture_upscaler(img_grey, target_size.x, target_size.y, wraparound)
		image_write_png(upscaled, output_to)
	case .grey_alpha:
		upscaled := texture_upscaler(img_grey_alpha, target_size.x, target_size.y, wraparound)
		image_write_png(upscaled, output_to)
	case .rgb:
		upscaled := texture_upscaler(img_rgb, target_size.x, target_size.y, wraparound)
		image_write_png(upscaled, output_to)
	case .rgb_alpha:
		upscaled := texture_upscaler(img_rgb_alpha, target_size.x, target_size.y, wraparound)
		image_write_png(upscaled, output_to)
	}
}

texture_upscaler :: proc(texture: Image($format), target_w, target_h: uint, wraparound: bool) -> Image(format) {
	IsGrey :: format == .grey || format == .grey_alpha
	IsAlpha :: format == .grey_alpha || format == .rgb_alpha
	skip_cutout := true

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
			normed_xy := normalize_coords([2]uint{ x, y }, [2]uint{ texture.width, texture.height })

			when IsAlpha {
				idx := y*texture.width + x

				when IsGrey {
					training_set_cutout[idx] = {
						normed_xy,
						[?]f32{ f32(pixel[1]) / 256 },
					}

					if pixel[1] == 0 {
						skip_cutout = false
						continue
					}
				} else {
					training_set_cutout[idx] = {
						normed_xy,
						[?]f32{ f32(pixel.a) / 256 },
					}

					if pixel.a == 0 {
						skip_cutout = false
						continue
					}
				}
			}

			when IsGrey {
				append(&training_set, TrainingDataPoint(2, OutputSize) {
					normed_xy,
					[?]f32{ f32(pixel[0])/256, },
				})
			} else {
				append(&training_set, TrainingDataPoint(2, OutputSize) {
					normed_xy,
					[?]f32{
						f32(pixel.r)/256,
						f32(pixel.g)/256,
						f32(pixel.b)/256,
					},
				})
			}
		}
	}

	wraparound_points: [dynamic]TrainingDataPoint(2, 2)
	wraparound_data: [dynamic]TrainingDataPoint(2, OutputSize)
	skip_wa_gens: int = 0
	if wraparound {
		size := 2*(target_w+1 + target_h+1) + 2*(texture.width+1 + texture.height+1)

		reserve(&wraparound_points, size)
		wraparound_data = make([dynamic]TrainingDataPoint(2, OutputSize), size)

		append(&wraparound_points, TrainingDataPoint(2, 2) {
			[?]f32{ 0, 0 },
			[?]f32{ 1, 1 },
		})
		append(&wraparound_points, TrainingDataPoint(2, 2) {
			[?]f32{ 0, 1 },
			[?]f32{ 1, 0 },
		})
		append(&wraparound_points, TrainingDataPoint(2, 2) {
			[?]f32{ 1, 0 },
			[?]f32{ 0, 1 },
		})
		append(&wraparound_points, TrainingDataPoint(2, 2) {
			[?]f32{ 1, 1 },
			[?]f32{ 0, 0 },
		})

		boundary_size := [2]int{ int(texture.width), int(texture.height) }

		append(&wraparound_points, TrainingDataPoint(2, 2) {
			normalize_coords([2]int{ -1, -1 }, boundary_size),
			normalize_coords([2]int{ int(texture.width-1), int(texture.height-1) }, boundary_size),
		})
		append(&wraparound_points, TrainingDataPoint(2, 2) {
			normalize_coords([2]int{ -1, int(texture.height-1) }, boundary_size),
			normalize_coords([2]int{ int(texture.width-1), -1 }, boundary_size),
		})
		append(&wraparound_points, TrainingDataPoint(2, 2) {
			normalize_coords([2]int{ int(texture.width-1), -1 }, boundary_size),
			normalize_coords([2]int{ -1, int(texture.height-1) }, boundary_size),
		})
		append(&wraparound_points, TrainingDataPoint(2, 2) {
			normalize_coords([2]int{ int(texture.width-1), int(texture.height-1) }, boundary_size),
			normalize_coords([2]int{ -1, -1 }, boundary_size),
		})

		for x in 1..=target_w-1 {
			norm_x := normalize_coords([2]int{ int(x), 0 }, [2]int{ int(target_w), 1 }).x
			append(&wraparound_points, TrainingDataPoint(2, 2) {
				[?]f32{ norm_x, 0 },
				[?]f32{ norm_x, 1 },
			})
			append(&wraparound_points, TrainingDataPoint(2, 2) {
				[?]f32{ norm_x, 1 },
				[?]f32{ norm_x, 0 },
			})
		}
		for y in 1..=target_h-1 {
			norm_y := normalize_coords([2]int{ 0, int(y) }, [2]int{ 1, int(target_h) }).y
			append(&wraparound_points, TrainingDataPoint(2, 2) {
				[?]f32{ 0, norm_y },
				[?]f32{ 1, norm_y },
			})
			append(&wraparound_points, TrainingDataPoint(2, 2) {
				[?]f32{ 1, norm_y },
				[?]f32{ 0, norm_y },
			})
		}

		for x in 0..<texture.width {
			normed_above := normalize_coords([2]int{ int(x), int(boundary_size.y+1) }, boundary_size)
			normed_below := normalize_coords([2]int{ int(x), -1 }, boundary_size)
			append(&wraparound_points, TrainingDataPoint(2, 2) {
				normed_above,
				[2]f32{ normed_above.x, normed_above.y - 1 }
			})
			append(&wraparound_points, TrainingDataPoint(2, 2) {
				normed_below,
				[2]f32{ normed_below.x, normed_below.y + 1 }
			})
		}
		for y in 0..<texture.width {
			normed_right := normalize_coords([2]int{ int(boundary_size.x+1), int(y) }, boundary_size)
			normed_left := normalize_coords([2]int{ -1, int(y) }, boundary_size)
			append(&wraparound_points, TrainingDataPoint(2, 2) {
				normed_right,
				[2]f32{ normed_right.x - 1, normed_right.y }
			})
			append(&wraparound_points, TrainingDataPoint(2, 2) {
				normed_left,
				[2]f32{ normed_left.x + 1, normed_left.y }
			})
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

		error: f32

		error = simple_net_backprop(net_texture, training_set[:], eta, true)

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

		if (g+1) % 10 == 0 && wraparound && skip_wa_gens <= 0 {
			for &point, i in wraparound_points {
				simple_net_propogate(net_texture, point.Y[:])
				Y := net_cutout.layers[7].activations[:net_cutout.layers[7].size]
				wraparound_data[i].X = point.X
				#unroll for j in 0..<OutputSize {
					wraparound_data[i].Y[j] = Y[j]
				}
			}

			error = simple_net_backprop(net_texture, wraparound_data[:], eta, true)
			if error < 1/1024. {
				fmt.printf("WA err = {}; skipping...\n", error)
				skip_wa_gens = 1024
			}

			if (g+1) % print_every == 0 {
				fmt.printf("Wraparound error: {}\n", error)
			}
		} else if skip_wa_gens > 0 do skip_wa_gens -= 1
	}

	when IsAlpha {
		if !skip_cutout {
			fmt.println("Training alpha cutout...")

			early_error_avg = 0

			for g in 0..<generations/10 {
				if g+1 == 1 {
					eta = 1/8.0
				}

				error: f32

				error = simple_net_backprop(net_cutout, training_set_cutout[:], eta, true)

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

				if (g+1) % (print_every/10) == 0 {
					fmt.printf("After {} generations of training, avg error is: {}\n", g+1, error)
				}

				if (g+1) % 10 == 0 && wraparound && skip_wa_gens <= 0 {
					for &point, i in wraparound_points {
						simple_net_propogate(net_cutout, point.Y[:])
						Y := net_cutout.layers[7].activations[:net_cutout.layers[7].size]
						wraparound_data[i].X = point.X
						#unroll for j in 0..<OutputSize {
							wraparound_data[i].Y[j] = Y[j]
						}
					}

					error = simple_net_backprop(net_cutout, wraparound_data[:], eta, true)
					if error < 1/1024. {
						fmt.printf("WA err = {}; skipping...\n", error)
						skip_wa_gens = 128
					}

					if (g+1) % (print_every/10) == 0 {
						fmt.printf("Wraparound error: {}\n", error)
					}
				} else if skip_wa_gens > 0 do skip_wa_gens -= 1
			}
		}
	}

	upscaled_img, err := image_new(format, target_w, target_h)
	assert(err == nil)

	for y in 0..<upscaled_img.height {
		for x in 0..<upscaled_img.width {
			normed_xy := normalize_coords([2]uint{ x, y }, [2]uint{ upscaled_img.width, upscaled_img.height })
			
			simple_net_propogate(net_texture, normed_xy[:])
			when IsAlpha {
				simple_net_propogate(net_cutout, normed_xy[:])
			}

			when IsGrey {
				grey := u8(net_texture.layers[7].activations[0]*256)
			} else {
				r := u8(net_texture.layers[7].activations[0]*256)
				g := u8(net_texture.layers[7].activations[1]*256)
				b := u8(net_texture.layers[7].activations[2]*256)
			}
			alpha := u8(net_cutout.layers[7].activations[0]*256) if !skip_cutout else 255

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
