package backprop

import "core:c"
import "core:fmt"

main :: proc() {
	fmt.println("Hello, world!")

	img := image_load("GoblinFace.jpg", .rgb)
	defer image_free(&img)
	if (!img.valid) {
		fmt.println(stbi_failure_reason())
		return
	}

	fmt.printf("{}x{}\n", img.width, img.height)
	fmt.printf("Top-left pixel: {}\n", image_get(img, 0, 0))

	image_modify(img, proc(x, y: uint, pixel: PixelRgb) -> PixelRgb {
		return { pixel.g, pixel.b, pixel.r }
	})
	image_set(img, 0, 0, PixelRgb { 255, 0, 0 })

	image_write_png(img, "output.png")
}
