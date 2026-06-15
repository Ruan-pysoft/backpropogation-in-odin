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
}
