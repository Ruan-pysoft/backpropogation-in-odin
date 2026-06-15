package backprop

import "core:c"
import "core:fmt"
import "core:strings"

when ODIN_OS == .Windows do foreign import "stb_image.lib"
when ODIN_OS == .Linux do foreign import "stb_image.a"

StbiFormat :: enum c.int {
	default = 1,
	grey = 1,
	grey_alpha = 2,
	rgb = 3,
	rgb_alpha = 4,
}

StbiIoCallbacks :: struct #packed {
	read: ^proc "c" (user: rawptr, data: [^]c.char, size: c.int) -> c.int,
	skip: ^proc "c" (user: rawptr, n: c.int),
	eof:  ^proc "c" (user: rawptr) -> c.int,
}

StbiImage :: [^]c.uchar

foreign stb_image {
	stbi_load_from_memory :: proc(buffer: ^/*const*/ c.uchar, len: c.int, x, y, channels_in_file: ^c.int, desired_channels: StbiFormat) -> StbiImage ---
	stbi_load_from_callbacks :: proc(clbk: ^/*const*/ StbiIoCallbacks, user: rawptr, x, y, channels_in_file: ^c.int, desired_channels: StbiFormat) -> StbiImage ---

	stbi_load :: proc(filename: cstring, x, y, channels_in_file: ^c.int, desired_channels: StbiFormat) -> StbiImage ---
	stbi_load_from_file :: proc(f: ^c.FILE, x, y, channels_in_file: ^c.int, desired_channels: StbiFormat) -> StbiImage ---
	// for stbi_load_from_file, file pointer is left pointing immediately after image

	stbi_load_gif_from_memory :: proc(buffer: ^/*const*/ c.uchar, len: c.int, delays: ^^c.int, x, y, z, comp: ^int, req_comp: int) -> StbiImage ---

	// ...

	stbi_failure_reason :: proc() -> cstring ---

	stbi_image_free :: proc(retval_from_stbi_load: StbiImage) ---

	stbi_info_from_memory :: proc(buffer: ^/*const*/ c.uchar, len: c.int, x, y, comp: ^c.int) -> c.int ---
	stbi_info_from_callbacks :: proc(clbk: ^/*const*/ StbiIoCallbacks, user: rawptr, x, y, comp: ^c.int) -> c.int ---

	stbi_info :: proc(filename: cstring, x, y, comp: ^c.int) -> c.int ---
	stbi_info_from_file :: proc(f: ^c.FILE, x, y, comp: ^c.int) -> c.int ---
}

Image :: struct($PixelFormat: StbiFormat) {
	valid: bool,
	width, height: uint,
	raw_data: StbiImage,
}

PixelGrey :: u8
PixelGreyAlpha :: struct #packed { grey, alpha: u8 }
PixelRgb :: struct #packed { r, g, b: u8 }
PixelRgbAlpha :: struct #packed { r, g, b, alpha: u8 }

image_load :: proc(file: string, $format: StbiFormat) -> Image(format) {
	w, h, c: c.int = 0, 0, 0
	img_raw := stbi_load(
		strings.clone_to_cstring(file, context.temp_allocator),
		&w, &h, &c, format
	)
	if (img_raw == nil) {
		return {
			valid = false,
			width = 0,
			height = 0,
			raw_data = nil,
		}
	}
	return {
		valid = true,
		width = uint(w),
		height = uint(h),
		raw_data = img_raw,
	}
}

image_free :: proc(img: ^Image($format)) {
	if (img.valid) {
		stbi_image_free(img.raw_data)
	}
	img.valid = false
}

image_get_grey :: proc(img: Image(.grey), x, y: uint) -> PixelGrey {
	assert(img.valid)
	assert(x < img.width)
	assert(y < img.height)

	return img.raw_data[x + y*img.width]
}
image_get_grey_alpha :: proc(img: Image(.grey_alpha), x, y: uint) -> PixelGreyAlpha {
	assert(img.valid)
	assert(x < img.width)
	assert(y < img.height)

	g := img.raw_data[2*(x + y*img.width) + 0]
	a := img.raw_data[2*(x + y*img.width) + 1]

	return { g, a }
}
image_get_rgb :: proc(img: Image(.rgb), x, y: uint) -> PixelRgb {
	assert(img.valid)
	assert(x < img.width)
	assert(y < img.height)

	r := img.raw_data[3*(x + y*img.width) + 0]
	g := img.raw_data[3*(x + y*img.width) + 1]
	b := img.raw_data[3*(x + y*img.width) + 2]

	return { r, g, b }
}
image_get_rgb_alpha :: proc(img: Image(.rgb_alpha), x, y: uint) -> PixelRgbAlpha {
	assert(img.valid)
	assert(x < img.width)
	assert(y < img.height)

	r := img.raw_data[4*(x + y*img.width) + 0]
	g := img.raw_data[4*(x + y*img.width) + 1]
	b := img.raw_data[4*(x + y*img.width) + 2]
	a := img.raw_data[4*(x + y*img.width) + 3]

	return { r, g, b, a }
}

image_get :: proc{
	image_get_grey,
	image_get_grey_alpha,
	image_get_rgb,
	image_get_rgb_alpha,
}
