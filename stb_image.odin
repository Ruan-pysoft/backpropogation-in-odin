package backprop

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:mem"
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

	// stb_image_write

	stbi_write_png :: proc(filename: cstring, w, h, comp: c.int, data: /*const*/ rawptr, stride_in_bytes: c.int) -> c.int ---
	stbi_write_bmp :: proc(filename: cstring, w, h, comp: c.int, data: /*const*/ rawptr, stride_in_bytes: c.int) -> c.int ---
	stbi_write_tga :: proc(filename: cstring, w, h, comp: c.int, data: /*const*/ rawptr, stride_in_bytes: c.int) -> c.int ---
	stbi_write_jpg :: proc(filename: cstring, w, h, comp: c.int, data: /*const*/ rawptr, stride_in_bytes: c.int, quality: c.int) -> c.int ---
	stbi_write_hdr :: proc(filename: cstring, w, h, comp: c.int, data: /*const*/ rawptr, stride_in_bytes: c.float) -> c.int ---

	stbi_flip_vertically_on_write :: proc(flag: c.bool) ---

	// ...
}

Image :: struct($PixelFormat: StbiFormat) {
	width, height: uint,
	allocation_source: enum { stbi, odin },
	raw_data: StbiImage,
}

PixelGrey :: struct #packed { grey: u8 }
PixelGreyAlpha :: struct #packed { grey, alpha: u8 }
PixelRgb :: struct #packed { r, g, b: u8 }
PixelRgbAlpha :: struct #packed { r, g, b, alpha: u8 }

image_new :: proc($format: StbiFormat, w, h: uint) -> (img: Image(format), err: runtime.Allocator_Error) {
	raw_data := mem.alloc(int(size_of(u8)*w*h*uint(format)), align_of(u8)) or_return

	return {
		width = w,
		height = h,
		allocation_source = .odin,
		raw_data = auto_cast raw_data,
	}, nil
}

image_load :: proc(file: string, $format: StbiFormat) -> (img: Image(format), ok: bool) {
	w, h, c: c.int = 0, 0, 0
	img_raw := stbi_load(
		strings.clone_to_cstring(file, context.temp_allocator),
		&w, &h, &c, format
	)
	if (img_raw == nil) {
		return { }, false
	}
	return {
		width = uint(w),
		height = uint(h),
		allocation_source = .stbi,
		raw_data = img_raw,
	}, true
}

image_free :: proc(img: ^Image($format)) {
	assert(img != nil)
	if img.raw_data != nil {
		switch img.allocation_source {
		case .stbi:
			stbi_image_free(img.raw_data)
		case .odin:
			free(img.raw_data)
		}
	}
	img^ = {}
}

image_get_grey :: proc(img: Image(.grey), x, y: uint) -> PixelGrey {
	assert(img.raw_data != nil)
	assert(x < img.width)
	assert(y < img.height)

	return { img.raw_data[x + y*img.width] }
}
image_get_grey_alpha :: proc(img: Image(.grey_alpha), x, y: uint) -> PixelGreyAlpha {
	assert(img.raw_data != nil)
	assert(x < img.width)
	assert(y < img.height)

	g := img.raw_data[2*(x + y*img.width) + 0]
	a := img.raw_data[2*(x + y*img.width) + 1]

	return { g, a }
}
image_get_rgb :: proc(img: Image(.rgb), x, y: uint) -> PixelRgb {
	assert(img.raw_data != nil)
	assert(x < img.width)
	assert(y < img.height)

	r := img.raw_data[3*(x + y*img.width) + 0]
	g := img.raw_data[3*(x + y*img.width) + 1]
	b := img.raw_data[3*(x + y*img.width) + 2]

	return { r, g, b }
}
image_get_rgb_alpha :: proc(img: Image(.rgb_alpha), x, y: uint) -> PixelRgbAlpha {
	assert(img.raw_data != nil)
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

image_set_grey :: proc(img: Image(.grey), x, y: uint, to: PixelGrey) {
	assert(img.raw_data != nil)
	assert(x < img.width)
	assert(y < img.height)

	img.raw_data[x + y*img.width] = to.grey
}
image_set_grey_alpha :: proc(img: Image(.grey_alpha), x, y: uint, to: PixelGreyAlpha) {
	assert(img.raw_data != nil)
	assert(x < img.width)
	assert(y < img.height)

	img.raw_data[2*(x + y*img.width) + 0] = to.grey
	img.raw_data[2*(x + y*img.width) + 1] = to.alpha
}
image_set_rgb :: proc(img: Image(.rgb), x, y: uint, to: PixelRgb) {
	assert(img.raw_data != nil)
	assert(x < img.width)
	assert(y < img.height)

	img.raw_data[3*(x + y*img.width) + 0] = to.r
	img.raw_data[3*(x + y*img.width) + 1] = to.g
	img.raw_data[3*(x + y*img.width) + 2] = to.b
}
image_set_rgb_alpha :: proc(img: Image(.rgb_alpha), x, y: uint, to: PixelRgbAlpha) {
	assert(img.raw_data != nil)
	assert(x < img.width)
	assert(y < img.height)

	img.raw_data[4*(x + y*img.width) + 0] = to.r
	img.raw_data[4*(x + y*img.width) + 1] = to.g
	img.raw_data[4*(x + y*img.width) + 2] = to.b
	img.raw_data[4*(x + y*img.width) + 3] = to.alpha
}

image_set :: proc{
	image_set_grey,
	image_set_grey_alpha,
	image_set_rgb,
	image_set_rgb_alpha,
}

// These are here to avoid the overhead of two rounds of asserts
// Probably unnecessary...

PixelGreyCallback :: proc(x, y: uint, pixel: PixelGrey) -> PixelGrey
PixelGreyAlphaCallback :: proc(x, y: uint, pixel: PixelGreyAlpha) -> PixelGreyAlpha
PixelRgbCallback :: proc(x, y: uint, pixel: PixelRgb) -> PixelRgb
PixelRgbAlphaCallback :: proc(x, y: uint, pixel: PixelRgbAlpha) -> PixelRgbAlpha

image_modify_grey :: proc(img: Image(.grey), cb: PixelGreyCallback) {
	assert(img.raw_data != nil)

	for y in 0..<img.height {
		for x in 0..<img.width {
			idx := x + y*img.width

			img.raw_data[idx] = cb(x, y, { img.raw_data[idx] }).grey
		}
	}
}
image_modify_grey_alpha :: proc(img: Image(.grey_alpha), cb: PixelGreyAlphaCallback) {
	assert(img.raw_data != nil)

	for y in 0..<img.height {
		for x in 0..<img.width {
			idx := 2*(x + y*img.width)

			pixel := cb(x, y, {
				img.raw_data[idx + 0],
				img.raw_data[idx + 1],
			})

			img.raw_data[idx + 0] = pixel.grey
			img.raw_data[idx + 1] = pixel.alpha
		}
	}
}
image_modify_rgb :: proc(img: Image(.rgb), cb: PixelRgbCallback) {
	assert(img.raw_data != nil)

	for y in 0..<img.height {
		for x in 0..<img.width {
			idx := 3*(x + y*img.width)

			pixel := cb(x, y, {
				img.raw_data[idx + 0],
				img.raw_data[idx + 1],
				img.raw_data[idx + 2],
			})

			img.raw_data[idx + 0] = pixel.r
			img.raw_data[idx + 1] = pixel.g
			img.raw_data[idx + 2] = pixel.b
		}
	}
}
image_modify_rgb_alpha :: proc(img: Image(.rgb_alpha), cb: PixelRgbAlphaCallback) {
	assert(img.raw_data != nil)

	for y in 0..<img.height {
		for x in 0..<img.width {
			idx := 4*(x + y*img.width)

			pixel := cb(x, y, {
				img.raw_data[idx + 0],
				img.raw_data[idx + 1],
				img.raw_data[idx + 2],
				img.raw_data[idx + 3],
			})

			img.raw_data[idx + 0] = pixel.r
			img.raw_data[idx + 1] = pixel.g
			img.raw_data[idx + 2] = pixel.b
			img.raw_data[idx + 3] = pixel.alpha
		}
	}
}

image_modify :: proc{
	image_modify_grey,
	image_modify_grey_alpha,
	image_modify_rgb,
	image_modify_rgb_alpha,
}

image_write_png :: proc(img: Image($format), filename: string) {
	stbi_write_png(
		strings.clone_to_cstring(filename, context.temp_allocator),
		c.int(img.width),
		c.int(img.height),
		c.int(format),
		img.raw_data,
		c.int(img.width * uint(format)),
	)
}
image_write_bmp :: proc(img: Image($format), filename: string) {
	stbi_write_bmp(
		strings.clone_to_cstring(filename, context.temp_allocator),
		c.int(img.width),
		c.int(img.height),
		c.int(format),
		img.raw_data,
		c.int(img.width * uint(format)),
	)
}
image_write_tga :: proc(img: Image($format), filename: string) {
	stbi_write_tga(
		strings.clone_to_cstring(filename, context.temp_allocator),
		c.int(img.width),
		c.int(img.height),
		c.int(format),
		img.raw_data,
		c.int(img.width * uint(format)),
	)
}
image_write_jpg :: proc(img: Image($format), filename: string, quality: uint) {
	stbi_write_jpg(
		strings.clone_to_cstring(filename, context.temp_allocator),
		c.int(img.width),
		c.int(img.height),
		c.int(format),
		img.raw_data,
		c.int(img.width * uint(format)),
		c.int(quality),
	)
}
