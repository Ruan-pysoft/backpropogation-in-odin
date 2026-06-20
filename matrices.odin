package backprop

import "base:intrinsics"
import "base:runtime"
import "core:mem"

Matrix :: struct($T: typeid) {
	rows, cols: uint,
	values: [^]T
}

mat_zero :: proc($T: typeid, rows, cols: uint) -> (mat: Matrix(T), err: runtime.Allocator_Error) {
	values := mem.alloc(int(size_of(T)*rows*cols), align_of(T)) or_return

	return { rows, cols, auto_cast values }, nil
}
mat_diagonal :: proc($T: typeid, size: uint, value: T) -> (mat: Matrix(T), err: runtime.Allocator_Error) {
	mat = mat_zero(T, size, size) or_return
	for i in 0..<mat.rows {
		mat_set(mat, i, i, value)
	}

	return
}
mat_identity :: proc($T: typeid, size: uint) -> (mat: Matrix(T), err: runtime.Allocator_Error)
where intrinsics.type_is_numeric(T) {
	return mat_diagonal(T, size, 1)
}
mat_row_of :: proc($T: typeid, size: uint, vector: [^]T) -> Matrix(T) {
	return { size, 1, vector }
}
mat_col_of :: proc($T: typeid, size: uint, vector: [^]T) -> Matrix(T) {
	return { 1, size, vector }
}
mat_delete :: proc(mat: ^Matrix($T)) {
	mat.rows = 0
	mat.cols = 0
	free(mat.values)
	mat.values = nil
}

@(private="file")
mat_idx :: #force_inline proc(mat: Matrix($T), row, col: uint) -> uint {
	return row + col*mat.rows
}

mat_get :: #force_inline proc(mat: Matrix($T), row, col: uint) -> T {
	assert(row < mat.rows)
	assert(col < mat.cols)

	return mat.values[mat_idx(mat, row, col)]
}
mat_set :: #force_inline proc(mat: Matrix($T), row, col: uint, val: T) {
	assert(row < mat.rows)
	assert(col < mat.cols)

	mat.values[mat_idx(mat, row, col)] = val
}

mat_add :: proc(res, a, b: Matrix($T)) {
	assert(res.rows == a.rows && a.rows == b.rows)
	assert(res.cols == a.cols && a.cols == b.cols)

	for row in 0..<res.rows {
		for col in 0..<res.cols {
			mat_set(res, row, col,
				mat_get(a, row, col)+mat_get(b, row, col)
			)
		}
	}
}
mat_sub :: proc(res, a, b: Matrix($T)) {
	assert(res.rows == a.rows && a.rows == b.rows)
	assert(res.cols == a.cols && a.cols == b.cols)

	for row in 0..<res.rows {
		for col in 0..<res.cols {
			mat_set(res, row, col,
				mat_get(a, row, col)-mat_get(b, row, col)
			)
		}
	}
}
// R = AB
mat_mul :: proc(res, a, b: Matrix($T)) {
	assert(res.rows == a.rows)
	assert(a.cols == b.rows)
	assert(b.cols == res.cols)

	for row in 0..<res.rows {
		for col in 0..<res.cols {
			val: T = 0
			for i in 0..<a.cols {
				val += mat_get(a, row, i)*mat_get(b, i, col)
			}
			mat_set(res, row, col, val)
		}
	}
}
