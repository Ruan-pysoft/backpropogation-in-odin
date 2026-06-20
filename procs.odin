package backprop

import "base:intrinsics"
import "core:math"

DifferentiableFunction :: struct($P: typeid)
where intrinsics.type_is_proc(P) {
	func, dfunc: P
}

BasicActFunc :: DifferentiableFunction(proc(f32) -> f32)
BasicLossFunc :: DifferentiableFunction(proc(y, o: f32) -> f32)

linear_act: BasicActFunc: {
	proc(x: f32) -> f32 { return x },
	proc(f32) -> f32 { return 1 },
}

logistic_act: BasicActFunc: {
	proc(x: f32) -> f32 { return 1 / (1 + math.exp(-x)) },
	proc(x: f32) -> f32 {
		act := logistic_act.func(x)
		return act*(1 - act)
	},
}

sin_act: BasicActFunc: {
	proc(x: f32) -> f32 { return math.sin(x) },
	proc(x: f32) -> f32 { return math.cos(x) },
}

adj_sin_act: BasicActFunc: {
	proc(x: f32) -> f32 { return math.abs(x) * math.sin(math.PI*x / 2) },
	proc(x: f32) -> f32 {
		x := math.abs(x)
		x = math.PI*x / 2
		return math.sin(x) + x*math.cos(x)
	},
}

/* appears I got the derivative wrong...
cutoff_sin_act: BasicActFunc: {
	proc(x: f32) -> f32 {
		if -1 <= x || x <= 1 {
			return math.sin(math.PI*x / 2)
		} else if x < -1 {
			return (x+1)/64 - 1
		} else {
			return (x-1)/64 + 1
		}
	},
	proc(x: f32) -> f32 {
		if -1 <= x || x <= 1 {
			return (math.PI/2)*math.cos(math.PI*x / 2)
		} else {
			return 1/64
		}
	},
}
*/

// TODO: support softmax output layer?

tanh_act: BasicActFunc: {
	proc(x: f32) -> f32 { return math.tanh(x) },
	proc(x: f32) -> f32 {
		t := math.tanh(x)
		return 1 - t*t
	},
}

relu_act: BasicActFunc: {
	proc(x: f32) -> f32 { return math.max(0, x) },
	proc(x: f32) -> f32 { return 1 if x >= 0 else 0 },
}

leaky_relu_act: BasicActFunc: {
	proc(x: f32) -> f32 { return x if x > 0 else x/32 },
	proc(x: f32) -> f32 { return 1 if x > 0 else 1/32 },
}

softplus_act: BasicActFunc: {
	//proc(x: f32) -> f32 { return math.log(1 + math.exp(x)) },
	proc(x: f32) -> f32 { return math.LN2 if x == 0 else x/(1 - math.exp(-x/math.LN2)) },
	proc(x: f32) -> f32 { return 1/(1 + math.exp(-x)) },
}

swish_act: BasicActFunc: {
	proc(x: f32) -> f32 { return x*logistic_act.func(x) },
	proc(x: f32) -> f32 { return logistic_act.func(x) + x*logistic_act.dfunc(x) },
	/*proc(x: f32) -> f32 {
		coshx2 := math.cosh(x/2)
		return (x + math.sinh(x))/(4*coshx2*coshx2) + 1/2
	}*/
}

mish_act: BasicActFunc: {
	proc(x: f32) -> f32 { return x*math.tanh(swish_act.func(x)) },
	proc(x: f32) -> f32 {
		swish_val := swish_act.func(x)
		return (0 +
			tanh_act.func(swish_val) +
			x*tanh_act.dfunc(swish_val)*swish_act.dfunc(x) +
		0)
	}
}

quad_loss: BasicLossFunc: {
	proc(y, o: f32) -> f32 { return (o-y)*(o-y) },
	proc(y, o: f32) -> f32 { return 2*(o-y) },
}

mad_loss: BasicLossFunc: {
	proc(y, o: f32) -> f32 { return math.abs(o-y) },
	proc(y, o: f32) -> f32 { return -1 if o-y < 0 else 1 },
}
