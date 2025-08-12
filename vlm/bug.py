import mlx.core as mx

data = mx.array(
    [
        0.0,
        1.1754943508222875e-38,
        -1.401298464324817e-45,
        0.0,
        459367.0,
    ],
    dtype=mx.float32
)

print(f"mx.argim on cpu {mx.argmin(data, stream=mx.cpu)}")
print(f"mx.argim on gpu {mx.argmin(data, stream=mx.gpu)}")

data = mx.array(
    [
        0.0,
        1.401298464324817e-45,
        -1.401298464324817e-45,
        0.0,
    ],
    dtype=mx.float32
)

print(f"mx.argmax on cpu {mx.argmax(data, stream=mx.cpu)}")
print(f"mx.argmax on gpu {mx.argmax(data, stream=mx.gpu)}")
