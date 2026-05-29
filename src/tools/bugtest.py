import mlx.core as mx

a = mx.random.normal((64, 512)).astype(mx.bfloat16)
b = mx.random.normal((512, 512)).astype(mx.bfloat16)
mx.eval(a, b)
with mx.stream(mx.gpu):
    g = a @ b
    mx.eval(g)
with mx.stream(mx.cpu):
    c = a @ b
    mx.eval(c)
g = g.astype(mx.float32)
c = c.astype(mx.float32)
mx.eval(g, c)
rd = float(((g - c) ** 2).sum() ** 0.5) / (float((c**2).sum() ** 0.5) + 1e-9)
if rd < 0.1:
    print("OK")
else:
    print("Buggy")
