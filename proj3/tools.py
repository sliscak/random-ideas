# the key must meet certain criteria
# based on https://arxiv.org/abs/2004.06278
def squares_rng(ctr, key):
    y = x = ctr * key
    z = y + key
    x = (x*x) + y
    x = (x>>32) | (x << 32)
    x = x*x + z
    x = (x>>32) | (x << 32)
    x = (x*x) + y
    x = (x >> 32) | (x << 32)
    x = x * x + z
    x = (x >> 32) | (x << 32)
    x = (x * x) + y
    x = (x >> 32) | (x << 32)
    return ((x*x) + z) >> 32

# ctr = 0
# key = 1024
# while True:
#     r = squares_rng(ctr, key)
#     r = str(r)[-4:-1]
#     print(r)
#     print('#'*10)
#     ctr += 1

array = [
    'a',
    'b',
]