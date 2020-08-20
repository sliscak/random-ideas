from array import array

genexpr = (x for x in range(-10, 10) if x <= 0)
print(list(genexpr))

class Iterator:
    def __init__(self, value, max_iter):
        self.value = value
        self.count = 0
        self.max_iter = max_iter

    def __iter__(self):
        # THIS RETURNS AN ITERATOR (an object that implements __next__ dunder method.)
        return self

    def __next__(self):
        if self.count >= self.max_iter:
            raise StopIteration
        self.count += 1
        return self.value

iterator = Iterator('ALFA', 10)
for item in iterator:
    print(item)

opcodes = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'div': lambda x, y: x / y
}

result = opcodes.get('add', None)(1,2)
print(result)
print(type(result))
print(list(opcodes.keys()))

values = 18, 15, 3
print(f'VALUES: {values}')
for code in opcodes.keys():
    try:
        result = opcodes.get(code)(*values)
        print(f'CODE: {code}-> result: {result}')
    except Exception as e:
        print(e)

# pipeline??
integers = range(-10, 10+1)
squares = (x**2 for x in integers)
absolute_values = (abs(x) for x in squares)
greater_than_six = (x for x in absolute_values if x > 6)
print(list(greater_than_six))

# result = list((for x in range(-10, 10)))
result = zip(range(-10, 10), range(0, 20))
print(result)
result = [sum(z) for z in result]
print(list(result))

# def summer(r1, r2):
#     r1 = iter(r1)
#     r2 = iter(r2)
#     # while rr1 != StopIteration and rr2 != StopIteration:
#     while True:
#         print(next(r1))
#         yield next(r1) + next(r2)
#
# result = summer(range(0, 3), range(0, 3))
# print(list(result))

# r = iter(range(0, 10))
# print(dir(r))

class Indenter:
    def print(self, text):
        string = ''.join(['\t' for x in range(self.count)]) + text
        print(string)
    def __nix__(self):
        pass
    def __init__(self):
        self.count = 1
    def __enter__(self):
        self.count += 1
        print('entered!')
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.count -= 1
        print('exited!')

with Indenter() as indent:
    print(dir(indent))
    indent.print('Ich will!')
    with indent:
        indent.print('Ich will')
    # print()

indent = Indenter()
print(indent.print.__code__)
from dis import dis
dis(indent.print)
