# for x in range(100000000000):
import shelve
from collections import Counter
from tqdm import tqdm

file_path="enwik9/enwik9"

big_array = []
with open(file_path, 'r', encoding="utf-8") as fd:
    chunk_size = 1000
    i = 0
    while len(chunk := fd.read(chunk_size)) > 0:
        # if i == 10000:
        #     break
        big_array.extend(chunk)
        i += 1
# print(d)
# print(f'LEN:{len(big_array)} with {i} iterations')

# counters = [Counter() for x in range(5)]
# print('here')
result = []
word_len = 1
step_size = 1

print('Starting')
chars = set()
for x in range(0, len(big_array) - 1, step_size):
    chars.update([''.join(big_array[0+x:1+x])])
    if (x % 1000) == 0:
        print(list(chars)[-10:])
result.extend(chars)
print(f"C:{x}/{word_len - 1} LEN: {len(chars)} iter: {x}")
with open('result2.txt', 'w', encoding="utf-8") as fd:
    for r in result:
        fd.write(str(r) + '\n')
    # breakpoint()

# for x in range(len(ba) - 5):
# ...     c1.update([''.join(ba[0+x:5+x])])
# print(result[0:100])
# with open('result2.txt', 'w', encoding="utf-8") as fd:
#     for r in result:
#         fd.write(str(r) + '\n')