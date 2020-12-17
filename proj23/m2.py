# for x in range(100000000000):
file_path="enwik9/enwik9"
sets = [set() for i in range(5)]
# s = set()
with open(file_path, 'r', encoding="utf-8") as fd:
    i = 0
    gi = 0
    while gi < 5:
        fd.seek(0)
        chunk_size = gi + 1
        while len(chunk := fd.read(chunk_size)) > 0:
            sets[gi].add(chunk)
            i += 1
        gi += 1
# print(d)
print(sets)
print(f'LEN: {[len(s) for s in sets]} with {i} iterations')