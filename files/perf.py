import time

def iter(from_val, to, call, prev):
    if from_val < to:
        res = call(from_val)
        return iter(from_val + 1, to, call, res)
    else:
        return prev

def work(x):
    def work_closure(y):
        xx = x * y
        tupl = (xx, x)
        f, s = tupl
        return f * s

    return iter(0, 800, work_closure, 0)

start_time = time.time()
iteration = iter(0, 180, work, 0)
end_time = time.time()

print(f"Result: {iteration}")
print(f"Time taken: {end_time - start_time} seconds")