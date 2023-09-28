def trampoline(bouncer):
    while callable(bouncer):
        bouncer = bouncer()
    return bouncer

def fibonacci_cps(n, cont):
    if n <= 1:
        return cont(n)
    else:
        return lambda: fibonacci_cps(n-1, lambda val1: lambda: fibonacci_cps(n-2, lambda val2: cont(val1 + val2)))

# To use with trampoline:
result = trampoline(fibonacci_cps(46, lambda x: x))
print(result)