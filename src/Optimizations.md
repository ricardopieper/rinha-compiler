The first attempt was a naive hashmap-based interpreter, where values were just stored inside a hashmap and looked up by string name.

These optimizations are tracked mostly by running ./files/perf.rinha, and there's a Criterion benchmark for it.


Disclaimer: Many things were attempted that actually resulted in a performance hit. This report is just the things that worked.


1 - Use BTreeMap: 
Simply change HashMap by BTreeMap.
43% improvement (28ms) over original (43ms)

2 - Use Vec:

Instead of btreemap, we have a Vec<Value>, and compile Var to an index. The program executor has to allocate vecs big enough to store at least the amount of data.
It's a table of names, since shadowing is allowed it's kinda the same as having mutation....

This resulted in a regression, 35ms. But I hope to optimize further...

3 - Variables of stack instead of stack of variables

Every fcall needs cloning, because the new stack frame might change the vars. What if we create all vars in frame 1, and the new stack frames
just store a new value? then when the frame pops we just pop the variables that the frame pushed. Each variable can evolve separately this way.
If a variable is changed more than once in a single frame, we have to be smart and correctly mutate the variable instead of simply pushing.
allocations. 

This resulted in a 60% improvement in runtime, down to 14ms    

4 - Frame reuse

When you pop a frame, you discard the memory it allocated during its execution. Why don't we reuse these frames and avoid a ton of allocations?

This resulted in a 109% improvement in runtime, down to 7ms

5 - Stack tuples

We are still allocating a whole lot because tuples always make a new box. Why not build new tuple variants for primitive types?

This resulted in a 84% improvement in runtime, down to 3.4ms

6 - Tail recursion

Suppose we have:

    let iter = fn (from, to, call, prev) => {
        if (from < to) {
            let res = call(from);
            iter(from + 1, to, call, res)
        } else {
            prev
        }
    };

What the interpreter does during pre-run time is to transform the function into a trampolined-version:

    let iter = fn (from, to, call, prev) => {
        if (from < to) {
            let res = call(from);
            fn() => { iter(from + 1, to, call, res) }
        } else {
            prev
        }
    };

When we need to materialize the value, we start the trampolining process.

Performance is the same, still needs optimization here... but at least we support deep recursion without stack overflows.

7 - Off-Enum values

If you look the code, you'll see there's a Value enum that holds runtime values. It is produced, returned, copied everywhere. 
To reduce the size of copies of values when functions execute, some values were moved to vecs inside the 
`ExecutionEngine` struct. 24 bytes down to 12. Down to 2.9ms from 3.4ms.

8 - Off-Enum closures and strings

Closures are 8 bytes, and strings are 8 byte pointers.
Same as off-enum values, they were moved into off-value storage inside ExecutionEngine. From 12 to 8 bytes.
Down to 2.5ms from 2.9.

It turns out a small Value really helps with general performance.

9 - Frame reuse in TCO

Finally addressed frame reutilization in TCO. Turns out this is just a flag on the stack frame.
Down to 2.2ms.

10 - Empty Closure Optimization

During Off-enum closure work, we ended up creating an ever-expanding closure array that for every function call needed a new closure in this array.
Turns out most of the time we need empty closures, so we preallocate them and ref them.
No performance improvements, but some benchmarks now run without memory issues.


11 - Small Tuple optimization

For tuples with int values that fit into i16, we don't heap allocate them, instead I created a SmallTuple(i16, i16) variant.
Down to 2.1ms.


12 - BinExpr function call in both sides

This optimization was done for const $op var, var $op const, and now it was done for call $op call. This makes it so one operation does more things in one go,
rather than calling another LambdaFunction for each side.

This had a small perf improvement for fib (2.7%) but nothing on perf.rinha.

13 - Closure fix

Turns out I had a closure problem where values were being loaded from the let bindings, but if that value had been overwritten by another function,
then the closure would load this new value instead of the value the closure was created with.

This had a massive negative impact on performance. With optimizations, we're back into 3.4ms territory.

14 - Minimal function closure

We are copying too much data into the closure. This is going beyond the smallvec's limit of 4 values.
for perf.rinha, there is a function doing 6 value copies, this needs to be fixed.

Once fixed we are back to 2.3ms, for some reason some recent changes (including ones to avoid deep recursion in the compiler itself)
made the fib benchmark 10% slower :(


15 - The actual stack

Since we support shadowing and it's all untyped, and every value has the same size (8 bytes), we could try to simulate an actual call stack,
in every call we allocate N bytes in the stack dynamically using DynStack (or, for now, just good old vecs). 

This could help saving time popping the values from the stack, which currently I think is a slow O(n) operation where N = number of variables.

For every callable we will store the function layout, which is a table containing the variable ID and its position
When we compile it, we just read the stack section and interpret it as a value.

When we call a function, we evaluate the arguments and copy it into their respective positions for the called function.
For closures, we kind of already do this. We build a closure and copy them onto the closure data.

For TCO, we will be reusing the frame, so in those cases it should be a big win.
