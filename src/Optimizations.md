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

9 - Frame reuse in TCO

Finally addressed frame reutilization in TCO. Turns out this is just a flag on the stack frame.
Down to 2.2ms.

10 - Empty Closure Optimization

During Off-enum closure work, we ended up creating an ever-expanding closure array that for every function call needed a new closure in this array.
Turns out most of the time we need empty closures, so we preallocate them and ref them.

No performance improvements, but some benchmarks now run without memory issues.
