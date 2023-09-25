1 - Use BTreeMap: 43% improvement (28ms) over original (43ms)
2 - Use Vec:

    Instead of btreemap, we have a Vec<Value>, and compile Var to an index. The program executor has to allocate vecs big enough to store at least the amount of data.
    It's a table of names, since shadowing is allowed it's kinda the same as having mutation....

    This resulted in a regression, 35ms. But I hope to optimize further...

4 - Variables of stack instead of stack of variables

    Every fcall needs cloning, because the new stack frame might change the vars. What if we create all vars in frame 1, and the new stack frames
just store a new value? then when the frame pops we just pop the variables that the frame pushed. Each variable can evolve separately this way.
If a variable is changed more than once in a single frame, we have to be smart and correctly mutate the variable instead of simply pushing.
allocations. 

    This resulted in a 60% improvement in runtime, down to 14ms    

5 - Frame reuse

    When you pop a frame, you discard the memory it allocated during its execution. Why don't reuse these frames and avoi a ton of allocations?

    This resulted in a 109% improvement in runtime, down to 7ms

6 - Stack tuples

    We are still allocating a whole lot because tuples always make a new box. Why not build new tuple variants for primitive types?

     This resulted in a 84% improvement in runtime, down to 3.4ms

7 - Tail recursion

Suppose we have:

    let iter = fn (from, to, call, prev) => {
        if (from < to) {
            let res = call(from);
            iter(from + 1, to, call, res)
        } else {
            prev
        }
    };


    we navigate through the value and check:
    go through the let chain, when there's no Next we check:
         - is it a function? 
           - is it the same function as we? 
             - true
         - is it an if statement?
           - Does one side ends in the function call?
             - true

    If so, we do the following:
    1 - make function return a closure
    let iter = fn (from, to, call, prev) => {
        if (from < to) {
            let res = call(from);
            fn() => { iter(from + 1, to, call, res) }
        } else {
            prev
        }
    };

    The issue is that the compiler builds a bunch of closures that call into each other

    
    iter(1, 10, {}, 0)
        load_var from
        load_var to
        <
        call call, [from]
                ...
        closure 0

    So we only do the trampoline when we want to materialize the value.

    Performance is the same, still needs optimization here...

Off-Enum values

    To reduce the size of copies of values when functions execute, some values were moved to vecs inside the 
    ExecutionEngine struct. 24 bytes down to 12. Down to 2.9ms from 3.4ms.