1 - Use BTreeMap: 43% improvement (28ms)
2 - Use Vec:

    Instead of btreemap, we have a Vec<Value>, and compile Var to an index. The program executor has to allocate vecs big enough to store at least the amount of data.
    It's a table of names, since shadowing is allowed it's kinda the same as having mutation....

    This resulted in a regression, 35ms. But I hope to optimize further...

4 - Variables of stack instead of stack of variables

    - Every fcall needs cloning, because the new stack frame might change the vars. What if we create all vars in frame 1, and the new stack frames
just store a new value? then when the frame pops we just pop the variables that the frame pushed. Each variable can evolve separately this way.
If a variable is changed more than once in a single frame, we have to be smart and correctly mutate the variable instead of simply pushing.
allocations. 