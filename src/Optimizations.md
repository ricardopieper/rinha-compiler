1 - Use BTreeMap: 43% improvement
2 - Use Vec:

    Instead of btreemap, we have a Vec<Value>, and compile Var to an index. The program executor has to allocate vecs big enough to store at least the amount of data.
    It's a table of names, since shadowing is allowed it's kinda the same as having mutation....

3 - No clone (Cow<Value>)