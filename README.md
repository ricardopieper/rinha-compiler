Lambda Rinha Compiler
=====================

A idéia desse interpretador é não usar VM nem tree-walker, e sim encadear funções pra cada "instrução" da linguagem. Talvez usar unsafe em alguns lugares.
Outros participantes da rinha chamaram isso de "bake" ou "HOAS" (High-Order Abstract Syntax), mas são basicamente um monte de funções anônimas que chamam umas as outras.

Para rodar:
-----------

Rodar um arquivo .rinha direto:
```
cargo run -- --mode=interpreter ./files/fib.rinha
```

Rodar um arquivo JSON já parseado (`--mode=rinha`):
```
cargo run ./files/fib.rinha.json
```

Otimizações implementadas:
--------------------------

Verifique o arquivo Optimizations.md dentro de `/src`.