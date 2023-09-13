A idéia desse interpretador é não usar VM nem tree-walker, e sim encadear funções pra cada "instrução" da linguagem. Talvez usar unsafe em alguns lugares.


Otimizacoes:
 - Memoization
 - JIT (em thread separada com OSR, tem como?)
 - Constant folding