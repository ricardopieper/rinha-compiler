A idéia desse interpretador é não usar VM nem tree-walker, e sim encadear funções pra cada "instrução" da linguagem. Talvez usar unsafe em alguns lugares.

Eu copiei e colei o codigo do repo original, visto que vou usar Rust e esse repo já tem o parser pronto :)


Eu estou pensando em traduzir a AST desse compilador para o formato HIR (ou MIR, não sei ainda) MIR do meu compilador Donkey (https://github.com/ricardopieper/donkey-lang), mas estou trabalhando nele há algum tempo (i.e. anos) então acho que nao é legal pro espirito da competição (se for performance) em usar código feito antes.
