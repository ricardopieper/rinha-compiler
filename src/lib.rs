#![feature(let_chains)]
use lalrpop_util::lalrpop_mod;

pub mod hir;
pub mod lambda_compiler;
pub mod ast;
pub mod parser;
pub mod typing;


// The lalrpop module, it does generate the parser and lexer
// for the language.
lalrpop_mod! {
    #[allow(warnings)]
    /// The parsing module
    pub rinha
}
