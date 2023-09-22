use crate::{ast::{Binary, Bool, Call, First, Function, If, Let, Print, Second, Tuple, BinaryOp, Term}, parser::Var};

#[derive(Debug, Clone)]
pub struct FuncDecl{
    pub params: Vec<String>,
    pub body: Box<Expr>, //the return type is the type of this body,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Let {
        name: String,
        value: Box<Expr>,
        next: Box<Expr>,
    },
    Var {
        name: String,
    },
    Int {
        value: i32,
    },
    Bool {
        value: bool,
    },
    String {
        value: String,
    },
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>
    },
    Tuple {
        first: Box<Expr>,
        second: Box<Expr>,
        //the type of tuple is the type Tuple with the types inferred from the first and second
    },
    First {
        value: Box<Expr>
    },
    Second {
        value: Box<Expr>,
    },
    Print {
        value: Box<Expr>,
    },
    FuncDecl(FuncDecl),
    FuncCall {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    BinOp {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
}

pub fn ast_to_hir(ast: Term) -> Expr {
    match ast {
        Term::Error(_) => todo!("Error exprs not supported"),
        Term::Int(i) => Expr::Int { value: i.value },
        Term::Str(s) => Expr::String { value: s.value },
        Term::Call(Call {
            arguments, callee, ..
        }) => {
            let args = arguments
                .into_iter()
                .map(|arg| ast_to_hir(arg))
                .collect::<Vec<_>>();

            let func = ast_to_hir(*callee);

            Expr::FuncCall {
                func: func.into(),
                args,
            }
        }
        Term::Binary(Binary { lhs, rhs, op, .. }) => binop_hir(lhs, rhs, op),
        Term::Function(Function {
            parameters, value, ..
        }) => {
            let args = parameters
                .into_iter()
                .map(|arg| arg.text)
                .collect::<Vec<_>>();

           
            let ast = ast_to_hir(*value);
            Expr::FuncDecl(FuncDecl {
                params: args,
                body: Box::new(ast),
            })
        }
        Term::Let(Let {
            name, value, next, ..
        }) => {
            let value = ast_to_hir(*value);
            Expr::Let {
                name: name.text,
                value: value.into(),
                next: Box::new(ast_to_hir(*next)),
            }
        }
        Term::If(If {
            condition,
            then,
            otherwise,
            ..
        }) =>{
            let condition = ast_to_hir(*condition);
            let then = ast_to_hir(*then);
            let otherwise = ast_to_hir(*otherwise);
         
            Expr::If {
                cond: Box::new(condition),
                then: Box::new(then),
                otherwise: Box::new(otherwise),
            }
        }
        Term::Print(Print { value, .. }) => {
            let value = ast_to_hir(*value);
            Expr::Print {
                value: Box::new(value),
            }
        }
        Term::First(First { value, .. }) => {
            let maybe_tuple = ast_to_hir(*value);
            Expr::First {
                value: maybe_tuple.into() 
            }
        }
        Term::Second(Second { value, .. }) => {
            let maybe_tuple = ast_to_hir(*value);
            //if we already know it's a tuple, we just return the first
            Expr::Second {
                value: maybe_tuple.into() 
            }
        }
        Term::Bool(Bool { value, .. }) => Expr::Bool { value },
        Term::Tuple(Tuple { first, second, .. }) => {

            let first = ast_to_hir(*first);
            let second = ast_to_hir(*second);
          
            Expr::Tuple {
                first: Box::new(first),
                second: Box::new(second),
            }
        }
        Term::Var(Var { text, .. }) => Expr::Var {
            name: text,
        },
    }
}

fn binop_hir(
    lhs: Box<Term>,
    rhs: Box<Term>,
    op: BinaryOp,
) -> Expr {
    let lhs = ast_to_hir(*lhs);
    let rhs = ast_to_hir(*rhs);
    Expr::BinOp {
        op,
        left: lhs.into(),
        right: rhs.into(),
        
    }
}
