use crate::{ast::{Binary, Bool, Call, First, Function, If, Let, Print, Second, Tuple, BinaryOp, Term}, parser::Var};

#[derive(Debug, Clone)]
pub struct FuncDecl{
    pub params: Vec<String>,
    pub body: Vec<Expr>,
    pub is_tco_trampoline: bool
}

#[derive(Debug, Clone)]
pub enum Expr {
    Let {
        name: String,
        value: Box<Expr>,
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
        then: Vec<Expr>,
        otherwise: Vec<Expr>
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

pub fn ast_to_hir(ast: Term) -> Vec<Expr> {
    let mut result: Vec<Expr> = vec![];
    let mut ast = ast;
    loop {
        let (expr, next) = match ast {
            Term::Error(_) => todo!("Error exprs not supported"),
            Term::Int(i) => (Expr::Int { value: i.value }, None),
            Term::Str(s) => (Expr::String { value: s.value }, None),
            Term::Call(Call {
                arguments, callee, ..
            }) => {
                //args can only be a single expr
                let args = arguments
                    .into_iter()
                    .map(|arg| ast_to_hir(arg).pop().unwrap())
                    .collect::<Vec<_>>();
    
                let func = ast_to_hir(*callee).pop().unwrap().into();
    
                (Expr::FuncCall {
                    func,
                    args,
                }, None)
            }
            Term::Binary(Binary { lhs, rhs, op, .. }) => (binop_hir(lhs, rhs, op), None),
            Term::Function(Function {
                parameters, value, ..
            }) => {
                let args = parameters
                    .into_iter()
                    .map(|arg| arg.text)
                    .collect::<Vec<_>>();
    
               
                let ast = ast_to_hir(*value);
                (Expr::FuncDecl(FuncDecl {
                    params: args,
                    body: ast,
                    is_tco_trampoline: false
                }), None)
            }
            Term::Let(Let {
                name, value, next, ..
            }) => {
                let value = ast_to_hir(*value).pop().unwrap();
                (Expr::Let {
                    name: name.text,
                    value: value.into(),
                }, Some(next))
                
            }
            Term::If(If {
                condition,
                then,
                otherwise,
                ..
            }) =>{
                let condition = ast_to_hir(*condition).pop().unwrap();
                let then = ast_to_hir(*then);
                let otherwise = ast_to_hir(*otherwise);
             
                (Expr::If {
                    cond: Box::new(condition),
                    then,
                    otherwise,
                }, None)
            }
            Term::Print(Print { value, .. }) => {
                let value = ast_to_hir(*value).pop().unwrap();
                (Expr::Print {
                    value: Box::new(value),
                }, None)
            }
            Term::First(First { value, .. }) => {
                let maybe_tuple = ast_to_hir(*value).pop().unwrap();
                (Expr::First {
                    value: maybe_tuple.into() 
                }, None)
            }
            Term::Second(Second { value, .. }) => {
                let maybe_tuple = ast_to_hir(*value).pop().unwrap();
                //if we already know it's a tuple, we just return the first
                (Expr::Second {
                    value: maybe_tuple.into() 
                }, None)
            }
            Term::Bool(Bool { value, .. }) => (Expr::Bool { value }, None),
            Term::Tuple(Tuple { first, second, .. }) => {
    
                let first = ast_to_hir(*first).pop().unwrap();
                let second = ast_to_hir(*second).pop().unwrap();
              
                (Expr::Tuple {
                    first: Box::new(first),
                    second: Box::new(second),
                }, None)
            }
            Term::Var(Var { text, .. }) => (Expr::Var {
                name: text,
            }, None)
        };
        result.push(expr);
        match next {
            Some(next) => {
                ast = *next;
            }
            None => {
                break;
            }
        }
    }

    result
   
}

fn binop_hir(
    lhs: Box<Term>,
    rhs: Box<Term>,
    op: BinaryOp,
) -> Expr {
    let lhs = ast_to_hir(*lhs).pop().unwrap();
    let rhs = ast_to_hir(*rhs).pop().unwrap();
    Expr::BinOp {
        op,
        left: lhs.into(),
        right: rhs.into(),
        
    }
}
