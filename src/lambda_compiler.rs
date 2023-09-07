use std::collections::HashMap;

use crate::{
    ast::{Binary, BinaryOp, Int, Let, Print, Term, Call, Function, Str, If, First, Second, Bool, Tuple},
    parser::Var,
};

/// Enum for runtime values
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Value {
    /// Ints
    Int(i32),
    Bool(bool),
    Str(String),
    Tuple(Box<Value>, Box<Value>),
    FunctionPointer(usize, &'static [&'static str]), //unhinged
    Error(String),
    Unit,
}

impl ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Value::Int(i) => i.to_string(),
            Value::Str(s) => s.to_string(),
            Value::Tuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::FunctionPointer(..) => "function".to_string(),
            Value::Error(e) => format!("error: {}", e),
            Value::Unit => "unit".to_string(),
            Value::Bool(b) => b.to_string(),
        }
    }
}

pub struct StackFrame {
    pub let_bindings: HashMap<&'static str, Value>,
    pub stack: Vec<Value>,
}
impl StackFrame {
    pub fn stack_pop(&mut self) -> Option<Value> {
        self.stack.pop()
    }
    pub fn stack_push(&mut self, value: Value) {
        self.stack.push(value);
    }
}


pub struct ExecutionContext {
    pub call_stack: Vec<StackFrame>,
    pub functions: Vec<Box<dyn Fn(&mut ExecutionContext) -> ()>>
}

impl ExecutionContext {

    pub fn new() -> Self {
        Self {
            call_stack: vec![StackFrame {
                let_bindings: HashMap::new(),
                stack: vec![]
            }],
            functions: vec![]
        }
    }

    pub fn frame(&mut self) -> &mut StackFrame {
        self.call_stack.last_mut().unwrap()
    }
    pub fn pop_frame(&mut self) -> StackFrame {
        self.call_stack.pop().unwrap()
    }
    pub fn push_frame(&mut self, frame: StackFrame) {
        self.call_stack.push(frame);
    }
}

pub struct LambdaCompiler {
    //this is added into during compilation, but should be moved into the execution context later
    pub all_functions: Vec<Box<dyn Fn(&mut ExecutionContext) -> ()>>
}

#[derive(Eq, PartialEq, PartialOrd, Ord)]
pub enum TypeHint {
    Int,
    Str,
    Bool,
    Tuple(Box<TypeHint>, Box<TypeHint>),
    Function,
    Error,
    Unit,
    NoHint,
}

impl LambdaCompiler {
    pub fn new() -> Self {
        Self {
            all_functions: vec![]
        }
    }

    //The idea of the Lambda compiler is to avoid the dispatching overhead that comes with a VM.
    //The idea is to compile the program to a series of functions that can be called directly. So there isn't a big match block on the
    //instructions of the VM. Just function calls.

    //In My github you can find 2 repos: Horse and Donkey. Horse is basically a python clone, and Donkey has evolved into a full blown compiler using LLVM.
    //But there was a time where I was experimenting with a VM for Donkey. I had a VM with some opcodes that manipulated a mmaped memory region. Compared to Horse, it was faster.
    //I then attempted a strategy of removing the bytecode from the program and just compile down to closures. Turns out that was pretty cool, and faster than looping through bytecodes
    //and manipulating a instruction pointer. Well, let's say that one closure counted for a bunch of opcode instructions in the VM. So it was faster.

    //This is also different from a treewalker that navigates through the AST directly. We are applying a transformation on the AST, and running the result of this transformation.

    //The Rinha language is basically a bunch of let bindings to expressions, binding to _ doesn't store anything, it's just made so to run
    //an expression, like a regular statement in a usual language.
    //The last expression can be called without the let binding.
    //Therefore, to compile this program, we have to navigate through the .next field of the let binding until we find the last one.

    //This program has no loops, makes this whole idea easier since we don't need to loop back. Basically, we don't need to keep an instruction pointer,
    //but this idea does work in those cases too.

    //whenever a function is declared, it captures the environment, which consists of the previous let bindings. This is necessary
    //for functions to call other functions.
    //The specs arent clear, but I assume that function parameters will take precedence over let bindings.

    //Let bindings work don't work the same as in Rust, except for _. This language has no shadowing.

    pub fn compile(&mut self, ast: Term) -> Box<dyn Fn(&mut ExecutionContext) -> ()> {
        return match ast {
            Term::Print(Print { value, .. }) => {
                let evaluate_printed_value = self.compile(*value);
                //because the type of the expression is only known at runtime, we have to check it in the print function during runtime :(
                Box::new(move |ec: &mut ExecutionContext| {
                    evaluate_printed_value(ec);
                    let value = ec.frame().stack_pop().unwrap();
                    println!("{}", value.to_string());
                    ec.frame().stack_push(Value::Unit);
                })
            }
            Term::Int(Int { value, .. }) => Box::new(move |ec: &mut ExecutionContext| {
                ec.frame().stack_push(Value::Int(value));
            }),
            Term::Var(Var { text, .. }) => {
                let varname_leaked = text.leak();
                Box::new(move |ec: &mut ExecutionContext| {
                    //@PERF no clone pls
                    let value = ec.frame().let_bindings.get(varname_leaked).cloned();
                    match value {
                        Some(value) => ec.frame().stack_push(value),
                        None => panic!("Variable {varname_leaked} not found"),
                    }
                })
            }
            Term::Let(Let {
                name: Var { text, .. },
                value,
                next,
                ..
            }) => {
                let evaluate_value = self.compile(*value);
                let evaluate_next = self.compile(*next);
                if text == "_" {
                    Box::new(move |ec: &mut ExecutionContext| {
                        evaluate_value(ec);
                        evaluate_next(ec);
                    })
                } else {
                    let varname_leaked: &'static str = text.leak();
                    Box::new(move |ec: &mut ExecutionContext| {
                        evaluate_value(ec);
                        let bound_value = ec.frame().stack_pop().unwrap();
                        ec.frame().let_bindings.insert(varname_leaked, bound_value);
                        evaluate_next(ec);
                    })
                }
            }
            Term::Error(e) => panic!("{}: {}", e.message, e.full_text),
            Term::Binary(Binary { lhs, op, rhs, .. }) => self.compile_binexp(lhs, rhs, op),
            Term::Str(Str{value, ..}) => {
                Box::new(move |ec: &mut ExecutionContext| {
                    ec.frame().stack_push(Value::Str(value.clone()))
                })
            }
            Term::Call(Call{callee, arguments, ..}) => {
                match &*callee {
                    Term::Var(v) => {
                        let called_name: &'static str = v.text.clone().leak();
                        let evaluate_callee = self.compile(*callee);
                        let arguments = arguments.into_iter().map(|arg| self.compile(arg)).collect::<Vec<_>>();
                        Box::new(move |ec: &mut ExecutionContext| {
                            evaluate_callee(ec);
                            let called_function = ec.frame().stack_pop().unwrap();
                    
                            let Value::FunctionPointer(index, params) = called_function else {
                                panic!("Call to non-function value {called_name}")
                            };

                            if params.len() != arguments.len() {
                                panic!("Wrong number of arguments for function {called_name}")
                            }

                            let mut new_frame = StackFrame {
                                let_bindings: HashMap::new(),
                                stack: vec![]
                            };

                            //insert into the new frame the let bindings of the previous frame
                            //@PERF maybe don't clone the values here, find a way to do something like pointers that auto-deref somehow...
                            for (key, value) in ec.frame().let_bindings.iter() {
                               new_frame.let_bindings.insert(key, value.clone());
                            }
                            
                            //in the let bindings, add ourselves so we can do recursion
                            new_frame.let_bindings.insert(called_name, Value::FunctionPointer(index, params));

                            //evaluate the arguments
                            for (argument, param) in arguments.iter().zip(params) {
                                argument(ec);
                                let value = ec.frame().stack_pop().unwrap();
                                new_frame.let_bindings.insert(&param, value);
                            }

                            ec.push_frame(new_frame);

                            //self.all_functions[index](ec);
                            //start running the function in this new frame
                            let function = &ec.functions[index];
                            
                            //HACK: It's 3AM, this borrow is ok... trust me bro...
                            //erase the lifetime of the function pointer
                            let function: &Box<dyn Fn(&mut ExecutionContext)> = unsafe { std::mem::transmute(function) };
                            function(ec);

                            let popped_value = ec.pop_frame().stack_pop().unwrap();
                            //push into the previous frame
                            ec.frame().stack_push(popped_value);
                        })
                    }
                    _ => panic!("Cannot call non-var term")
                }
            }
            Term::Function(Function { parameters, value, .. }) => {
                //Functions are compiled and stored into a big vector of functions, each one has a unique ID.
                let new_function = self.compile(*value);
                let index_of_new_function = self.all_functions.len();
                self.all_functions.push(new_function);
                let parameters = parameters.into_iter().map(|param| param.text).collect::<Vec<_>>();
                let all_params_leaked = parameters.into_iter().map(|param| -> &'static str { param.leak()}).collect::<Vec<_>>();
                let all_params_leaked: &'static [&'static str] = all_params_leaked.leak(); //absolutely unhinged
                Box::new(move |ec: &mut ExecutionContext| {
                    ec.frame().stack_push(Value::FunctionPointer(index_of_new_function, all_params_leaked));
                })
            }
            Term::If(If {condition, then, otherwise, ..}) => {
                let evaluate_condition = self.compile(*condition);
                let evaluate_then = self.compile(*then);
                let evaluate_otherwise = self.compile(*otherwise);
                Box::new(move |ec: &mut ExecutionContext| {
                    evaluate_condition(ec);
                    let condition = ec.frame().stack_pop().unwrap();
                    match condition {
                        Value::Bool(true) => evaluate_then(ec),
                        Value::Bool(false) => evaluate_otherwise(ec),
                        _ => panic!("Type error: Cannot evaluate condition on this value: {condition:?}"),
                    }
                })
            } 
            //only works on tuples
            Term::First(First{value, ..}) => {
                let evaluate_value = self.compile(*value);
                Box::new(move |ec: &mut ExecutionContext| {
                    evaluate_value(ec);
                    let value = ec.frame().stack_pop().unwrap();
                    match value {
                        Value::Tuple(a, _) => ec.frame().stack_push(*a),
                        _ => panic!("Type error: Cannot evaluate first on this value: {value:?}"),
                    }
                })
            }
            Term::Second(Second{value, ..}) => {
                let evaluate_value = self.compile(*value);
                Box::new(move |ec: &mut ExecutionContext| {
                    evaluate_value(ec);
                    let value = ec.frame().stack_pop().unwrap();
                    match value {
                        Value::Tuple(_, a) => ec.frame().stack_push(*a),
                        _ => panic!("Type error: Cannot evaluate first on this value: {value:?}"),
                    }
                })
            }
            Term::Bool(Bool {value, ..}) => {
                Box::new(move |ec: &mut ExecutionContext| {
                    ec.frame().stack_push(Value::Bool(value));
                })
            },
            Term::Tuple(Tuple{first, second, ..}) => {
                let evaluate_first = self.compile(*first);
                let evaluate_second = self.compile(*second);
                Box::new(move |ec: &mut ExecutionContext| {
                    evaluate_first(ec);
                    evaluate_second(ec);
                    let second = ec.frame().stack_pop().unwrap();
                    let first = ec.frame().stack_pop().unwrap();
                    ec.frame().stack_push(Value::Tuple(Box::new(first), Box::new(second)));
                })
            }
        };
    }

    fn compile_binexp(
        &mut self, 
        lhs: Box<Term>,
        rhs: Box<Term>,
        op: BinaryOp
    ) -> Box<dyn Fn(&mut ExecutionContext)> {
        let evaluate_lhs = self.compile(*lhs);
        let evaluate_rhs = self.compile(*rhs);

        macro_rules! int_binary_numeric_op {
            ($op:tt) => {
                Box::new(move |ec: &mut ExecutionContext| {
                    evaluate_lhs(ec);
                    evaluate_rhs(ec);
                    let rhs = ec.frame().stack_pop().unwrap();
                    let lhs = ec.frame().stack_pop().unwrap();
                    match (&lhs, &rhs) {
                        (Value::Int(lhs), Value::Int(rhs)) => {
                            ec.frame().stack_push(Value::Int(lhs $op rhs));
                        }
                        _ => panic!("Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}", op = stringify!($op)),
                    }
                })
            };
        }

        macro_rules! int_binary_logical_op {
            ($op:tt) => {
                Box::new(move |ec: &mut ExecutionContext| {
                    evaluate_lhs(ec);
                    evaluate_rhs(ec);
                    let rhs = ec.frame().stack_pop().unwrap();
                    let lhs = ec.frame().stack_pop().unwrap();
                    match (&lhs, &rhs) {
                        (Value::Int(lhs), Value::Int(rhs)) => {
                            ec.frame().stack_push(Value::Bool(lhs $op rhs));
                        }
                        _ => panic!("Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}", op = stringify!($op)),
                    }
                })
            };
        }

        match op {
            BinaryOp::Add => Box::new(move |ec: &mut ExecutionContext| {
                evaluate_lhs(ec);
                evaluate_rhs(ec);
                let rhs = ec.frame().stack_pop().unwrap();
                let lhs = ec.frame().stack_pop().unwrap();
                let value = match (&lhs, &rhs) {
                    (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs + rhs),
                    (Value::Int(lhs), Value::Bool(rhs)) => Value::Int(lhs + *rhs as i32),
                    (_, Value::Str(_)) | (Value::Str(_), _) => Value::Str(format!("{}{}", lhs.to_string(), rhs.to_string())),                           
                    (Value::Bool(lhs), Value::Int(rhs)) => Value::Int(*lhs as i32 + *rhs),
                    _ => panic!("Type error: Cannot apply binary operator + on these values: {lhs:?} and {rhs:?}"),
                };
                ec.frame().stack_push(value);
            }),
            BinaryOp::Sub => int_binary_numeric_op!(-),
            BinaryOp::Mul => int_binary_numeric_op!(*),
            BinaryOp::Div => int_binary_numeric_op!(/),
            BinaryOp::Rem => int_binary_numeric_op!(%),
            BinaryOp::Eq => int_binary_logical_op!(==),
            BinaryOp::Neq => int_binary_logical_op!(!=),
            BinaryOp::Lt => int_binary_logical_op!(<),
            BinaryOp::Gt => int_binary_logical_op!(>),
            BinaryOp::Lte => int_binary_logical_op!(<=),
            BinaryOp::Gte => int_binary_logical_op!(>=),
            BinaryOp::And => Box::new(move |ec: &mut ExecutionContext| {
                evaluate_lhs(ec);
                evaluate_rhs(ec);
                let rhs = ec.frame().stack_pop().unwrap();
                let lhs = ec.frame().stack_pop().unwrap();
                let value = match (&lhs, &rhs) {
                (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(*lhs && *rhs),
                _ => panic!("Type error: Cannot apply binary operator && (and) on these values: {lhs:?} and {rhs:?}"),
            };
                ec.frame().stack_push(value);
            }),
            BinaryOp::Or => Box::new(move |ec: &mut ExecutionContext| {
                evaluate_lhs(ec);
                evaluate_rhs(ec);
                let rhs = ec.frame().stack_pop().unwrap();
                let lhs = ec.frame().stack_pop().unwrap();
                let value = match (&lhs, &rhs) {
                (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(*lhs && *rhs),
                _ => panic!("Type error: Cannot apply binary operator || (or) on these values: {lhs:?} and {rhs:?}"),
            };
                ec.frame().stack_push(value);
            }),
        }
    }
}
