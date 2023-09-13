use std::collections::BTreeMap;

use crate::{
    ast::{Binary, BinaryOp, Int, Let, Print, Term, Call, Function, Str, If, First, Second, Bool, Tuple},
    parser::Var,
};


#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Closure {
    pub callable_index: usize,
    pub environment: Vec<Value>
}

/// Enum for runtime values
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum Value {
    /// Ints
    Int(i32),
    Bool(bool),
    Str(String),
    Tuple(Box<Value>, Box<Value>),
    Closure(Closure), //unhinged
    Error(String),
    Unit,
}

impl Value {
    fn get_type(&self) -> Type {
        match self {
            Value::Int(_) => Type::Int,
            Value::Str(_) => Type::Str,
            Value::Tuple(..) => Type::Tuple,
            Value::Closure(..) => Type::Function,
            Value::Error(_) => Type::Error,
            Value::Unit => Type::Unit,
            Value::Bool(_) => Type::Bool,
        }
    }
}

impl ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Value::Int(i) => i.to_string(),
            Value::Str(s) => s.to_string(),
            Value::Tuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::Closure(..) => "function".to_string(),
            Value::Error(e) => format!("error: {}", e),
            Value::Unit => "unit".to_string(),
            Value::Bool(b) => b.to_string(),
        }
    }
}

pub struct StackFrame {
    pub let_bindings: Vec<Value>,
}

pub struct Callable {
    pub parameters: &'static [usize],
    pub body: Box<dyn Fn(&mut ExecutionContext) -> Value>
}

pub struct ExecutionContext<'a> {
    pub call_stack: Vec<StackFrame>,
    pub functions: &'a [Callable],
    pub enable_memoization: bool,
    pub num_of_vars: usize
}

pub struct CompilationResult {
    pub main: Box<dyn Fn(&mut ExecutionContext) -> Value>,
    pub strings: Vec<&'static str>,
    pub functions: Vec<Callable>,
}

impl<'a> ExecutionContext<'a> {

    pub fn new(program: &'a CompilationResult) -> Self {
        Self {
            call_stack: vec![StackFrame {
                let_bindings: vec![Value::Unit; program.strings.len()],
            }],
            functions: &program.functions,
            enable_memoization: true,
            num_of_vars: program.strings.len()
        }
    }

    pub fn disable_memoization(&mut self) {
        self.enable_memoization = false;
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
    pub all_functions: Vec<Callable>,
    pub var_names: Vec<&'static str>,
    pub strict_equals: bool,

}

#[derive(Eq, PartialEq, PartialOrd, Ord, Debug)]
pub enum Type {
    Int,
    Str,
    Bool,
    Tuple,
    Function,
    Error,
    Unit,
    NoHint,
}

impl LambdaCompiler {
    pub fn new() -> Self {
        Self {
            all_functions: vec![],
            strict_equals: true,
            var_names: vec![]
        }
    }

    pub fn relax_equals(&mut self) {
        self.strict_equals = false;
    }

    pub fn intern_string(&mut self, s: &'static str) -> usize {
        //check if it already exists
        if let Some(index) = self.var_names.iter().position(|x| x == &s) {
            return index;
        }

        let index = self.var_names.len();
        self.var_names.push(s);
        index
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

    //whenever a function is constructed, it captures the environment. This is a closure.

    fn compile_internal(&mut self, ast: Term) -> Box<dyn Fn(&mut ExecutionContext) -> Value> {
        return match ast {
            Term::Print(Print { value, .. }) => {
                let evaluate_printed_value = self.compile_internal(*value);
                //because the type of the expression is only known at runtime, we have to check it in the print function during runtime :(
                Box::new(move |ec: &mut ExecutionContext| {
                    //println!("Stack frame on print: {:?}", ec.frame().let_bindings);
                    let value_to_print = evaluate_printed_value(ec);
                    println!("{}", value_to_print.to_string());
                    return Value::Unit;
                })
            }
            Term::Int(Int { value, .. }) => Box::new(move |_| {
                return Value::Int(value);
            }),
            Term::Var(Var { text, .. }) => {
                let varname_leaked: &'static str = text.leak();
                let index = self.intern_string(varname_leaked);

                Box::new(move |ec: &mut ExecutionContext| {
                    //@PERF no clone pls
                   // println!("Stack frame on var: {:?} trying to get var {varname_leaked}", ec.frame().let_bindings);
                    let value = ec.frame().let_bindings.get(index);
                    match value {
                        Some(value) => value.clone(),
                        None => panic!("Variable {varname_leaked} not found"),
                    }
                })
            }
            Term::Let(Let {
                name: Var { text: var_name, .. },
                value,
                next,
                ..
            }) => {
                let evaluate_value = self.compile_internal(*value);
                let evaluate_next = self.compile_internal(*next);
                if var_name == "_" {
                    Box::new(move |ec: &mut ExecutionContext| {
                        evaluate_value(ec);
                        let next = evaluate_next(ec);
                        return next;
                    })
                } else {
                    let var_name_leaked: &'static str = var_name.leak();
                    let next_index = self.intern_string(var_name_leaked);
                    Box::new(move |ec: &mut ExecutionContext| {
                        let bound_value = evaluate_value(ec);
                        ec.frame().let_bindings[next_index] = bound_value;
                        //print stack frame
                        //println!("Stack frame: {:?}", ec.frame().let_bindings);
                        let next = evaluate_next(ec);
                        return next;
                    })
                }
            }
            Term::Error(e) => panic!("{}: {}", e.message, e.full_text),
            Term::Binary(Binary { lhs, op, rhs, .. }) => self.compile_binexp(lhs, rhs, op),
            Term::Str(Str{value, ..}) => {
                Box::new(move |_| Value::Str(value.clone()))
            }
            Term::Call(Call{callee, arguments, ..}) => {
                match &*callee {
                    Term::Var(v) => {
                        let called_name: &'static str = v.text.clone().leak();
                        let function_name_index = self.intern_string(called_name); 
                        let evaluate_callee = self.compile_internal(*callee);
                        let arguments = arguments.into_iter().map(|arg| self.compile_internal(arg)).collect::<Vec<_>>();
                        Box::new(move |ec: &mut ExecutionContext| {
                                                        
                            let callee_function = evaluate_callee(ec);

                            //println!("called {callee_function:?}");

                            let Value::Closure(Closure{callable_index, environment}) = callee_function else {
                                panic!("Call to non-function value {called_name}")
                            };
                            {
                                let callable = &ec.functions[callable_index];

                                if callable.parameters.len() != arguments.len() {
                                    panic!("Wrong number of arguments for function {called_name}")
                                }
                            }

                             //copy the environment to the let bindings of the new frame, they might be overriden by parameters
                           
                            let mut new_frame = StackFrame {
                                let_bindings: environment.clone(),
                            };

                            
                            //in the let bindings, add ourselves so we can do recursion
                            new_frame.let_bindings[function_name_index] = Value::Closure(Closure { callable_index, environment });

                            {
                                let params = ec.functions[callable_index].parameters;
                                //evaluate the arguments
                                for (argument, param) in arguments.iter().zip(params) {
                                    let value = argument(ec);
                                    new_frame.let_bindings[*param] = value;
                                }
                            }
                            

                            ec.push_frame(new_frame);

                            //HACK: It's 3AM, this borrow is ok... trust me bro...
                            //erase the lifetime of the function pointer
                            let function: &Box<dyn Fn(&mut ExecutionContext) -> Value> = unsafe { std::mem::transmute(&ec.functions[callable_index].body) };
                            //let function = &ec.functions[callable_index].body;
                            let function_result = function(ec);

                            ec.pop_frame();
                            return function_result;
                        })
                    }
                    _ => panic!("Cannot call non-var term")
                }
            }
            Term::Function(Function { parameters, value, .. }) => {
                //Functions are compiled and stored into a big vector of functions, each one has a unique ID.
                let new_function = self.compile_internal(*value);

               
                let parameters = parameters.into_iter().map(|param| param.text).collect::<Vec<_>>();
                let mut param_indices = vec![];
                for param in parameters {
                    let interned = self.intern_string(param.leak());
                    param_indices.push(interned);
                }
                let callable = Callable {
                    body: new_function,
                    parameters: param_indices.leak()
                };
                let index_of_new_function = self.all_functions.len();

                self.all_functions.push(callable);
                Box::new(move |ec| {
                    let environment = ec.frame().let_bindings.clone();
                    Value::Closure(Closure { callable_index: index_of_new_function, environment })
                })
            }
            Term::If(If {condition, then, otherwise, ..}) => {
                let evaluate_condition = self.compile_internal(*condition);
                let evaluate_then = self.compile_internal(*then);
                let evaluate_otherwise = self.compile_internal(*otherwise);
                Box::new(move |ec: &mut ExecutionContext| {
                    let condition_result = evaluate_condition(ec);
                    match condition_result {
                        Value::Bool(true) => evaluate_then(ec),
                        Value::Bool(false) => evaluate_otherwise(ec),
                        _ => panic!("Type error: Cannot evaluate condition on this value: {condition_result:?}"),
                    }
                })
            } 
            //only works on tuples
            Term::First(First{value, ..}) => {
                let evaluate_value = self.compile_internal(*value);
                Box::new(move |ec: &mut ExecutionContext| {
                    let value = evaluate_value(ec);
                    match value {
                        Value::Tuple(a, _) => *a,
                        _ => panic!("Type error: Cannot evaluate first on this value: {value:?}"),
                    }
                })
            }
            Term::Second(Second{value, ..}) => {
                let evaluate_value = self.compile_internal(*value);
                Box::new(move |ec: &mut ExecutionContext| {
                    let value = evaluate_value(ec);
                    match value {
                        Value::Tuple(_, a) => *a,
                        _ => panic!("Type error: Cannot evaluate first on this value: {value:?}"),
                    }
                })
            }
            Term::Bool(Bool {value, ..}) => {
                Box::new(move |_| Value::Bool(value))
            },
            Term::Tuple(Tuple{first, second, ..}) => {
                let evaluate_first = self.compile_internal(*first);
                let evaluate_second = self.compile_internal(*second);
                Box::new(move |ec: &mut ExecutionContext| {
                    Value::Tuple(Box::new(evaluate_first(ec)), Box::new(evaluate_second(ec)))
                })
            }
        };
    }

    pub fn compile(mut self, ast: Term) -> CompilationResult {
        let main = self.compile_internal(ast);
       
        CompilationResult {
            main,
            strings: self.var_names,
            functions: self.all_functions
        }
    }

    fn compile_binexp(
        &mut self, 
        lhs: Box<Term>,
        rhs: Box<Term>,
        op: BinaryOp
    ) -> Box<dyn Fn(&mut ExecutionContext) -> Value> {
        let evaluate_lhs = self.compile_internal(*lhs);
        let evaluate_rhs = self.compile_internal(*rhs);

        macro_rules! int_binary_numeric_op {
            ($op:tt) => {
                Box::new(move |ec: &mut ExecutionContext| {
                    let lhs = evaluate_lhs(ec);
                    let rhs = evaluate_rhs(ec);
                    match (&lhs, &rhs) {
                        (Value::Int(lhs), Value::Int(rhs)) => {
                            Value::Int(lhs $op rhs)
                        }
                        _ => panic!("Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}", op = stringify!($op)),
                    }
                })
            };
        }

        macro_rules! binary_comparison_op {
            ($op:tt) => {

                if (self.strict_equals) {
                    Box::new(move |ec: &mut ExecutionContext| {
                        let lhs = evaluate_lhs(ec);
                        let rhs = evaluate_rhs(ec);
                        
                        match (&lhs, &rhs) {
                            (Value::Int(lhs), Value::Int(rhs)) => {
                                Value::Bool(lhs $op rhs)
                            }
                            (Value::Str(lhs), Value::Str(rhs)) => {
                                Value::Bool(lhs $op rhs)
                            }
                            (Value::Bool(lhs), Value::Bool(rhs)) => {
                                Value::Bool(lhs $op rhs)
                            }
                            _ => panic!("Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}", op = stringify!($op)),
                        }
                    })
                } else {
                    Box::new(move |ec: &mut ExecutionContext| {
                        let lhs = evaluate_lhs(ec);
                        let rhs = evaluate_rhs(ec);
                        
                        return Value::Bool(lhs $op rhs);
                    })
                }
            };
        }

        match op {
            BinaryOp::Add => Box::new(move |ec: &mut ExecutionContext| {
                let lhs = evaluate_lhs(ec);
                let rhs = evaluate_rhs(ec);
                match (&lhs, &rhs) {
                    (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs + rhs),
                    (Value::Int(lhs), Value::Bool(rhs)) => Value::Int(lhs + *rhs as i32),
                    (_, Value::Str(_)) | (Value::Str(_), _) => Value::Str(format!("{}{}", lhs.to_string(), rhs.to_string())),
                    (Value::Bool(lhs), Value::Int(rhs)) => Value::Int(*lhs as i32 + *rhs),
                    _ => panic!("Type error: Cannot apply binary operator + on these values: {lhs:?} and {rhs:?}"),
                }
            }),
            BinaryOp::Sub => int_binary_numeric_op!(-),
            BinaryOp::Mul => int_binary_numeric_op!(*),
            BinaryOp::Div => int_binary_numeric_op!(/),
            BinaryOp::Rem => int_binary_numeric_op!(%),
            BinaryOp::Eq => binary_comparison_op!(==),
            BinaryOp::Neq => binary_comparison_op!(!=),
            BinaryOp::Lt => binary_comparison_op!(<),
            BinaryOp::Gt => binary_comparison_op!(>),
            BinaryOp::Lte => binary_comparison_op!(<=),
            BinaryOp::Gte => binary_comparison_op!(>=),
            BinaryOp::And => Box::new(move |ec: &mut ExecutionContext| {
                let lhs = evaluate_lhs(ec);
                let rhs = evaluate_rhs(ec);
               
                match (&lhs, &rhs) {
                    (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(*lhs && *rhs),
                    _ => panic!("Type error: Cannot apply binary operator && (and) on these values: {lhs:?} and {rhs:?}"),
                }
            }),
            BinaryOp::Or => Box::new(move |ec: &mut ExecutionContext| {
                let lhs = evaluate_lhs(ec);
                let rhs = evaluate_rhs(ec);
                match (&lhs, &rhs) {
                    (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(*lhs || *rhs),
                    _ => panic!("Type error: Cannot apply binary operator || (or) on these values: {lhs:?} and {rhs:?}"),
                }
            }),
        }
    }
}
