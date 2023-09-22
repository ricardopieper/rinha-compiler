

use crate::{
    ast::BinaryOp
};

use crate::hir::{Expr, FuncDecl};

pub type LambdaFunction = Box<dyn Fn(&mut ExecutionContext) -> Value>;

#[derive(Debug)]
pub struct Stats {
    pub new_frames: usize,
    pub reused_frames: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Closure {
    pub callable_index: usize,
    //  pub environment: Vec<Value>
}

/// Enum for runtime values
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum Value {
    /// Ints
    Int(i32),
    Bool(bool),
    Str(&'static str),
    IntTuple(i32, i32),
    BoolIntTuple(bool, i32),
    IntBoolTuple(i32, bool),
    BoolTuple(bool, bool),
    DynamicTuple(Box<Value>, Box<Value>),
    Closure(Closure),
    Trampoline(Closure),
    Error(&'static str),
    //Unit,
    //None,
}


impl ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Value::Trampoline(_) => "trampoline".to_string(),
            Value::Int(i) => i.to_string(),
            Value::Str(s) => s.to_string(),
            Value::DynamicTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::IntTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::BoolTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::IntBoolTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::BoolIntTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::Closure(..) => "function".to_string(),
            Value::Error(e) => format!("error: {}", e),
           // Value::Unit => "unit".to_string(),
            Value::Bool(b) => b.to_string(),
        }
    }
}

pub struct StackFrame {
    function: usize,
    pub let_bindings_pushed: Vec<usize>,
}

pub struct Callable {
    pub parameters: &'static [usize],
    pub body: LambdaFunction,
    //pub location: Location,
    pub tco_optimized: bool,
    pub trampoline_of: Option<usize> //the callable index this callable belongs to
}

pub struct ExecutionContext<'a> {
    pub call_stack: Vec<StackFrame>,
    pub functions: &'a [Callable],
    pub enable_memoization: bool,
    pub let_bindings: Vec<Vec<Value>>,
    pub reusable_frames: Vec<StackFrame>,
    pub stats: Stats,
}

pub struct CompilationResult {
    pub main: LambdaFunction,
    pub strings: Vec<&'static str>,
    pub functions: Vec<Callable>,
}

impl<'a> ExecutionContext<'a> {
    pub fn new(program: &'a CompilationResult) -> Self {
        Self {
            call_stack: vec![StackFrame {
                function: usize::MAX,
                let_bindings_pushed: vec![],
            }],
            functions: &program.functions,
            enable_memoization: true,
            let_bindings: vec![vec![]; program.strings.len()],
            reusable_frames: vec![],
            stats: Stats {
                new_frames: 1,
                reused_frames: 0,
            },
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


    #[inline(always)]
    fn eval_closure(&self, index_of_new_function: usize) -> Value {
        Value::Closure(Closure {
            callable_index: index_of_new_function,
        })
    }

    #[inline(always)]
    fn eval_int(&self, value: i32) -> Value {
        return Value::Int(value);
    }

    #[inline(always)]
    fn eval_let(
        &mut self,
        evaluate_value: &LambdaFunction,
        evaluate_next: &LambdaFunction,
        var_index: usize,
    ) -> Value {
        let bound_value = self.run_lambda_trampoline(evaluate_value);
        let call_stack_index = self.call_stack.len() - 1;
        let frame_bindings = &mut self.let_bindings[var_index];
        let var_in_stack = frame_bindings.get_mut(call_stack_index);
        if let Some(var_in_stack) = var_in_stack {
            *var_in_stack = bound_value;
        } else {
            frame_bindings.push(bound_value);
            self.frame().let_bindings_pushed.push(var_index);
        }
        //Does not run trampoline because it could be the last statement in the function, which would cause recursion
        let next = evaluate_next(self);
        return next;
    }

    #[inline(always)]
    fn eval_var(&self, index: usize, varname: &'static str) -> Value {
        //@PERF no clone pls
        //println!("Trying to get var {index} {varname}");
        let var_stack = self.let_bindings.get(index);
        match var_stack {
            Some(value) => value.last().unwrap().clone(), //will only be cloned and heap allocated on complex structures
            None => panic!("Variable {varname} not found"),
        }
    }

    #[inline(always)]
    fn eval_if(
        &mut self,
        evaluate_condition: &LambdaFunction,
        evaluate_then: &LambdaFunction,
        evaluate_otherwise: &LambdaFunction,
    ) -> Value {
        let condition_result = self.run_lambda_trampoline(evaluate_condition);
        match condition_result {
            Value::Bool(true) => evaluate_then(self),
            Value::Bool(false) => evaluate_otherwise(self),
            _ => {
                panic!("Type error: Cannot evaluate condition on this value: {condition_result:?}")
            }
        }
    }

    fn binop_mul(&mut self, evaluate_lhs: &LambdaFunction, evaluate_rhs: &LambdaFunction) -> Value {
        let lhs = evaluate_lhs(self);
        let rhs = evaluate_rhs(self);
        match (&lhs, &rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs * rhs),
            _ => panic!(
                "Type error: Cannot apply binary operator Mul on these values: {lhs:?} and {rhs:?}"
            ),
        }
    }

    fn binop_add(&mut self, evaluate_lhs: &LambdaFunction, evaluate_rhs: &LambdaFunction) -> Value {
        let lhs = self.run_lambda_trampoline(&evaluate_lhs);
        let rhs = self.run_lambda_trampoline(&evaluate_rhs);
        match (&lhs, &rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs + rhs),
            (Value::Int(lhs), Value::Bool(rhs)) => Value::Int(lhs + *rhs as i32),
            (_, Value::Str(_)) | (Value::Str(_), _) => {
                Value::Str(format!("{}{}", lhs.to_string(), rhs.to_string()).leak())
            }
            (Value::Bool(lhs), Value::Int(rhs)) => Value::Int(*lhs as i32 + *rhs),
            _ => panic!(
                "Type error: Cannot apply binary operator + on these values: {lhs:?} and {rhs:?}"
            ),
        }
    }

    fn eval_call(
        &mut self,
        evaluate_callee: &LambdaFunction,
        arguments: &Vec<LambdaFunction>,
        function_name_index: usize,
        called_name: &str,
    ) -> Value {
        let callee_function = evaluate_callee(self);
        let Value::Closure(Closure { callable_index }) = callee_function else {
            panic!("Call to non-function value {called_name}")
        };
        {
            let callable = &self.functions[callable_index];
    
            if callable.parameters.len() != arguments.len() {
                panic!("Wrong number of arguments for function {called_name}")
            }
           // println!("Calling {called_name}");

        }
        self.make_new_frame(function_name_index, callable_index, arguments);
        //Erase the lifetime of the function pointer. This is a hack, forgive me for I have sinned.
        //Let it wreck havoc on the interpreter state if it wants.
        let function: &LambdaFunction =
            unsafe { std::mem::transmute(&self.functions[callable_index].body) };
        let function_result = function(self);
        //println!("Eval result: {function_result:?} caller: {}", self.frame().function);
        
        if let Value::Trampoline(Closure { callable_index }) = function_result {
            //analyze the resulting trampoline, see if we should execute it or pass along
            let callee = &self.functions[callable_index];
            if let Some(_) = callee.trampoline_of {
                return Value::Trampoline(Closure {callable_index})    
            } else {
                println!("Value returned is a function but is not a TCO trampoline");
                self.pop_frame_and_bindings();
                return function_result;
            }

        } else {
            self.pop_frame_and_bindings(); 
            return function_result;
        }


    }


    pub fn run_trampoline(&mut self, maybe_trampoline: Value) -> Value {
        let mut current @ Value::Trampoline(_) = maybe_trampoline else {
            return maybe_trampoline;
        };

        //the top of the stack here is the stack of the call on the first iteration
        //which needs to be destroyed after the while loop
        let num_stack_frames_before = self.call_stack.len(); 

        while let Value::Trampoline(Closure { callable_index }) = current {
            //println!("state {:#?}", ec.let_bindings);
            let callee = &self.functions[callable_index];
            //println!("Eval result: {function_result:?} caller: {}", self.frame().function);
            
            /*
            In order for the trampoline mechanism to work, we must ensure we're not calling
            the trampoline inside the execution of the trampolined function. This would be the same as
            just calling the function as is. We must end this stack frame (this lambda function)
            by returning the closure value.

            The trampoline function belongs to a recursive function, therefore we can check if the trampoline returned
            belongs to this executing function. Counter intuitively, if that's the case, we *skip* the execution and just return it,
            so that the original caller of the recursive function (that is not itself) can perform the trampoline.
            */

            let function: &LambdaFunction = unsafe { std::mem::transmute(&callee.body) };

            //in theory we are in the same function, so just pass the current function along
            //self.make_trampoline_frame(callable_index);
            //We don't need to setup a new stack frame because when we call this function, 
            //eval_call will run (because it's a *tail call optimization*), which will setup its own call stack.
            //however, we count how many stacks 
            current = function(self);

            //self.pop_frame_and_bindings();
        }

        let num_stack_frames_after = self.call_stack.len();

        let num_created_stack_frames = num_stack_frames_after - num_stack_frames_before;
        let num_frames_to_pop = num_created_stack_frames + 1; //includes the very first one

        for _ in 0 .. num_frames_to_pop {
            self.pop_frame_and_bindings();
        }

        return current;
        
    }


    pub fn run_lambda_trampoline(&mut self, lambda: &LambdaFunction) -> Value {
        let maybe_trampoline = lambda(self);
        return self.run_trampoline(maybe_trampoline);
    }

    fn pop_frame_and_bindings(&mut self) {
        let mut popped_frame = self.pop_frame();
        for let_binding in popped_frame.let_bindings_pushed.iter() {
            self.let_bindings[*let_binding].pop();
        }
        popped_frame.let_bindings_pushed.clear();
        self.reusable_frames.push(popped_frame);
    }

    //This frame returned a trampoline, but the values in the stack
    //are still alive. In order for the trampoline to execute,
    //the execution must end and backtrack to the original caller, which will run
    //the trampoline. However, the values in the stack must still be there, so that 
    //the trampoline can load the function args. However, if we don't do anything, 
    //the trampoline won't execute because of the anti-recursion mechanism that is 
    //based on the current function index, and whose function the trampoline belongs to.

    //Therefore instead of popping the frame, we actually create a new frame with the 
    //trampoline function id. This retains the stack frame, emulates recursion, and 
    //changes the current function ID so that the trampoline is allowed to run.
    fn make_trampoline_frame(&mut self, callable_index: usize) {
        let new_frame = match self.reusable_frames.pop() {
            Some(mut new_frame) => {
                self.stats.reused_frames += 1;
                new_frame.function = callable_index;
                new_frame
            }
            None => {
                self.stats.new_frames += 1;

                StackFrame {
                    function: callable_index,
                    let_bindings_pushed: vec![],
                }
            }
        };
        self.push_frame(new_frame);
    }

    fn make_new_frame(&mut self, function_name_index: usize, callable_index: usize, arguments: &[LambdaFunction]) {
        let mut new_frame = match self.reusable_frames.pop() {
            Some(mut new_frame) => {
                self.stats.reused_frames += 1;
                new_frame.function = callable_index;
                new_frame
            }
            None => {
                self.stats.new_frames += 1;

                StackFrame {
                    function: callable_index,
                    let_bindings_pushed: vec![],
                }
            }
        };

        //In this new stack frame, we push the function name, so we can do recursion.
        //To comply with the rest of the interpreter logic we also push the function name into the let bindings

        new_frame.let_bindings_pushed.push(function_name_index);
        self.let_bindings[function_name_index].push(Value::Closure(Closure { callable_index }));
        {
            let params = self.functions[callable_index].parameters;
            //evaluate the arguments
            for (argument, param) in arguments.iter().zip(params) {
                let value = argument(self);
                new_frame.let_bindings_pushed.push(*param);
                self.let_bindings[*param].push(value);
            }
        }
        self.push_frame(new_frame);
    }

    fn eval_tuple(
        &mut self,
        evaluate_first: &LambdaFunction,
        evaluate_second: &LambdaFunction,
    ) -> Value {
        let f = evaluate_first(self);
        let s = evaluate_second(self);
        match (f, s) {
            (Value::Int(a), Value::Int(b)) => Value::IntTuple(a, b),
            (Value::Bool(a), Value::Bool(b)) => Value::BoolTuple(a, b),
            (Value::Int(a), Value::Bool(b)) => Value::IntBoolTuple(a, b),
            (Value::Bool(a), Value::Int(b)) => Value::BoolIntTuple(a, b),
            (a, b) => Value::DynamicTuple(Box::new(a), Box::new(b)),
        }
    }
}

pub struct LambdaCompiler {
    //this is added into during compilation, but should be moved into the execution context later
    pub all_functions: Vec<Callable>,
    pub var_names: Vec<&'static str>,
    pub strict_equals: bool,
    pub closure_stack: Vec<usize>
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
            var_names: vec![],
            closure_stack: vec![usize::MAX]
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

    fn compile_internal(&mut self, ast: Expr) -> LambdaFunction {
        let lambda: LambdaFunction = match ast {
            Expr::Print{ value } => {
                let evaluate_printed_value = self.compile_internal(*value);
                //because the type of the expression is only known at runtime, we have to check it in the print function during runtime :(
                Box::new(move |ec: &mut ExecutionContext| {
                    //println!("Stack frame on print: {:?}", ec.frame().let_bindings);
                    let value_to_print = evaluate_printed_value(ec);
                    println!("{}", value_to_print.to_string());
                    return value_to_print
                })
            }
            Expr::Int{value } => Box::new(move |ec| ec.eval_int(value)),
            Expr::Var{ name, .. } => {
                let varname_leaked: &'static str = name.leak();
                let index = self.intern_string(varname_leaked);

                Box::new(move |ec: &mut ExecutionContext| ec.eval_var(index, varname_leaked))
            }
           
            Expr::Let{
                name,
                value,
                next
            } => {

                if let Expr::FuncDecl(FuncDecl { params, body, ..  }) = &*value && name != "_" && self.is_tail_recursive(&body, &name) {
                    //the recursive call becomes a closure
                    let trampolined_function = self.into_tco(&body, &name);

                    //println!("Function got TCO'd {trampolined_function:#?}");
                    let evaluate_value = self.compile_function(trampolined_function, params, true, None);
                    let evaluate_next = self.compile_internal(*next);
                    let var_name_leaked: &'static str = name.leak();
                    let var_index = self.intern_string(var_name_leaked);
                    Box::new(move |ec: &mut ExecutionContext| {
                        //println!("Evaluating Let {var_name_leaked} TCO");
                        ec.eval_let(&evaluate_value, &evaluate_next, var_index)
                    })
                    
                }
                else {
                    let evaluate_value = self.compile_internal(*value);
                    let evaluate_next = self.compile_internal(*next);
                    if name == "_" {
                        Box::new(move |ec: &mut ExecutionContext| {
                            evaluate_value(ec);
                            let next = evaluate_next(ec);
                            return next;
                        })
                    } else {
                        let var_name_leaked: &'static str = name.leak();
                        let var_index = self.intern_string(var_name_leaked);
                        Box::new(move |ec: &mut ExecutionContext| {
                            //println!("Evaluating Let {var_name_leaked} non-tco");
                            //continuation into recursive call in next
                            ec.eval_let(&evaluate_value, &evaluate_next, var_index)
                        })
                    }
                }
            }
           
           // Expr::Error(e) => panic!("{}: {}", e.message, e.full_text),
            Expr::BinOp { op, left, right } => self.compile_binexp(left, right, op),
            Expr::String{value} => {
                let leaked: &'static str = value.leak();
                Box::new(move |_| Value::Str(leaked))
            }
            Expr::FuncCall {
                func, args
            } => match &*func {
                Expr::Var {name} => {
                    let called_name: &'static str = name.clone().leak();
                    let function_name_index = self.intern_string(called_name);
                    let evaluate_callee = self.compile_internal(*func);
                    let arguments = args
                        .into_iter()
                        .map(|arg| self.compile_internal(arg))
                        .collect::<Vec<_>>();
                    Box::new(move |ec: &mut ExecutionContext| {
                        //println!("Calling {called_name} {}", ec.frame().function);
                        let result = ec.eval_call(
                            &evaluate_callee,
                            &arguments,
                            function_name_index,
                            called_name,
                        );
                        //println!("Called {called_name} result = {result:?}");
                        result
                    })
                }
                _ => panic!("Cannot call non-var term"),
            },
            Expr::FuncDecl(FuncDecl {
                params, body, is_tco_trampoline,
            }) => {
                let trampoline = if is_tco_trampoline {
                    Some(self.closure_stack.last().copied().unwrap())
                } else {
                    None
                };
                self.compile_function(*body, &params, false, trampoline)
            }
            Expr::If {
                cond,
                then,
                otherwise,
                ..
            } => {
                let evaluate_condition = self.compile_internal(*cond);
                let evaluate_then = self.compile_internal(*then);
                let evaluate_otherwise = self.compile_internal(*otherwise);
                Box::new(move |ec: &mut ExecutionContext| {
                    ec.eval_if(&evaluate_condition, &evaluate_then, &evaluate_otherwise)
                })
            }
            //only works on tuples
            Expr::First{ value, .. } => {
                let evaluate_value = self.compile_internal(*value);
                Box::new(move |ec: &mut ExecutionContext| {
                    let value = evaluate_value(ec);
                    match value {
                        Value::DynamicTuple(a, _) => *a,
                        Value::IntTuple(a, _) => Value::Int(a),
                        Value::BoolTuple(a, _) => Value::Bool(a),
                        Value::BoolIntTuple(a, _) => Value::Bool(a),
                        Value::IntBoolTuple(a, _) => Value::Int(a),
                        _ => panic!("Type error: Cannot evaluate first on this value: {value:?}"),
                    }
                })
            }
            Expr::Second{ value, .. } => {
                let evaluate_value = self.compile_internal(*value);
                Box::new(move |ec: &mut ExecutionContext| {
                    let value = evaluate_value(ec);
                    match value {
                        Value::DynamicTuple(_, a) => *a,
                        Value::IntTuple(_, a) => Value::Int(a),
                        Value::BoolTuple(_, a) => Value::Bool(a),
                        Value::BoolIntTuple(_, a) => Value::Int(a),
                        Value::IntBoolTuple(_, a) => Value::Bool(a),
                        _ => panic!("Type error: Cannot evaluate first on this value: {value:?}"),
                    }
                })
            }
            Expr::Bool { value, .. } => Box::new(move |_| Value::Bool(value)),
            Expr::Tuple { first, second, .. } => {
                let evaluate_first = self.compile_internal(*first);
                let evaluate_second = self.compile_internal(*second);
                Box::new(move |ec: &mut ExecutionContext| {
                    ec.eval_tuple(&evaluate_first, &evaluate_second)
                })
            }
        };

        return lambda;
        /*if do_trampolining {
            Box::new(move |ec| {
                let result = lambda(ec);
                let trampoline_result = ec.run_trampoline(result);
                return trampoline_result;
            })
        } else {
            lambda
        }*/

    }

    fn compile_function(&mut self, value: Expr, parameters: &[String], tco: bool, trampoline_of: Option<usize>) -> LambdaFunction {
        //Functions are compiled and stored into a big vector of functions, each one has a unique ID.
        //value will be the call itself, we're compiling the body of the tco trampoline
        //println!("Compile function: {value:#?} location: {location:?}");
        let index_of_new_function = self.all_functions.len();
        let callable = Callable {
            body: Box::new(|_| panic!("Empty body, function not compiled")),
            parameters: &[],
            tco_optimized: false,
            trampoline_of: None
            //location
        };
        self.all_functions.push(callable);
        self.closure_stack.push(index_of_new_function);

        //this will return the lambda for the call to iter
        let new_function = self.compile_internal(value);

        let parameters = parameters
            .into_iter()
            .map(|param| param.clone())
            .collect::<Vec<_>>();
        let mut param_indices = vec![];
        for param in parameters {
            let interned = self.intern_string(param.leak());
            param_indices.push(interned);
        }
        let callable = self.all_functions.get_mut(index_of_new_function).unwrap();
        *callable = Callable {
            body: new_function,
            parameters: param_indices.leak(),
            tco_optimized: tco,
            trampoline_of,
            //location
        };
        self.closure_stack.pop();
        if trampoline_of.is_none() {
            Box::new(move |ec: &mut ExecutionContext| {
                //println!("Evaluating function at location {}", loc);
                ec.eval_closure(index_of_new_function)
            })
        } else {
            Box::new(move |_| {
                //println!("Evaluating function at location {}", loc);
                Value::Trampoline(Closure { callable_index: index_of_new_function })
            })
        }
    }

    //only call when is_tail_recursive is true
    fn into_tco(&mut self, value: &Expr, fname: &str) -> Expr {
        match value {
            Expr::Let{ name, value, next } => {
                Expr::Let { name: name.clone(), value: value.clone(), next: self.into_tco(next, fname).into() }
            }
            Expr::If { cond, then, otherwise } => {
                Expr::If { cond: cond.clone(), then: self.into_tco(then, fname).into(), otherwise: self.into_tco(&otherwise, fname).into() }
            },
            call @ Expr::FuncCall { func, .. } => {
                match &**func {
                    Expr::Var{ name, .. } if name == fname => { 
                        Expr::FuncDecl(FuncDecl {
                            params: vec![],
                            body: call.clone().into(), 
                            is_tco_trampoline: true
                        }) // fn() => iter(from+1, ...)
                    },
                    _ =>call.clone()
                }
            }
            t => t.clone() 
        }
    }

    pub fn is_tail_recursive(&mut self, value: &Expr, fname: &str) -> bool {
        //println!("Checking if is tail recursive: {fname} \n${value:#?}");
        match value {
            Expr::FuncCall { func, .. } => {
                match &**func {
                    Expr::Var{ name, .. } if name == fname => true,
                    _ => false
                }
            }
            Expr::Let{ next, .. } => {
                self.is_tail_recursive(next, fname)
            }
            Expr::If { then, otherwise, .. } => {
                self.is_tail_recursive(then, fname) || self.is_tail_recursive(otherwise, fname)
            }
            _ => false
        }
    }

    pub fn compile(mut self, ast: Expr) -> CompilationResult {
        let main = self.compile_internal(ast);
        //trampolinize the returned value
        let trampolinized = Box::new(move |ec: &mut ExecutionContext| {
            ec.run_lambda_trampoline(&main)
        });
        CompilationResult {
            main: trampolinized,
            strings: self.var_names,
            functions: self.all_functions,
        }
    }

    fn compile_binexp(
        &mut self,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        op: BinaryOp,
    ) -> LambdaFunction {
        let evaluate_lhs = self.compile_internal(*lhs);
        let evaluate_rhs = self.compile_internal(*rhs);

        macro_rules! int_binary_numeric_op {
            ($op:tt) => {
                Box::new(move |ec: &mut ExecutionContext| {
                    let lhs = ec.run_lambda_trampoline(&evaluate_lhs);
                    let rhs = ec.run_lambda_trampoline(&evaluate_rhs);
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
                        let lhs = ec.run_lambda_trampoline(&evaluate_lhs);
                        let rhs = ec.run_lambda_trampoline(&evaluate_rhs);

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
                        let lhs = ec.run_lambda_trampoline(&evaluate_lhs);
                        let rhs = ec.run_lambda_trampoline(&evaluate_rhs);

                        return Value::Bool(lhs $op rhs);
                    })
                }
            };
        }

        match op {
            BinaryOp::Add => Box::new(move |ec: &mut ExecutionContext| {
                ec.binop_add(&evaluate_lhs, &evaluate_rhs)
            }),
            BinaryOp::Sub => int_binary_numeric_op!(-),
            BinaryOp::Mul => Box::new(move |ec: &mut ExecutionContext| {
                ec.binop_mul(&evaluate_lhs, &evaluate_rhs)
            }),
            BinaryOp::Div => int_binary_numeric_op!(/),
            BinaryOp::Rem => int_binary_numeric_op!(%),
            BinaryOp::Eq => binary_comparison_op!(==),
            BinaryOp::Neq => binary_comparison_op!(!=),
            BinaryOp::Lt => binary_comparison_op!(<),
            BinaryOp::Gt => binary_comparison_op!(>),
            BinaryOp::Lte => binary_comparison_op!(<=),
            BinaryOp::Gte => binary_comparison_op!(>=),
            BinaryOp::And => Box::new(move |ec: &mut ExecutionContext| {
                let lhs = ec.run_lambda_trampoline(&evaluate_lhs);
                let rhs = ec.run_lambda_trampoline(&evaluate_rhs);
                match (&lhs, &rhs) {
                    (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(*lhs && *rhs),
                    _ => panic!("Type error: Cannot apply binary operator && (and) on these values: {lhs:?} and {rhs:?}"),
                }
            }),
            BinaryOp::Or => Box::new(move |ec: &mut ExecutionContext| {
                let lhs = ec.run_lambda_trampoline(&evaluate_lhs);
                let rhs = ec.run_lambda_trampoline(&evaluate_rhs);
                match (&lhs, &rhs) {
                    (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(*lhs || *rhs),
                    _ => panic!("Type error: Cannot apply binary operator || (or) on these values: {lhs:?} and {rhs:?}"),
                }
            }),
        }
    }
}
