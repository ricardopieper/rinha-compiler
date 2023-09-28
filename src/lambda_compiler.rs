

use std::collections::{HashSet, BTreeMap};

use crate::ast::BinaryOp;

use crate::hir::{Expr, FuncDecl};

pub type LambdaFunction = Box<dyn Fn(&mut ExecutionContext) -> Value>;

#[derive(Debug)]
pub struct Stats {
    pub new_frames: usize,
    pub reused_frames: usize,
}

#[derive(Clone,  Copy,Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Closure {
    pub callable_index: usize,
    pub closure_env_index: usize
    //pub environment: BTreeMap<usize, Value>
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct HeapPointer(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct StringPointer(u32);


#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct ClosurePointer(u32);


#[derive(Clone,  Copy,Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct TuplePointer(u32);

/// Enum for runtime values
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum Value {
    Int(i32),
    Bool(bool),
    Str(StringPointer),
    Tuple(TuplePointer),
    SmallTuple(i16, i16),
    Closure(ClosurePointer),
    Trampoline(ClosurePointer),
}


impl Value {
    fn to_string(&self, ec: &ExecutionContext) -> String {
        match self {
            Value::Trampoline(_) => "trampoline".to_string(),
            Value::Int(i) => i.to_string(),
            Value::Str(StringPointer(s)) => {
                ec.string_values[*s as usize].to_string()
            }
            Value::Tuple(TuplePointer(t)) => {
                let (f, s) = &ec.tuples[*t as usize];
                let a = &ec.heap[f.0 as usize];
                let b = &ec.heap[s.0 as usize];
                format!("({}, {})", a.to_string(ec), b.to_string(ec))
            }
            Value::SmallTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            /*Value::BoolTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::IntBoolTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::BoolIntTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),*/
            Value::Closure(..) => "function".to_string(),
            //Value::Error(e) => format!("error: {}", e),
           // Value::Unit => "unit".to_string(),
            Value::Bool(b) => b.to_string(),
        }
    }
}

pub struct StackFrame {
    function: usize,
    pub let_bindings_pushed: Vec<usize>,
    pub closure_environment: usize,
    pub tco_reuse_frame: bool
}

pub struct Callable {
    pub parameters: &'static [usize],
    pub closure_indices: &'static [usize],
    pub body: LambdaFunction,
    //pub location: Location,
    pub tco_optimized: bool,
    pub trampoline_of: Option<usize>, //the callable index this callable belongs to,
   
}

pub struct ExecutionContext<'a> {
    pub call_stack: Vec<StackFrame>,
    pub functions: &'a [Callable],
    pub enable_memoization: bool,
    //First vec represents the variable itself (there's only a set number of possible values),
    //second vec represents the stack frame's value of that variable
    pub let_bindings: Vec<Vec<Value>>,
    pub arg_eval_buffer: Vec<Vec<Value>>,
    pub reusable_frames: Vec<StackFrame>,
    pub closure_environments: Vec<BTreeMap<usize, Value>>,
    pub string_values: Vec<String>,
    //most things are done in the "stack" for some definition of stack which is very loose here....
    //this heap is for things like dynamic tuples that could very well be infinite.
    //This makes the Value enum smaller than storing boxes directly
    pub heap: Vec<Value>,
    pub tuples: Vec<(HeapPointer, HeapPointer)>,
    pub closures: Vec<Closure>
   
}

pub struct CompilationResult {
    pub main: LambdaFunction,
    pub strings: Vec<String>,
    pub functions: Vec<Callable>,
}

impl<'a> ExecutionContext<'a> {
    pub fn new(program: &'a CompilationResult) -> Self {

        //whenever we want an empty closure, we'll access by the function index into closures without allocating a new one
        let empty_closures = {
            let mut closures = vec![];
            for (i, _) in program.functions.iter().enumerate() {
                closures.push(Closure { callable_index: i, closure_env_index: usize::MAX })
            }
            closures
        };

        Self {
            call_stack: vec![StackFrame {
                function: usize::MAX,
                let_bindings_pushed: vec![],
                closure_environment: usize::MAX,
                tco_reuse_frame: false
            }],
            arg_eval_buffer: vec![],
            functions: &program.functions,
            enable_memoization: true,
            let_bindings: vec![vec![]; program.strings.len()],
            reusable_frames: vec![],
            closure_environments: vec![],
            heap: vec![],
            string_values: vec![],
            tuples: vec![],
            closures: empty_closures
            
        }
    }

    pub fn disable_memoization(&mut self) {
        self.enable_memoization = false;
    }

    pub fn frame_mut(&mut self) -> &mut StackFrame {
        self.call_stack.last_mut().unwrap()
    }
    pub fn frame(&self) -> &StackFrame {
        self.call_stack.last().unwrap()
    }
    pub fn pop_frame(&mut self) -> StackFrame {
        self.call_stack.pop().unwrap()
    }
    pub fn push_frame(&mut self, frame: StackFrame) {
        self.call_stack.push(frame);
    }

    fn eval_closure_no_env(&mut self, index_of_new_function: usize) -> ClosurePointer {
        return ClosurePointer(index_of_new_function as u32)
    }

    fn eval_closure_with_env(&mut self, index_of_new_function: usize) -> ClosurePointer {

        let function = self.functions.get(index_of_new_function).unwrap();
        let closure = function.closure_indices;

       //start copy of values into environment
        let mut environment = BTreeMap::new();
        for idx in closure {
            if let Some(v)  = self.let_bindings.get(*idx).unwrap().last() {
                environment.insert(*idx, 
                    *v
                );
            }
        }
        //println!("{:#?}", environment);
        let current_id = self.closure_environments.len();
        self.closure_environments.push(environment);
 
        let closure = Closure {
            callable_index: index_of_new_function, closure_env_index: current_id
        };
        let closure_p = self.closures.len();
        self.closures.push(closure);
        return ClosurePointer(closure_p as u32);

    }

    fn eval_int(&self, value: i32) -> Value {
        return Value::Int(value);
    }

    fn eval_let(
        &mut self,
        evaluate_value: &LambdaFunction,
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
            self.frame_mut().let_bindings_pushed.push(var_index);
        }
        return bound_value;
    }

    fn eval_var(&self, index: usize) -> Value {
        //@PERF no clone pls
        //println!("Trying to get var {index} {varname}");
        let var_stack = self.let_bindings.get(index);
        match var_stack {
            Some(value) => match value.last() {
                Some(s) => {
                    //println!("Loading var {varname} {index}");
                    *s
                }
                None => {
                    //try to load from environment
                    let closure_env = self.frame().closure_environment;
                    let env = &self.closure_environments[closure_env];
                    //println!("Loading closure var {varname} {index}");

                    if let Some(s) = env.get(&index) {
                        *s
                    } else {
                        let var_name = &self.string_values[index];
                        panic!("Variable {var_name} not found")
                    }
                }
            }, //will only be cloned and heap allocated on complex structures
            None => {
                let var_name = &self.string_values[index];
                panic!("Variable {var_name} not found")
            }
        }
    }

    fn eval_var_ref(&self, index: usize) -> &Value {
        let var_stack = self.let_bindings.get(index);

        match var_stack {
            Some(value) => match value.last() {
                Some(s) => {
                    //println!("Loading var {varname} {index}");
                    s
                }
                None => {
                    //try to load from environment
                    let closure_env = self.frame().closure_environment;
                    let env = &self.closure_environments[closure_env];
                    //println!("Loading closure var {varname} {index}");

                    if let Some(s) = env.get(&index) {
                        s
                    } else {
                        let var_name = &self.string_values[index];
                        panic!("Variable {var_name} not found")
                    }
                }
            }, //will only be cloned and heap allocated on complex structures
            None => {
                let var_name = &self.string_values[index];
                panic!("Variable {var_name} not found")
            }
        }
    }

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
            (Value::Int(lhs), Value::Int(rhs)) => {
                //println!("Sum {lhs} {rhs}");
                Value::Int(lhs + rhs)
            }
            (Value::Int(lhs), Value::Bool(rhs)) => Value::Int(lhs + *rhs as i32),
            (lhs, rhs @ Value::Str(..)) | (lhs @ Value::Str(..), rhs) => {
                let str = format!("{}{}", lhs.to_string(self), rhs.to_string(self));
                let current = StringPointer(self.string_values.len() as u32);
                self.string_values.push(str);
                Value::Str(current)
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
        function_name_index: Option<usize>,
        called_name: &str,
    ) -> Value {
        let callee_function = evaluate_callee(self);
        let Value::Closure(ClosurePointer(p)) = callee_function else {
            panic!("Call to non-function value {called_name}")
        };
        let Closure { callable_index, closure_env_index } = self.closures[p as usize];
        let callable = &self.functions[callable_index as usize];

    

        if callable.parameters.len() != arguments.len() {
            panic!("Wrong number of arguments for function {called_name}")
        }
        // println!("Calling {called_name}");

        
        let mut reuse_frame = false;
        
       // println!("Running trampolines: {:?}", self.running_trampolines);

        if self.frame().tco_reuse_frame {
            if self.frame().function == callable_index {
                reuse_frame = true;
            }
        }

        if reuse_frame {
            self.update_let_bindings_for_frame_reuse(arguments, callable);
        } else {
            self.make_new_frame(function_name_index, callable_index, closure_env_index, ClosurePointer(p), arguments);
        }

        //Erase the lifetime of the function pointer. This is a hack, forgive me for I have sinned.
        //Let it wreck havoc on the interpreter state if it wants.
        let function: &LambdaFunction =
            unsafe { std::mem::transmute(&self.functions[callable_index as usize].body) };
        let function_result = function(self);
        //println!("Eval result: {function_result:?} caller: {}", self.frame().function);
        
        if let Value::Trampoline(ClosurePointer(p)) = &function_result {
            //analyze the resulting trampoline, see if we should execute it or pass along
            let Closure { callable_index, .. } = self.closures[*p as usize];
            let callee = &self.functions[callable_index];
            if let Some(_) = callee.trampoline_of {
                return function_result;  
            } else {
                //println!("Value returned is a function but is not a TCO trampoline");
                self.pop_frame_and_bindings();
                return function_result;
            }

        } else {
            if !reuse_frame {
                self.pop_frame_and_bindings();
            } 
            return function_result;
        }


    }

    fn update_let_bindings_for_frame_reuse(&mut self, arguments: &[LambdaFunction], callable: &Callable) {
        let popped_argbuf = self.arg_eval_buffer.pop();

        //we are re-calling ourselves (a.k.a recursion) but we are in a trampoline,
        //therefore we can only update the let bindings.
        //we can reload the arguments and set them in the let bindings
        let mut buf = if let Some(b) = popped_argbuf {
            b
        } else {
            vec![]
        };

        buf.clear();
        arguments.iter()
            .map(|argument| (argument(self)))
            .collect_into(&mut buf);
        //evaluate the arguments
        for (argument, param) in buf.iter_mut().zip(callable.parameters) {
            *self.let_bindings[*param].last_mut().unwrap() = argument.clone()
        }
        self.arg_eval_buffer.push(buf);
    }


    pub fn run_trampoline(&mut self, maybe_trampoline: Value) -> Value {
        let mut current @ Value::Trampoline(..) = maybe_trampoline else {
            return maybe_trampoline;
        };

        //the top of the stack here is the stack of the call on the first iteration
        //which needs to be destroyed after the while loop
        let num_stack_frames_before = self.call_stack.len(); 

        //even TCO'd functions have environments and they must be passed along somehow.
        //The tail function call might have referenced something other than the current function args...
        let mut tco_env_set = false;

        while let Value::Trampoline(ClosurePointer(p)) = current {
            let Closure { callable_index, closure_env_index } = self.closures[p as usize];

            if !tco_env_set {
                self.frame_mut().tco_reuse_frame = true;
                //TCO hack: we modify the current stack instead of creating a new one because
                //this is faster and TCO is already a huge hack.
                self.frame_mut().closure_environment = closure_env_index as usize;
                tco_env_set = true;

            }

            //println!("state {:#?}", ec.let_bindings);
            let callee = &self.functions[callable_index as usize];
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

            //We don't need to setup a new stack frame because when we call this function, 
            //eval_call will run (because it's a *tail call optimization*), which will setup its own call stack.
            //however, we count how many stacks we pushed and pop them all later

        

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

    fn make_new_frame(&mut self, function_name_index: Option<usize>, callable_index: usize, closure_env_index: usize, 
        closure_pointer: ClosurePointer, arguments: &[LambdaFunction]) {
        let mut new_frame = match self.reusable_frames.pop() {
            Some(mut new_frame) => {
                new_frame.function = callable_index as usize;
                new_frame.closure_environment = closure_env_index as usize;
                new_frame.tco_reuse_frame = false;
                new_frame
            }
            None => {
                StackFrame {
                    function: callable_index as usize,
                    let_bindings_pushed: vec![],
                    closure_environment: closure_env_index as usize,
                    tco_reuse_frame: false
                }
            }
        };

        //In this new stack frame, we push the function name, so we can do recursion.
        //To comply with the rest of the interpreter logic we also push the function name into the let bindings
        if let Some(function_name_index) = function_name_index {
            new_frame.let_bindings_pushed.push(function_name_index);
            self.let_bindings[function_name_index].push(Value::Closure(closure_pointer));
        }
        {
            let params = self.functions[callable_index as usize].parameters;
            
            let mut buf = if let Some(b) = self.arg_eval_buffer.pop() {
                b
            } else {
                vec![]
            };

            buf.clear();
            arguments.iter()
                .map(|argument| (argument(self)))
                .collect_into(&mut buf);
                //evaluate the arguments
            for (argument, param) in buf.iter().zip(params) {
                new_frame.let_bindings_pushed.push(*param);
                self.let_bindings[*param].push(argument.clone());
            }
            self.arg_eval_buffer.push(buf);
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
            (Value::Int(a), Value::Int(b)) => {
                //transform into i16 if in range
                if a < i16::MAX as i32 && a > i16::MIN as i32 && b < i16::MAX as i32 && b > i16::MIN as i32 {
                    Value::SmallTuple(a as i16, b as i16)
                } else {
                    let heap_a = self.heap.len();
                    self.heap.push(Value::Int(a));
                    let heap_b = self.heap.len();
                    self.heap.push(Value::Int(b));

                    let tuple_p = self.tuples.len();
                    self.tuples.push( 
                        (HeapPointer(heap_a as u32),
                         HeapPointer(heap_b as u32)));

                    Value::Tuple(
                       TuplePointer(tuple_p as u32)
                    )
                }

            }/*
            (Value::Bool(a), Value::Bool(b)) => Value::BoolTuple(a, b),
            (Value::Int(a), Value::Bool(b)) => Value::IntBoolTuple(a, b),
            (Value::Bool(a), Value::Int(b)) => Value::BoolIntTuple(a, b),*/
            (a, b) => {
                let heap_a = self.heap.len();
                self.heap.push(a);
                let heap_b = self.heap.len();
                self.heap.push(b);

                let tuple_p = self.tuples.len();
                self.tuples.push( 
                    (HeapPointer(heap_a as u32),
                     HeapPointer(heap_b as u32)));

                Value::Tuple(
                   TuplePointer(tuple_p as u32)
                )
            }
        }
    }
}

pub struct LambdaCompiler {
    //this is added into during compilation, but should be moved into the execution context later
    pub all_functions: Vec<Callable>,
    pub var_names: Vec<String>,
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

    pub fn intern_var_name(&mut self, s: &str) -> usize {
        //check if it already exists
        if let Some(index) = self.var_names.iter().position(|x| x == &s) {
            return index;
        }

        let index = self.var_names.len();
        self.var_names.push(s.to_string());
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
    //also don't closure over let bindings

    
    fn get_closure(&mut self, body: &[&Expr], params: &mut HashSet<String>, current_fn_let_bindings: &mut HashSet<String>) -> HashSet<String> {
        let mut closure = HashSet::new();
        for expr in body {
            match expr {
                Expr::Let { value, name } =>{
                    current_fn_let_bindings.insert(name.to_string());
                    let value_closure = self.get_closure(&[value], params, current_fn_let_bindings);
                    closure.extend(value_closure);
                }
                Expr::Var { name } => {
                    if !(params.contains(name) || current_fn_let_bindings.contains(name)) {
                        closure.extend(vec![name.to_string()].into_iter());
                    }
                }
                Expr::If { cond, then, otherwise } =>{
                    let mut cond_closure = self.get_closure(&[cond], params, current_fn_let_bindings);
                    let then_closure = self.get_closure(&as_ref_slice(then), params, current_fn_let_bindings);
                    let otherwise_closure = self.get_closure(&as_ref_slice(otherwise), params, current_fn_let_bindings);
                    cond_closure.extend(then_closure);
                    cond_closure.extend(otherwise_closure);
                    closure.extend(cond_closure)
                }
                Expr::Tuple { first, second } => {
                    let mut first_closure = self.get_closure(&[first], params, current_fn_let_bindings);
                    let second_closure = self.get_closure(&[second], params, current_fn_let_bindings);
                    first_closure.extend(second_closure);

                    closure.extend(first_closure)
                }
                Expr::First { value } => {
                    closure.extend(self.get_closure(&[value], params, current_fn_let_bindings))
                }
                Expr::Second { value } => {
                    closure.extend(self.get_closure(&[value], params, current_fn_let_bindings))
                }
                Expr::Print { value } => {
                    closure.extend(self.get_closure(&[value], params, current_fn_let_bindings))
                }
                Expr::FuncDecl(FuncDecl { is_tco_trampoline, .. }) if *is_tco_trampoline => {
                
                    //this may be an extreme oversimplification, but tco trampolines don't close over anything
                    //they actually do but for them we can probably get away with it
                  
                }
                Expr::FuncDecl(FuncDecl { params: func_params, body, .. }) => {
                    //the analysis of regular funcdecls, not trampolines, is done by analyzing it in isolation, 
                    //does not matter the context it is in.
                    //but in the case of a TCO trampoline, we detect the variables in params that were used in the call, and don't closure over them again
                    //we do this by extending the current params and passing along the chain
                    //the funcdecl itself has 0 parameters
                    //thi is an optimization, ```trampolines are generated in a very controlled way
                    let mut new_function_params = HashSet::new();
                    for p in func_params {
                        new_function_params.insert(p.clone());
                    }
                    let mut new_let_bindings = HashSet::new();
                    closure.extend(self.get_closure(&as_ref_slice(&body), &mut new_function_params, &mut new_let_bindings))
                }
                Expr::FuncCall { func, args } => {
                    let mut f_closure = HashSet::new();

                    f_closure.extend(self.get_closure(&[func], params, current_fn_let_bindings));

                    for a in args {
                        f_closure.extend(self.get_closure(&[a], params, current_fn_let_bindings));
                    }
                    closure.extend(f_closure)
                }
                Expr::BinOp { left, right, .. } => {
                    let mut lhs_closure = self.get_closure(&[left], params, current_fn_let_bindings);
                    let rhs_closure = self.get_closure(&[right], params, current_fn_let_bindings);
                    lhs_closure.extend(rhs_closure);
                    closure.extend(lhs_closure)
                }
                _ => {}
            }
        }
        closure
    }

    fn compile_body(&mut self, body: &[Expr], funcs_params: &mut HashSet<String>) -> LambdaFunction {
        let mut lambdas = vec![];
        for ast in body {
            lambdas.push(self.compile_internal(ast.clone(), funcs_params));
        }
        Box::new(move |ec: &mut ExecutionContext| {
            let mut result = None;
            for lambda in lambdas.iter() {
                result = Some(lambda(ec));
            }
            result.unwrap()
        })
    }

    fn compile_internal(&mut self, ast: Expr, funcs_params: &mut HashSet<String>) -> LambdaFunction {
        let lambda: LambdaFunction = match ast {
            Expr::Print{ value } => {
                let evaluate_printed_value = self.compile_internal(*value, funcs_params);
                //because the type of the expression is only known at runtime, we have to check it in the print function during runtime :(
                Box::new(move |ec: &mut ExecutionContext| {
                    //println!("Stack frame on print: {:?}", ec.frame().let_bindings);
                    let value_to_print =  ec.run_lambda_trampoline(&evaluate_printed_value);
                    println!("{}", value_to_print.to_string(ec));
                    return value_to_print
                })
            }
            Expr::Int{value } => Box::new(move |ec| ec.eval_int(value)),
            Expr::Var{ name, .. } => {
                let varname_leaked: &'static str = name.leak();
                let index = self.intern_var_name(varname_leaked);

                Box::new(move |ec: &mut ExecutionContext| ec.eval_var(index))
            }
           
            Expr::Let{
                name,
                value,
               // next
            } => {

                if let Expr::FuncDecl(FuncDecl { params, body, ..  }) = &*value && name != "_" && self.is_tail_recursive(&body, &name) {
                    //the recursive call becomes a closure
                    let trampolined_function = self.into_tco(&body, &name);
                    let mut all_function_params = HashSet::new();
                    for p in params {
                        all_function_params.insert(p.to_string());
                    }
                    let mut let_bindings = HashSet::new();
                    let closure = self.get_closure(&as_ref_slice(&trampolined_function), &mut all_function_params, &mut let_bindings);
                    let closure_set: Vec<_> = closure.into_iter().collect();
    
                    //println!("Function got TCO'd {trampolined_function:#?}");
                    let evaluate_value = self.compile_function(funcs_params, trampolined_function, params, &closure_set, true, None);

                    let var_name_leaked: &'static str = name.leak();
                    let var_index = self.intern_var_name(var_name_leaked);
                    Box::new(move |ec: &mut ExecutionContext| {
                        //println!("Evaluating Let {var_name_leaked} TCO");
                        ec.eval_let(&evaluate_value, var_index)
                    })
                    
                }
                else {
                    let evaluate_value = self.compile_internal(*value, funcs_params);
                    if name == "_" {
                        Box::new(move |ec: &mut ExecutionContext| {
                            ec.run_lambda_trampoline(&evaluate_value)
                        })
                    } else {
                        let var_name_leaked: &'static str = name.leak();
                        let var_index = self.intern_var_name(var_name_leaked);
                        Box::new(move |ec: &mut ExecutionContext| {
                            //println!("Evaluating Let {var_name_leaked} non-tco");
                            //continuation into recursive call in next
                            ec.eval_let(&evaluate_value, var_index)
                        })
                    }
                }
            }
           
           // Expr::Error(e) => panic!("{}: {}", e.message, e.full_text),
            Expr::BinOp { op, left, right } => self.compile_binexp(funcs_params, left, right, op),
            Expr::String{value} => {
                //let leaked: &'static str = value.leak();
                Box::new(move |ec| {
                    let current = ec.string_values.len();
                    ec.string_values.push(value.clone());
                    Value::Str(StringPointer(current as u32))
                })
            }
            Expr::FuncCall {
                func, args
            } => {
                let (callee, arguments, name, called_name) = self.compile_call_data(&func, funcs_params, &args);
                Box::new(move |ec: &mut ExecutionContext| {
                    //println!("Calling {called_name} {}", ec.frame().function);
                    let result = ec.eval_call(
                        &callee,
                        &arguments,
                        name,
                        called_name,
                    );
                    //println!("Called {called_name} result = {result:?}");
                    result
                })
            },
            fdecl @ Expr::FuncDecl(..) => {
                //let it find the function params
                let mut all_function_params = funcs_params.clone();
                let mut let_bindings = HashSet::new();
                let closure = self.get_closure(&[&fdecl], &mut all_function_params, &mut let_bindings);
                let Expr::FuncDecl(FuncDecl {
                    params, body, is_tco_trampoline,
                }) = fdecl else {
                    unreachable!()
                };
                
                let closure_set: Vec<_> = closure.into_iter().collect();

                let trampoline = if is_tco_trampoline {
                    Some(self.closure_stack.last().copied().unwrap())
                } else {
                    None
                };
                self.compile_function(funcs_params, body, &params, &closure_set, false, trampoline)
            }
            Expr::If {
                cond,
                then,
                otherwise,
                ..
            } => {
                let evaluate_condition = self.compile_internal(*cond, funcs_params);
                let evaluate_then = self.compile_body(&then, funcs_params);
                let evaluate_otherwise = self.compile_body(&otherwise, funcs_params);
                Box::new(move |ec: &mut ExecutionContext| {
                    ec.eval_if(&evaluate_condition, &evaluate_then, &evaluate_otherwise)
                })
            }
            //only works on tuples
            Expr::First{ value, .. } => {
                let evaluate_value = self.compile_internal(*value, funcs_params);
                Box::new(move |ec: &mut ExecutionContext| {
                    let value = evaluate_value(ec);
                    match value {
                        Value::Tuple(ptr) => {
                            let (a, _) = &ec.tuples[ptr.0 as usize];
                            ec.heap[a.0 as usize].clone()
                        },
                        Value::SmallTuple(a, _) => Value::Int(a as i32),
                        /*Value::IntTuple(a, _) => Value::Int(a),
                        Value::BoolTuple(a, _) => Value::Bool(a),
                        Value::BoolIntTuple(a, _) => Value::Bool(a),
                        Value::IntBoolTuple(a, _) => Value::Int(a),*/
                        _ => panic!("Type error: Cannot evaluate first on this value: {value:?}"),
                    }
                })
            }
            Expr::Second{ value, .. } => {
                let evaluate_value = self.compile_internal(*value, funcs_params);
                Box::new(move |ec: &mut ExecutionContext| {
                    let value = evaluate_value(ec);
                    match value {
                        Value::Tuple(ptr) => {
                            let (_, b) = &ec.tuples[ptr.0 as usize];
                            ec.heap[b.0 as usize].clone()
                        },
                        Value::SmallTuple(_, a) => Value::Int(a as i32),
                        /*Value::IntTuple(_, a) => Value::Int(a),
                        Value::BoolTuple(_, a) => Value::Bool(a),
                        Value::BoolIntTuple(_, a) => Value::Int(a),
                        Value::IntBoolTuple(_, a) => Value::Bool(a),*/
                        _ => panic!("Type error: Cannot evaluate first on this value: {value:?}"),
                    }
                })
            }
            Expr::Bool { value, .. } => Box::new(move |_| Value::Bool(value)),
            Expr::Tuple { first, second, .. } => {
                let evaluate_first = self.compile_internal(*first, funcs_params);
                let evaluate_second = self.compile_internal(*second, funcs_params);
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

    fn compile_call_data(&mut self, func: &Box<Expr>, funcs_params: &mut HashSet<String>, args: &[Expr]) -> (Box<dyn Fn(&mut ExecutionContext<'_>) -> Value>, Vec<Box<dyn Fn(&mut ExecutionContext<'_>) -> Value>>, Option<usize>, &'static str) {
        let (callee, arguments, name, called_name) = match &**func {
            Expr::Var {name} => {
                let called_name: &'static str = name.clone().leak();
                let function_name_index = self.intern_var_name(called_name);
                let evaluate_callee = self.compile_internal(*func.clone(), funcs_params);
                let arguments = args
                    .into_iter()
                    .map(|arg| self.compile_internal(arg.clone(), funcs_params))
                    .collect::<Vec<_>>();
                (evaluate_callee, arguments, Some(function_name_index), called_name)
        
            }
            other => {
                let called_name: &'static str = "anonymous function";
                let evaluate_callee = self.compile_internal(other.clone(), funcs_params);
                let arguments = args
                    .into_iter()
                    .map(|arg| self.compile_internal(arg.clone(), funcs_params))
                    .collect::<Vec<_>>();
       
                (evaluate_callee, arguments, None, called_name)
            }
        };
        (callee, arguments, name, called_name)
    }

    fn compile_function(&mut self, funcs_params: &mut HashSet<String>,  body: Vec<Expr>, parameters: &[String], closure: &[String], tco: bool, trampoline_of: Option<usize>) -> LambdaFunction {
        //Functions are compiled and stored into a big vector of functions, each one has a unique ID.
        //value will be the call itself, we're compiling the body of the tco trampoline
        //println!("Compile function: {value:#?} location: {location:?}");
        let index_of_new_function = self.all_functions.len();
        let callable = Callable {
            body: Box::new(|_| panic!("Empty body, function not compiled")),
            parameters: &[],
            tco_optimized: false,
            trampoline_of: None,
            closure_indices: &[]
            //location
        };
        self.all_functions.push(callable);
        self.closure_stack.push(index_of_new_function);
        let mut new_params = funcs_params.clone();
        for param in parameters.iter() {
            new_params.insert(param.to_string());
        }
        //this will return the lambda for the call to iter
        //let new_function = self.compile_internal(value, &mut new_params);

        let body_lambdas: Vec<LambdaFunction> = body
            .into_iter()
            .map(|expr| self.compile_internal(expr, &mut new_params))
            .collect::<Vec<_>>();

        let new_function: LambdaFunction = Box::new(move |ec: &mut ExecutionContext| {
            let mut last = None;
            for l in body_lambdas.iter() {
                let result = &l;
                last = Some(result(ec));
            }
            last.unwrap()
        });


        let parameters = parameters
            .into_iter()
            .map(|param| param.clone())
            .collect::<Vec<_>>();
        let mut param_indices = vec![];
        for param in parameters {
            let interned = self.intern_var_name(param.leak());
            param_indices.push(interned);
        }

        let closure = closure
            .into_iter()
            .map(|param| param.clone())
            .collect::<Vec<_>>();
        let mut closure_indices = vec![];
        for closure_param in closure {
            let interned = self.intern_var_name(closure_param.leak());
            closure_indices.push(interned);
        }
        let callable = self.all_functions.get_mut(index_of_new_function).unwrap();
        *callable = Callable {
            body: new_function,
            parameters: param_indices.leak(),
            tco_optimized: tco,
            trampoline_of,
            closure_indices: closure_indices.leak()
            //location
        };
        self.closure_stack.pop();
        if trampoline_of.is_none() {
            if callable.closure_indices.len() > 0 {
                Box::new(move |ec: &mut ExecutionContext| {
                    Value::Closure(ec.eval_closure_with_env(index_of_new_function))
                })

            } else {
                Box::new(move |ec: &mut ExecutionContext| {
                    Value::Closure(ec.eval_closure_no_env(index_of_new_function))
                })
            }
        } else {
            if callable.closure_indices.len() > 0 {
                Box::new(move |ec: &mut ExecutionContext| {
                    Value::Trampoline(ec.eval_closure_with_env(index_of_new_function))
                })

            } else {
                Box::new(move |ec: &mut ExecutionContext| {
                    Value::Trampoline(ec.eval_closure_no_env(index_of_new_function))
                })
            }
        }
    }

    //only call when is_tail_recursive is true
    fn into_tco(&mut self, body: &[Expr], fname: &str) -> Vec<Expr> {

        let mut new_body = vec![];
        for value in body {
            match value {
                
                Expr::If { cond, then, otherwise } => {
                    let new_if = Expr::If { cond: cond.clone(), then: self.into_tco(then, fname).into(), otherwise: self.into_tco(&otherwise, fname).into() };
                    new_body.push(new_if);
                },
                call @ Expr::FuncCall { func, .. } => {
                    match &**func {
                        Expr::Var{ name, .. } if name == fname => { 
                            let fdecl = Expr::FuncDecl(FuncDecl {
                                params: vec![],
                                body: vec![call.clone().into()], 
                                is_tco_trampoline: true
                            });
                            new_body.push(fdecl);
                        },
                        t => {
                            new_body.push(t.clone())
                        }
                    }
                }
                t => {
                    new_body.push(t.clone());
                }
            }
        }
        new_body
    }

    pub fn is_tail_recursive(&mut self, body: &[Expr], fname: &str) -> bool {
        //println!("Checking if is tail recursive: {fname} \n${value:#?}");
        for value in body {
            match value {
                Expr::FuncCall { func, .. } => {
                    match &**func {
                        Expr::Var{ name, .. } if name == fname => return true,
                        _ => { }
                    }
                }
               
                Expr::If { then, otherwise, .. } => {
                    let is_tailrec = self.is_tail_recursive(then, fname) || self.is_tail_recursive(otherwise, fname);
                    if is_tailrec {
                        return true
                    }
                }
                _ => { }
            }
        }
        false
    }

    pub fn compile(mut self, ast: Vec<Expr>) -> CompilationResult {
        let mut params = HashSet::new();
        let lambdas: Vec<LambdaFunction> = ast
            .into_iter()
            .map(|expr| self.compile_internal(expr, &mut params))
            .collect::<Vec<_>>();
        //trampolinize the returned value
        let trampolinized: LambdaFunction = Box::new(move |ec: &mut ExecutionContext| {
            let mut last = None;
            for l in lambdas.iter() {
                let result = l(ec);
                last = Some(result);
            }
            ec.run_trampoline(last.unwrap())
        });
        CompilationResult {
            main: trampolinized,
            strings: self.var_names,
            functions: self.all_functions,
        }
    }

  
    fn compile_binexp_opt<'a>(
        &mut self,
        funcs_params: &mut HashSet<String>,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        op: BinaryOp,
    ) -> Option<LambdaFunction> { 
        macro_rules! comparison_op {
            ($lhs:expr, $rhs:expr, $ec:expr, $op:tt) => {
                match ($lhs, $rhs) {
                    (Value::Int(lhs), Value::Int(rhs)) => {
                        //println!("Sum {lhs} {rhs}");
                        Value::Bool(lhs $op rhs)
                    },
                    (Value::Str(StringPointer(lhs)), Value::Str(StringPointer(rhs))) => {
                        let lhs = &$ec.string_values[*lhs as usize];
                        let rhs = &$ec.string_values[*rhs as usize];
                        Value::Bool(lhs $op rhs)
                    }
                    (Value::Bool(lhs), Value::Bool(rhs)) => {
                        Value::Bool(lhs $op rhs)
                    }
                    _ => panic!(
                        "Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}",
                        op = stringify!($op)
                    ),
                }
            }
        }

        macro_rules! binary_and_or {
            ($lhs:expr, $rhs:expr, $ec:expr, $op:tt) => {
                match ($lhs, $rhs) {
                    (Value::Bool(lhs), Value::Bool(rhs)) => {
                        //println!("Sum {lhs} {rhs}");
                        Value::Bool(*lhs $op *rhs)
                    },
                    _ => panic!(
                        "Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}",
                        op = stringify!($op)
                    ),
                }
            };
        }

        macro_rules! int_arith_op {
            ($lhs:expr, $rhs:expr, $ec:expr, $op:tt) => {
                match ($lhs, $rhs) {
                    (Value::Int(lhs), Value::Int(rhs)) => {
                        //println!("Sum {lhs} {rhs}");
                        Value::Int(lhs $op rhs)
                    }
                    _ => panic!(
                        "Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}",
                        op = stringify!($op)
                    ),
                }
            }
        }

        macro_rules! binop_add {
            ($lhs:expr, $rhs:expr, $ec:expr, $op:tt) => {
                match ($lhs, $rhs) {
                    (Value::Int(lhs), Value::Int(rhs)) => {
                        //println!("Sum {lhs} {rhs}");
                        Value::Int(lhs + rhs)
                    }
                    (lhs, rhs @ Value::Str(..)) | (lhs @ Value::Str(..), rhs) => {
                        let str = format!("{}{}", lhs.to_string($ec), rhs.to_string($ec));
                        let current = StringPointer($ec.string_values.len() as u32);
                        $ec.string_values.push(str);
                        Value::Str(current)
                    }
                    _ => panic!(
                        "Type error: Cannot apply binary operator + on these values: {lhs:?} and {rhs:?}"
                    ),
                }
            };
        }

        macro_rules! dispatch_bin_op {
            ($mode:ident, $op:tt) => {
                match $op {
                    BinaryOp::Add => $mode!(+, binop_add),
                    BinaryOp::Sub => $mode!(-, int_arith_op),
                    BinaryOp::Mul => $mode!(*, int_arith_op),
                    BinaryOp::Div => $mode!(/, int_arith_op),
                    BinaryOp::Rem => $mode!(%, int_arith_op),
                    BinaryOp::Eq => $mode!(==, comparison_op),
                    BinaryOp::Neq => $mode!(!=, comparison_op),
                    BinaryOp::Lt => $mode!(<, comparison_op),
                    BinaryOp::Gt => $mode!(>, comparison_op),
                    BinaryOp::Lte => $mode!(<=, comparison_op),
                    BinaryOp::Gte => $mode!(>=, comparison_op),
                    BinaryOp::And => $mode!(&&, binary_and_or),
                    BinaryOp::Or => $mode!(||, binary_and_or),
                }
            };
        }

        let e = (&*lhs, &*rhs);
        let result: LambdaFunction = match e {
         
            //If we know beforehand that both sides are var, avoid doing too much indirection just to get the var values
            (Expr::Var { name: lhs_name }, Expr::Var { name: rhs_name }) => {
                
                let index_lhs = self.intern_var_name(lhs_name);
                let index_rhs = self.intern_var_name(rhs_name);

                macro_rules! variable_binop {
                    ($op:tt, $operation_name:ident) => {
                        Box::new(move |ec: &mut ExecutionContext| {
                            let lhs = ec.eval_var_ref(index_lhs);
                            let rhs = ec.eval_var_ref(index_rhs);
                           
                            $operation_name!(lhs, rhs, ec, $op)
                        })
                    }
                }
                dispatch_bin_op!(variable_binop, op)
            }
             //LHS is var, RHS is int
            (Expr::Var { name: lhs_name }, Expr::Int { value } ) => {
                
                let index_lhs = self.intern_var_name(lhs_name);
                let int_value = Value::Int(*value);
                macro_rules! variable_int_binop {
                    ($op:tt, $operation_name:ident) => {
                        Box::new(move |ec: &mut ExecutionContext| {
                            let lhs = ec.eval_var_ref(index_lhs);
                            $operation_name!(lhs, &int_value, ec, $op)
                        })
                    }
                }
                dispatch_bin_op!(variable_int_binop, op)
            }
            //LHS is var, RHS is bool
            (Expr::Var { name: lhs_name }, Expr::Bool { value } ) => {
                
                let index_lhs = self.intern_var_name(lhs_name);
                let bool_value = Value::Bool(*value);

                macro_rules! variable_int_binop {
                    ($op:tt, $operation_name:ident) => {
                        Box::new(move |ec: &mut ExecutionContext| {
                            let lhs = ec.eval_var_ref(index_lhs);
                            $operation_name!(lhs, &bool_value, ec, $op)
                        })
                    }
                }
                dispatch_bin_op!(variable_int_binop, op)
            }
             //LHS is int, RHS is var
             ( Expr::Int { value }, Expr::Var { name: lhs_name } ) => {
                
                let int_value = Value::Int(*value);
                let index_rhs = self.intern_var_name(lhs_name);

                macro_rules! variable_int_binop {
                    ($op:tt, $operation_name:ident) => {
                        Box::new(move |ec: &mut ExecutionContext| {
                            let rhs = ec.eval_var_ref(index_rhs);
                            $operation_name!(&int_value, rhs, ec, $op)
                        })
                    }
                }
                dispatch_bin_op!(variable_int_binop, op)
            }
            //LHS is bool, RHS is var
            ( Expr::Bool { value }, Expr::Var { name: lhs_name } ) => {
                
                let bool_value = Value::Bool(*value);
                let index_rhs = self.intern_var_name(lhs_name);

                macro_rules! variable_int_binop {
                    ($op:tt, $operation_name:ident) => {
                        Box::new(move |ec: &mut ExecutionContext| {
                            let rhs = ec.eval_var_ref(index_rhs);
                            $operation_name!(&bool_value, rhs, ec, $op)
                        })
                    }
                }
                dispatch_bin_op!(variable_int_binop, op)
            }
            //both sides are function calls, just trigger both
            ( Expr::FuncCall {func: func_lhs, args:args_lhs }, Expr::FuncCall { func: func_rhs, args: args_rhs } ) => {
                
                let (callee_lhs, args_lhs, name_index_lhs, called_name_lhs) = self.compile_call_data(func_lhs, funcs_params, args_lhs);
                let (callee_rhs, args_rhs, name_index_rhs, called_name_rhs) = self.compile_call_data(func_rhs, funcs_params, args_rhs);
                 
                macro_rules! call_both_sides {
                    ($op:tt, $operation_name:ident) => {
                        Box::new(move |ec: &mut ExecutionContext| {
                            let call_lhs = ec.eval_call(&callee_lhs, &args_lhs, name_index_lhs, called_name_lhs);
                            let call_lhs = ec.run_trampoline(call_lhs);
    
                            let call_rhs = ec.eval_call(&callee_rhs, &args_rhs, name_index_rhs, called_name_rhs);
                            let call_rhs = ec.run_trampoline(call_rhs);
                           
                            $operation_name!(&call_lhs, &call_rhs, ec, $op)
                        })
                    }
                }

                dispatch_bin_op!(call_both_sides, op)
            }
            //we could do constant folding here
            _ => return None
        };
        return Some(result)
    }


    fn compile_binexp(
        &mut self,
        funcs_params: &mut HashSet<String>,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        op: BinaryOp,
    ) -> LambdaFunction {

        //tries to return an optimized version that does less eval_calls
        let optimized = self.compile_binexp_opt(funcs_params, lhs.clone(), rhs.clone(), op.clone());
        if let Some(opt) = optimized {
            return opt;
        }


        let evaluate_lhs = self.compile_internal(*lhs, funcs_params);
        let evaluate_rhs = self.compile_internal(*rhs, funcs_params);

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
                            (Value::Str(StringPointer(lhs)), Value::Str(StringPointer(rhs))) => {
                                let lhs = &ec.string_values[*lhs as usize];
                                let rhs = &ec.string_values[*rhs as usize];
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

fn as_ref_slice(v: &[Expr]) -> Vec<&Expr> {
    v.iter().collect()
}
