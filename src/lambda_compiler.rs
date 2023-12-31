use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ops::{Index, IndexMut};

use crate::ast::BinaryOp;

use crate::hir::{Expr, FuncDecl};

pub type LambdaFunction = Box<dyn Fn(&mut ExecutionContext, &mut CallFrame) -> Value>;

pub type ClosureStorage = Vec<Value>;

#[derive(Debug)]
pub struct Stats {
    pub new_frames: usize,
    pub reused_frames: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Closure {
    pub callable_index: usize,
    pub closure_env_index: usize, //pub environment: BTreeMap<usize, Value>
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct HeapPointer(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct StringPointer(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct ClosurePointer(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
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
    None,
}

impl Value {
    fn to_string(self, ec: &ExecutionContext) -> String {
        match self {
            Value::Trampoline(_) => "trampoline".to_string(),
            Value::Int(i) => i.to_string(),
            Value::Str(StringPointer(s)) => ec.string_values[s as usize].to_string(),
            Value::Tuple(TuplePointer(t)) => {
                let (f, s) = &ec.tuples[t as usize];
                let a = &ec.heap[f.0 as usize];
                let b = &ec.heap[s.0 as usize];
                format!("({}, {})", a.to_string(ec), b.to_string(ec))
            }
            Value::SmallTuple(a, b) => format!("({}, {})", a, b),
            /*Value::BoolTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::IntBoolTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),
            Value::BoolIntTuple(a, b) => format!("({}, {})", a.to_string(), b.to_string()),*/
            Value::Closure(..) => "function".to_string(),
            //Value::Error(e) => format!("error: {}", e),
            // Value::Unit => "unit".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::None => "None".to_string(),
        }
    }
}

pub struct StackData {
    //@TODO replace by actual stack
    backing_store: Vec<Value>,
}

impl StackData {
    pub fn reset(&mut self, size: usize) {
        let add = (size as isize - self.backing_store.len() as isize).max(0) as usize;
        self.backing_store.reserve(add);
        //SAFETY: This is a massive hack to allow the [] operator to set in the capacity
        //available of the vec.

        unsafe { self.backing_store.set_len(self.backing_store.capacity()) }
    }

    fn new(len: usize) -> StackData {
        let mut s = StackData {
            backing_store: vec![],
        };
        s.reset(len);
        s
    }
}

impl Index<StackPosition> for StackData {
    type Output = Value;

    fn index(&self, index: StackPosition) -> &Self::Output {
        &self.backing_store[index.0]
    }
}

impl IndexMut<StackPosition> for StackData {
    fn index_mut(&mut self, index: StackPosition) -> &mut Self::Output {
        &mut self.backing_store[index.0]
    }
}

pub struct CallFrame {
    pub closure_ptr: ClosurePointer,
    pub stack_data: StackData,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct StackPosition(usize);

type ClosureBuilder = Box<dyn Fn(&mut ExecutionContext, &mut CallFrame) -> ClosureStorage>;

pub struct Callable {
    pub name: Option<Symbol>,
    pub parameters: &'static [Symbol],
    pub closure_vars: &'static [Symbol],
    pub body: LambdaFunction,
    pub trampoline_of: Option<usize>, //the callable index this callable belongs to,
    pub layout: BTreeMap<Symbol, StackPosition>, //index is the variable id, value at index is the stack index
    pub closure_builder: Option<ClosureBuilder>, //this builds a closure environment
}

//used in compile_internal to track the state of the function being compiled
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct FunctionData {
    name: Option<Symbol>,
    callable_index: usize,

    //order is important so external users should use this property
    let_bindings_so_far: Vec<Symbol>,

    //but for large programs we query in this field to quickly check if it's already added
    let_bindings_so_far_idx: BTreeSet<Symbol>,
    parameters: Vec<Symbol>,
    closure: Vec<Symbol>, //this is in the order found in the source code (order of access)
    trampoline_of: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub enum VariableType {
    RecursiveFunction { callable_index: usize },
    StackPosition(StackPosition),
    Trampoline { callable_index: usize },
    Closure,
}

impl FunctionData {
    fn add_let_binding(&mut self, symbol: Symbol) {
        if !self.let_bindings_so_far_idx.contains(&symbol) {
            self.let_bindings_so_far.push(symbol);
            self.let_bindings_so_far_idx.insert(symbol);
        }
    }

    fn get_variable_type(&self, s: &Symbol, compiler: &LambdaCompiler) -> VariableType {
        if let Some(n) = self.name && *s == n {
            return VariableType::RecursiveFunction { callable_index: self.callable_index };
        }

        //check trampoline call: if the symbol relates to the name of the function we're the trampoline of,
        //return the original function
        if let Some(t) = self.trampoline_of {
            let callable = &compiler.all_functions[t];
            if let Some(original_name) = callable.name && original_name == *s {
                return VariableType::Trampoline{ callable_index: t};
            }
        }

        for (i, lb) in self.let_bindings_so_far.iter().enumerate() {
            if lb == s {
                return VariableType::StackPosition(StackPosition(self.parameters.len() + i));
            }
        }

        for (i, lb) in self.parameters.iter().enumerate() {
            if lb == s {
                return VariableType::StackPosition(StackPosition(i));
            }
        }

        //If we have no idea then it has to come from a closure

        VariableType::Closure
    }

    fn process_closure(&mut self, func: &FunctionData, compiler: &mut LambdaCompiler) {
        //the closure of func has to be considered into our own closure space.
        //we also have to build the closure builder

        for c in func.closure.iter() {
            if self.parameters.contains(c)
                || self.let_bindings_so_far.contains(c)
                || self.closure.contains(c)
            {
                continue;
            }
            self.closure.push(*c);
        }

        //we will now build the closure builder for the func function, then store it in the closure builder
        //for that function, but the data is loaded from this function. Confused?
        if !func.closure.is_empty() {
            let mut current_function_data = self.clone();
            let mut lambdas = vec![];
            for closure_arg in func.closure.clone().iter() {
                let (compiled, new_function_data) =
                    compiler.compile_eval_var(*closure_arg, current_function_data);
                current_function_data = new_function_data;
                lambdas.push(compiled);
            }
            let leaked_lambdas: &'static [LambdaFunction] = lambdas.leak();
            let builder: ClosureBuilder =
                Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    let mut vec = ClosureStorage::new();
                    for l in leaked_lambdas {
                        vec.push(ec.run_lambda_trampoline(l, frame));
                    }
                    vec
                });

            compiler.all_functions[func.callable_index].closure_builder = Some(builder);
        }
    }
}

pub struct ExecutionContext<'a> {
    // pub call_stack: Vec<CallFrame>,
    //pub stack: DynStack<'s>,
    pub functions: &'a [Callable],
    pub reusable_frames: Vec<StackData>,
    //First vec represents the variable itself (there's only a set number of possible values),
    //second vec represents the stack frame's value of that variable
    //pub let_bindings: Vec<Vec<Value>>,
    //pub reusable_frames: Vec<StackFrame>,
    //this SmallVec in ClosureStorage is for performance, I just don't know how to reuse closure vecs...
    //And I think reusing a vec is actually faster, but for now this will have to suffice.
    pub closure_environments: Vec<ClosureStorage>,
    pub string_values: Vec<String>,
    //most things are done in the "stack" for some definition of stack which is very loose here....
    //this heap is for things like dynamic tuples that could very well be infinite.
    //This makes the Value enum smaller than storing boxes directly
    pub heap: Vec<Value>,
    pub tuples: Vec<(HeapPointer, HeapPointer)>,
    pub closures: Vec<Closure>,
}

pub struct CompilationResult {
    pub main: Callable,
    pub strings: Vec<String>,
    pub functions: Vec<Callable>,
}

#[derive(Eq, PartialEq)]
//Enum used to detect when a function is tail recursive. This is not a property of the function,
//but rather a property of a particular expression. A function is tail recursive if we recurse/iterate over the AST
//and find that all expressions are tail recursive or not recursive (base cases)
enum TailRecursivity {
    //The last expression is a call to the same function, this is an intermediate result until the whole function is evaluated
    TailRecursive,
    //No self function call found, maybe it's a base case? This is an intermediate result until the whole function is evaluated.
    NotSelfRecursive,
    //It cannot be a tail recursive function... not even necessarily recursive.
    //if the last expression is a binary operation, then it's guaranteed to not be tail recursive.
    //This is a final result. Once we detect a non tail recursive situation we can stop analyzing the function.
    NotTailRecursive,
}

//Using dynstack to store closure values, frame data, etc.
impl<'a> ExecutionContext<'a> {
    pub fn new(program: &'a CompilationResult) -> (Self, CallFrame) {
        //whenever we want an empty closure, we'll access by the function index into closures without allocating a new one
        let empty_closures = {
            let mut closures = vec![];
            for (i, _) in program.functions.iter().enumerate() {
                closures.push(Closure {
                    callable_index: i,
                    closure_env_index: usize::MAX,
                })
            }
            closures
        };

        //let let_bindings = program.strings.iter().map(|s| vec![] ).collect();
        let initial_stack = {
            let mut vec = StackData {
                backing_store: vec![],
            };
            vec.reset(program.main.layout.len());
            vec
        };
        (
            Self {
                reusable_frames: vec![],
                functions: &program.functions,
                closure_environments: vec![],
                heap: vec![],
                string_values: program.strings.clone(),
                tuples: vec![],
                closures: empty_closures,
            },
            CallFrame {
                closure_ptr: ClosurePointer(u32::MAX),
                stack_data: initial_stack,
            },
        )
    }

    #[inline(always)]
    fn eval_closure_no_env(&mut self, callable_index: usize) -> ClosurePointer {
        ClosurePointer(callable_index as u32)
    }

    #[inline(always)]
    fn eval_closure_with_env(
        &mut self,
        callable_index: usize,
        frame: &mut CallFrame,
    ) -> ClosurePointer {
        let function = self.functions.get(callable_index).unwrap();

        if let Some(builder) = &function.closure_builder {
            let current_id = self.closure_environments.len();
            let vec = builder(self, frame);
            self.closure_environments.push(vec);

            let closure = Closure {
                callable_index,
                closure_env_index: current_id,
            };
            let closure_p = self.closures.len();
            self.closures.push(closure);
            ClosurePointer(closure_p as u32)
        } else {
            ClosurePointer(callable_index as u32)
        }
    }

    #[inline(always)]
    fn eval_closure_with_env_trapolined(
        &mut self,
        callable_index: usize,
        frame: &mut CallFrame,
    ) -> ClosurePointer {
        let function = self.functions.get(callable_index).unwrap();

        if let Some(builder) = &function.closure_builder {
            let current_id = self.closure_environments.len();
            let vec = builder(self, frame);
            self.closure_environments.push(vec);

            let closure = Closure {
                callable_index,
                closure_env_index: current_id,
            };
            let closure_p = self.closures.len();
            self.closures.push(closure);
            ClosurePointer(closure_p as u32)
        } else {
            ClosurePointer(callable_index as u32)
        }
    }

    #[inline(always)]
    fn eval_int(&self, value: i32) -> Value {
        Value::Int(value)
    }

    #[inline(always)]
    fn eval_let(
        &mut self,
        evaluate_value: &LambdaFunction,
        stack_pos: StackPosition,
        frame: &mut CallFrame,
    ) -> Value {
        let bound_value = self.run_lambda_trampoline(evaluate_value, frame);
        frame.stack_data[stack_pos] = bound_value;
        Value::None
    }

    #[inline(always)]
    fn eval_if(
        &mut self,
        evaluate_condition: &LambdaFunction,
        evaluate_then: &LambdaFunction,
        evaluate_otherwise: &LambdaFunction,
        frame: &mut CallFrame,
    ) -> Value {
        let condition_result = self.run_lambda_trampoline(evaluate_condition, frame);
        match condition_result {
            Value::Bool(true) => evaluate_then(self, frame),
            Value::Bool(false) => evaluate_otherwise(self, frame),
            _ => {
                panic!("Type error: Cannot evaluate condition on this value: {condition_result:?}")
            }
        }
    }

    #[inline(always)]
    fn binop_add(
        &mut self,
        evaluate_lhs: &LambdaFunction,
        evaluate_rhs: &LambdaFunction,
        frame: &mut CallFrame,
    ) -> Value {
        let lhs = self.run_lambda_trampoline(evaluate_lhs, frame);
        let rhs = self.run_lambda_trampoline(evaluate_rhs, frame);
        match (&lhs, &rhs) {
            (Value::Int(lhs), Value::Int(rhs)) => Value::Int(lhs + rhs),
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

    #[inline(always)]
    fn fastcall(
        &mut self,
        callee_function: ClosurePointer,
        arguments: &[LambdaFunction],
        frame: &mut CallFrame,
    ) -> Value {
        let ClosurePointer(p) = callee_function;

        let Closure { callable_index, .. } = self.closures[p as usize];
        let mut new_frame =
            self.make_new_frame(callable_index, ClosurePointer(p), arguments, frame);

        let function = &self.functions[callable_index].body;
        let function_result = function(self, &mut new_frame);

        if let Value::Trampoline(ClosurePointer(p)) = &function_result {
            //if the trampoline returned is not for ourselves, then we evaluate it
            let closure = &self.closures[*p as usize];
            let Closure {
                callable_index: trampoline_callable_index,
                ..
            } = closure;
            if *trampoline_callable_index != callable_index {
                //should this run in the current or new frame?
                self.run_trampoline(function_result, &mut new_frame)
            } else {
                function_result
            }
        } else {
            self.reusable_frames.push(new_frame.stack_data);
            function_result
        }
    }

    fn eval_generic_call(
        &mut self,
        evaluate_callee: &LambdaFunction,
        arguments: &Vec<LambdaFunction>,
        called_name: &str,
        frame: &mut CallFrame,
    ) -> Value {
        let callee_function = evaluate_callee(self, frame);

        let Value::Closure(ClosurePointer(p)) = callee_function else {
            panic!("Call to non-function value {called_name}: {callee_function:?}")
        };
        let Closure { callable_index, .. } = self.closures[p as usize];
        let callable = &self.functions[callable_index];

        if callable.parameters.len() != arguments.len() {
            panic!("Wrong number of arguments for function {called_name}")
        }

        let mut new_frame =
            self.make_new_frame(callable_index, ClosurePointer(p), arguments, frame);

        let function = &self.functions[callable_index].body;

        let function_result = function(self, &mut new_frame);

        if let Value::Trampoline(ClosurePointer(p)) = &function_result {
            //if the trampoline returned is not for ourselves, then we evaluate it
            let closure = &self.closures[*p as usize];
            let Closure {
                callable_index: trampoline_callable_index,
                ..
            } = closure;
            if *trampoline_callable_index != callable_index {
                self.run_trampoline(function_result, &mut new_frame)
            } else {
                function_result
            }
        } else {
            //here the frame ends.
            //we can reuse the stack data
            self.reusable_frames.push(new_frame.stack_data);
            function_result
        }
    }

    #[inline(always)]
    fn eval_trampoline(
        &mut self,
        callable_index: usize,
        arguments: &[LambdaFunction],
        frame: &mut CallFrame,
    ) -> Value {
        self.update_let_bindings_for_frame_reuse(arguments, frame);
        let function = &self.functions[callable_index].body;
        function(self, frame)
    }

    #[inline(always)]
    fn update_let_bindings_for_frame_reuse(
        &mut self,
        arguments: &[LambdaFunction],
        frame: &mut CallFrame,
    ) {
        for (i, arg) in arguments.iter().enumerate() {
            let trampolined = self.run_lambda_trampoline(arg, frame);
            frame.stack_data[StackPosition(i)] = trampolined;
        }
    }

    #[inline(always)]
    pub fn run_trampoline(&mut self, maybe_trampoline: Value, frame: &mut CallFrame) -> Value {
        let mut current @ Value::Trampoline(..) = maybe_trampoline else {
            return maybe_trampoline;
        };

        while let Value::Trampoline(ClosurePointer(p)) = current {
            let Closure { callable_index, .. } = self.closures[p as usize];

            frame.closure_ptr = ClosurePointer(p);

            //this needs to be the TCO obj itself
            // let callee = &self.functions[callable_index as usize];
            /*
            In order for the trampoline mechanism to work, we must ensure we're not calling
            the trampoline inside the execution of the trampolined function. This would be the same as
            just calling the function as is. We must end this stack frame (this lambda function)
            by returning the closure value.

            The trampoline function belongs to a recursive function, therefore we can check if the trampoline returned
            belongs to this executing function. Counter intuitively, if that's the case, we *skip* the execution and just return it,
            so that the original caller of the recursive function (that is not itself) can perform the trampoline.
            */

            let function = &self.functions[callable_index].body;

            //We don't need to setup a new stack frame because when we call this function,
            //eval_call will run (because it's a *tail call optimization*), which will setup its own call stack.
            //however, we count how many stacks we pushed and pop them all later

            current = function(self, frame);

            //self.pop_frame_and_bindings();
        }

        current
    }

    #[inline(always)]
    pub fn run_lambda_trampoline(
        &mut self,
        lambda: &LambdaFunction,
        frame: &mut CallFrame,
    ) -> Value {
        let maybe_trampoline = lambda(self, frame);
        self.run_trampoline(maybe_trampoline, frame)
    }

    #[inline(always)]
    fn make_new_frame(
        &mut self,
        callable_index: usize,
        closure_pointer: ClosurePointer,
        arguments: &[LambdaFunction],
        current_frame: &mut CallFrame,
    ) -> CallFrame {
        let callable = &self.functions[callable_index];

        let stack_data = if let Some(mut s) = self.reusable_frames.pop() {
            s.reset(callable.layout.len());
            s
        } else {
            StackData::new(callable.layout.len())
        };

        let mut new_frame = CallFrame {
            stack_data, // { backing_store: () },
            closure_ptr: closure_pointer
        };

        for (i, arg) in arguments.iter().enumerate() {
            let trampolined = self.run_lambda_trampoline(arg, current_frame);
            new_frame.stack_data[StackPosition(i)] = trampolined;
        }

        new_frame
    }

    #[inline(always)]
    fn eval_tuple(
        &mut self,
        evaluate_first: &LambdaFunction,
        evaluate_second: &LambdaFunction,
        frame: &mut CallFrame,
    ) -> Value {
        let f = self.run_lambda_trampoline(evaluate_first, frame);
        let s = self.run_lambda_trampoline(evaluate_second, frame);
        match (f, s) {
            (Value::Int(a), Value::Int(b)) => {
                //transform into i16 if in range
                if a < i16::MAX as i32
                    && a > i16::MIN as i32
                    && b < i16::MAX as i32
                    && b > i16::MIN as i32
                {
                    Value::SmallTuple(a as i16, b as i16)
                } else {
                    let heap_a = self.heap.len();
                    self.heap.push(Value::Int(a));
                    let heap_b = self.heap.len();
                    self.heap.push(Value::Int(b));

                    let tuple_p = self.tuples.len();
                    self.tuples
                        .push((HeapPointer(heap_a as u32), HeapPointer(heap_b as u32)));

                    Value::Tuple(TuplePointer(tuple_p as u32))
                }
            } /*
            (Value::Bool(a), Value::Bool(b)) => Value::BoolTuple(a, b),
            (Value::Int(a), Value::Bool(b)) => Value::IntBoolTuple(a, b),
            (Value::Bool(a), Value::Int(b)) => Value::BoolIntTuple(a, b),*/
            (a, b) => {
                let heap_a = self.heap.len();
                self.heap.push(a);
                let heap_b = self.heap.len();
                self.heap.push(b);

                let tuple_p = self.tuples.len();
                self.tuples
                    .push((HeapPointer(heap_a as u32), HeapPointer(heap_b as u32)));

                Value::Tuple(TuplePointer(tuple_p as u32))
            }
        }
    }
}

pub struct LambdaCompiler {
    //this is added into during compilation, but should be moved into the execution context later
    pub all_functions: Vec<Callable>,
    pub var_names: Vec<String>,
    pub strict_equals: bool,
    pub closure_stack: Vec<usize>, //tracks the closure stack we're in
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Symbol(usize, &'static str);

impl Default for LambdaCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl LambdaCompiler {
    pub fn new() -> Self {
        Self {
            all_functions: vec![],
            strict_equals: true,
            var_names: vec![],
            closure_stack: vec![usize::MAX],
        }
    }

    pub fn relax_equals(&mut self) {
        self.strict_equals = false;
    }

    pub fn intern_var_name(&mut self, s: &str) -> Symbol {
        //check if it already exists
        if let Some(index) = self.var_names.iter().position(|x| x == s) {
            return Symbol(index, self.var_names[index].clone().leak());
        }

        let index = self.var_names.len();
        self.var_names.push(s.to_string());
        Symbol(index, s.to_string().leak())
    }

    fn join_lambdas(&mut self, mut lambdas: Vec<LambdaFunction>) -> LambdaFunction {
        if lambdas.len() == 1 {
            return lambdas.pop().unwrap();
        }

        let leaked: &'static [LambdaFunction] = lambdas.leak();
        if leaked.len() == 2 {
            let l0 = &leaked[0];
            let l1 = &leaked[1];
            return Box::new(move |ec, frame| {
                l0(ec, frame);
                l1(ec, frame)
            });
        }

        if leaked.len() == 3 {
            let l0 = &leaked[0];
            let l1 = &leaked[1];
            let l2 = &leaked[2];
            return Box::new(move |ec, frame| {
                l0(ec, frame);
                l1(ec, frame);
                l2(ec, frame)
            });
        }

        if leaked.len() == 4 {
            let l0 = &leaked[0];
            let l1 = &leaked[1];
            let l2 = &leaked[2];
            let l3 = &leaked[3];
            return Box::new(move |ec, frame| {
                l0(ec, frame);
                l1(ec, frame);
                l2(ec, frame);
                l3(ec, frame)
            });
        }

        if leaked.len() == 5 {
            let l0 = &leaked[0];
            let l1 = &leaked[1];
            let l2 = &leaked[2];
            let l3 = &leaked[3];
            let l4 = &leaked[4];
            return Box::new(move |ec, frame| {
                l0(ec, frame);
                l1(ec, frame);
                l2(ec, frame);
                l3(ec, frame);
                l4(ec, frame)
            });
        }

        return Box::new(move |ec, frame| {
            let mut last = None;
            for l in leaked {
                last = Some(l(ec, frame));
            }
            last.unwrap()
        });
    }

    fn compile_body(
        &mut self,
        body: &[Expr],
        function_data: FunctionData,
    ) -> (LambdaFunction, FunctionData) {
        let mut lambdas = vec![];
        let mut current_fdata = function_data;

        for ast in body {
            let (lambda, fdata) = self.compile_internal(ast.clone(), current_fdata.clone(), None);
            current_fdata = fdata;
            lambdas.push(lambda);
        }

        let joined = self.join_lambdas(lambdas);

        return (joined, current_fdata);
    }

    fn compile_internal(
        &mut self,
        ast: Expr,
        mut function_data: FunctionData,
        current_let: Option<Symbol>,
    ) -> (LambdaFunction, FunctionData) {
        match ast {
            Expr::Print { value } => {
                let (evaluate_printed_value, fdata) =
                    self.compile_internal(*value, function_data, current_let);
                //because the type of the expression is only known at runtime, we have to check it in the print function during runtime :(
                (
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        let value_to_print =
                            ec.run_lambda_trampoline(&evaluate_printed_value, frame);
                        println!("{}", value_to_print.to_string(ec));
                        value_to_print
                    }),
                    fdata,
                )
            }
            Expr::Int { value } => (Box::new(move |ec, _| ec.eval_int(value)), function_data),
            Expr::Var { name, .. } => {
                let varname: &'static str = name.leak();
                let symbol = self.intern_var_name(varname);
                self.compile_eval_var(symbol, function_data)
            }

            Expr::Let { name, value } => {
                if let Expr::FuncDecl(FuncDecl { params, body, ..  }) = &*value && name != "_" && LambdaCompiler::is_tail_recursive(body, &name) == TailRecursivity::TailRecursive {
                    let trampolined_function = LambdaCompiler::to_tco(body, &name);
                    let mut all_function_params = HashSet::new();
                    for p in params {
                        all_function_params.insert(p.to_string());
                    }
                    let var_name_leaked: &'static str = name.leak();
                    let var_index = self.intern_var_name(var_name_leaked);

                    let (evaluate_value, func) = self.compile_function( trampolined_function, params, None, function_data.clone(), Some(var_index));

                    function_data.add_let_binding(var_index);
                    function_data.process_closure(&func, self);
                    let pos = function_data.get_variable_type(&var_index, self);
                    match pos {
                        VariableType::StackPosition(pos) => {
                            (Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                   ec.eval_let(&evaluate_value, pos, frame)
                               }), function_data)
                        },
                        _ => panic!("Unexpected let binding position in closure or self ref, should be a position in the stack")
                    }
                }
                else {
                    let name_interned = self.intern_var_name(&name);
                    let (evaluate_value, mut fdata) = self.compile_internal(*value,  function_data, Some(name_interned));
                    if name == "_" {
                        (Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                            ec.run_lambda_trampoline(&evaluate_value, frame) //we're only interested in the side effect of this
                        }), fdata)
                    } else {
                        fdata.add_let_binding(name_interned);
                        let pos = fdata.get_variable_type(&name_interned, self);
                        match pos {
                            VariableType::StackPosition(pos) => {
                                (Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                       ec.eval_let(&evaluate_value, pos, frame)
                                   }), fdata)
                            }
                            _ => panic!("Unexpected let binding position in closure or self ref, should be a position in the stack")
                        }
                    }
                }
            }

            // Expr::Error(e) => panic!("{}: {}", e.message, e.full_text),
            Expr::BinOp { op, left, right } => self.compile_binexp(left, right, op, function_data),
            Expr::String { value } => {
                //let leaked: &'static str = value.leak();
                (
                    Box::new(move |ec, _| {
                        let current = ec.string_values.len();
                        ec.string_values.push(value.clone());
                        Value::Str(StringPointer(current as u32))
                    }),
                    function_data,
                )
            }
            Expr::FuncCall { func, args } => {
                if let Expr::Var { name } = *func.clone() {
                    let called_name: &'static str = name.clone().leak();
                    let function_name_interned = self.intern_var_name(called_name);
                    let var_type = function_data.get_variable_type(&function_name_interned, self);
                    if let VariableType::RecursiveFunction { callable_index } = var_type {
                        //get the current callable
                        let callable = &self.all_functions[callable_index];

                        if callable.parameters.len() != args.len() {
                            panic!("Compile time check: call to {called_name} expects {} args but got {}", callable.parameters.len(), args.len())
                        }
                        let (arguments, new_fdata) = self.compile_lambda_list(&args, function_data);
                        let args_leaked = arguments.leak();

                        return (
                            Box::new(move |ec, frame| {
                                ec.fastcall(frame.closure_ptr, args_leaked, frame)
                            }),
                            new_fdata,
                        );
                    }
                };

                self.compile_generic_fcall(func, args, function_data)
            }
            Expr::TrampolineCall { func, args } => {
                let Expr::Var { name } = *func else {
                    panic!("Compilation error: can only trampoline to a named function")
                };
                let interned = self.intern_var_name(&name);
                let ty = function_data.get_variable_type(&interned, self);

                let VariableType::Trampoline { callable_index } = ty else {
                    panic!(
                        "Compilation error: Expected var to be compiled as a trampoline var type"
                    )
                };

                let (arguments, new_fdata) = self.compile_lambda_list(&args, function_data);
                (
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        ec.eval_trampoline(callable_index, &arguments, frame)
                    }),
                    new_fdata,
                )
            }
            fdecl @ Expr::FuncDecl(..) => {
                let Expr::FuncDecl(FuncDecl {
                    params,
                    body,
                    is_tco_trampoline,
                }) = fdecl
                else {
                    unreachable!()
                };

                let trampoline = if is_tco_trampoline {
                    Some(self.closure_stack.last().copied().unwrap())
                } else {
                    None
                };
                let (lambda_f, fdata) = self.compile_function(
                    body,
                    &params,
                    trampoline,
                    function_data.clone(),
                    current_let,
                );
                function_data.process_closure(&fdata, self);
                (lambda_f, function_data)
            }
            Expr::If {
                cond,
                then,
                otherwise,
                ..
            } => {
                let (evaluate_condition, condition_fdata) =
                    self.compile_internal(*cond.clone(), function_data, current_let);
                let (evaluate_then, then_fdata) = self.compile_body(&then.clone(), condition_fdata);
                let (evaluate_otherwise, otherwise_fdata) =
                    self.compile_body(&otherwise.clone(), then_fdata);
                (
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        ec.eval_if(
                            &evaluate_condition,
                            &evaluate_then,
                            &evaluate_otherwise,
                            frame,
                        )
                    }),
                    otherwise_fdata,
                )
            }
            //only works on tuples
            Expr::First { value, .. } => {
                let (evaluate_value, first_fdata) =
                    self.compile_internal(*value, function_data, current_let);
                (
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        let value = ec.run_lambda_trampoline(&evaluate_value, frame);
                        match value {
                            Value::Tuple(ptr) => {
                                let (a, _) = &ec.tuples[ptr.0 as usize];
                                ec.heap[a.0 as usize]
                            }
                            Value::SmallTuple(a, _) => Value::Int(a as i32),
                            /*Value::IntTuple(a, _) => Value::Int(a),
                            Value::BoolTuple(a, _) => Value::Bool(a),
                            Value::BoolIntTuple(a, _) => Value::Bool(a),
                            Value::IntBoolTuple(a, _) => Value::Int(a),*/
                            _ => {
                                panic!("Type error: Cannot evaluate first on this value: {value:?}")
                            }
                        }
                    }),
                    first_fdata,
                )
            }
            Expr::Second { value, .. } => {
                let (evaluate_value, second_fdata) =
                    self.compile_internal(*value, function_data, current_let);
                (
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        let value = ec.run_lambda_trampoline(&evaluate_value, frame);
                        match value {
                            Value::Tuple(ptr) => {
                                let (_, b) = &ec.tuples[ptr.0 as usize];
                                ec.heap[b.0 as usize]
                            }
                            Value::SmallTuple(_, a) => Value::Int(a as i32),
                            /*Value::IntTuple(_, a) => Value::Int(a),
                            Value::BoolTuple(_, a) => Value::Bool(a),
                            Value::BoolIntTuple(_, a) => Value::Int(a),
                            Value::IntBoolTuple(_, a) => Value::Bool(a),*/
                            _ => panic!(
                                "Type error: Cannot evaluate second on this value: {}",
                                value.to_string(ec)
                            ),
                        }
                    }),
                    second_fdata,
                )
            }
            Expr::Bool { value, .. } => (Box::new(move |_, _| Value::Bool(value)), function_data),
            Expr::Tuple { first, second, .. } => {
                let (evaluate_first, first_fdata) =
                    self.compile_internal(*first, function_data, None);
                let (evaluate_second, second_fdata) =
                    self.compile_internal(*second, first_fdata, None);
                (
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        ec.eval_tuple(&evaluate_first, &evaluate_second, frame)
                    }),
                    second_fdata,
                )
            }
        }
    }

    fn compile_generic_fcall(
        &mut self,
        func: Box<Expr>,
        args: Vec<Expr>,
        function_data: FunctionData,
    ) -> (LambdaFunction, FunctionData) {
        let (callee, arguments, _, called_name, call_fdata) =
            self.compile_call_data(&func, &args, function_data);
        (
            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                ec.eval_generic_call(&callee, &arguments, called_name, frame)
            }),
            call_fdata,
        )
    }

    pub fn compile_eval_var(
        &mut self,
        symbol: Symbol,
        mut function_data: FunctionData,
    ) -> (LambdaFunction, FunctionData) {
        let var_position = function_data.get_variable_type(&symbol, self);

        let lambda: LambdaFunction = match var_position {
            VariableType::Trampoline { .. } => {
                panic!("Trampoline calls should be resolved at compile time on Expr::TrampolineCall compilation! This is a compiler bug");
            }
            VariableType::RecursiveFunction { .. } => {
                Box::new(move |_, frame: &mut CallFrame| {
                    let var = frame.closure_ptr;
                    Value::Closure(var)
                })
            }
            VariableType::StackPosition(pos) => {
                Box::new(move |_, frame: &mut CallFrame| {
                    let var = &frame.stack_data;
                    var[pos]
                })
            }
            VariableType::Closure => {
                //need to find the index of this name in the closure
                let mut found = function_data.closure.iter().position(|var| var == &symbol);
                if found.is_none() {
                    found = Some(function_data.closure.len());
                    function_data.closure.push(symbol);
                }
                let closure_var_idx = found.unwrap();
                Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    let closure_env = frame.closure_ptr;
                    let Closure {
                        closure_env_index, ..
                    } = ec.closures[closure_env.0 as usize];
                    // Can we do this in-frame? No because we can call a function that deep into its stack frame returns a closure all the way to us, at that stage
                    // we have no context where the values might have come from, they might be stack data already popped, so it will construct a closure instance and we get it here

                    let env = &ec.closure_environments[closure_env_index];
                    env[closure_var_idx]
                })
            }
        };

        (lambda, function_data)
    }

    fn compile_call_data(
        &mut self,
        func: &Expr,
        args: &[Expr],
        function_data: FunctionData,
    ) -> (
        LambdaFunction,
        Vec<LambdaFunction>,
        Option<Symbol>,
        &'static str,
        FunctionData,
    ) {
        match func {
            Expr::Var { name } => {
                //the loop call here will be inside a trampoline
                let called_name: &'static str = name.clone().leak();
                let function_name_index = self.intern_var_name(called_name);
                let (evaluate_callee, fdata) =
                    self.compile_internal(func.clone(), function_data, None);

                let (arguments, new_fdata) = self.compile_lambda_list(args, fdata);
                (
                    evaluate_callee,
                    arguments,
                    Some(function_name_index),
                    called_name,
                    new_fdata,
                )
            }
            other => {
                let called_name: &'static str = "anonymous function";
                let (evaluate_callee, mut fdata) =
                    self.compile_internal(other.clone(), function_data, None);

                let mut arguments: Vec<LambdaFunction> = vec![];
                for arg in args {
                    let (lambda, new_fdata) =
                        self.compile_internal(arg.clone(), fdata.clone(), None);
                    fdata = new_fdata;
                    arguments.push(lambda);
                }
                (evaluate_callee, arguments, None, called_name, fdata)
            }
        }
    }

    fn compile_lambda_list(
        &mut self,
        args: &[Expr],
        fdata: FunctionData,
    ) -> (Vec<LambdaFunction>, FunctionData) {
        let mut arguments: Vec<LambdaFunction> = vec![];
        let mut fdata = fdata;
        for arg in args {
            let (lambda, new_fdata) = self.compile_internal(arg.clone(), fdata.clone(), None);
            fdata = new_fdata;
            arguments.push(lambda);
        }
        (arguments, fdata)
    }

    fn compile_function(
        &mut self,
        body: Vec<Expr>,
        parameters: &[String],
        trampoline_of: Option<usize>,
        parent_function_data: FunctionData,
        current_let: Option<Symbol>,
    ) -> (LambdaFunction, FunctionData) {
        //Functions are compiled and stored into a big vector of functions, each one has a unique ID.

        //This is necessary so that functions inside functions can access the parent function
        //via closures
        let mut let_bindings_so_far = {
            if let Some(cur) = &current_let {
                vec![*cur]
            } else {
                vec![]
            }
        };

        let new_callable_index = self.all_functions.len();

        let mut function_data = if let Some(t) = trampoline_of {
            //trampoline optimization: we're going to reuse the frame of the function we're trampolining
            //therefore the let bindings, paramters and closure should be the same

            //also we need the let bindings of the parent function too
            //this will include the parent function itself
            let_bindings_so_far.extend(parent_function_data.let_bindings_so_far.clone());

            FunctionData {
                name: None,
                let_bindings_so_far_idx: let_bindings_so_far.clone().into_iter().collect(),
                let_bindings_so_far,
                parameters: parent_function_data.parameters,
                closure: parent_function_data.closure,
                trampoline_of: Some(t),
                callable_index: new_callable_index,
            }
        } else {
            FunctionData {
                let_bindings_so_far_idx: let_bindings_so_far.clone().into_iter().collect(),
                let_bindings_so_far,
                name: current_let,
                parameters: parameters.iter().map(|x| self.intern_var_name(x)).collect(),
                //the closure will be figured out as we compile the function
                closure: vec![],
                trampoline_of,
                callable_index: new_callable_index,
            }
        };

        let callable = Callable {
            name: current_let,
            body: Box::new(|_, _| panic!("Empty body, function not compiled yet")),
            //because recursive call checks will check for correct arg number so that we don't do it over and over and over and over and over....
            parameters: function_data.parameters.clone().leak(),
            //layout is TBD after the compilation finishes
            layout: BTreeMap::new(),
            trampoline_of: function_data.trampoline_of,
            //we don't know yet
            closure_vars: &[],
            //not built yet
            closure_builder: None,
        };
        self.all_functions.push(callable);
        self.closure_stack.push(new_callable_index);

        //this will return the lambda for the call to iter
        //let new_function = self.compile_internal(value, &mut new_params);

        let mut body_lambdas = vec![];
        for expr in body.iter() {
            let (lambda, new_fdata) = self.compile_internal(expr.clone(), function_data, None);
            function_data = new_fdata;
            body_lambdas.push(lambda);
        }
        let new_function = self.join_lambdas(body_lambdas);

        let parameters = parameters.to_vec();
        let mut param_indices = vec![];
        for param in parameters {
            let interned = self.intern_var_name(param.leak());
            param_indices.push(interned);
        }

        let closure = function_data.closure.to_vec();

        let callable = self.all_functions.get_mut(new_callable_index).unwrap();

        let mut function_layout = BTreeMap::new();

        let params_len = function_data.parameters.len();
        //first the parameters
        for (i, p) in function_data.parameters.iter().enumerate() {
            function_layout.insert(*p, StackPosition(i));
        }
        //then let bindings
        for (i, p) in function_data.let_bindings_so_far.iter().enumerate() {
            function_layout.insert(*p, StackPosition(i + params_len));
        }

        *callable = Callable {
            name: current_let,
            body: new_function,
            parameters: param_indices.leak(),
            trampoline_of,
            closure_vars: closure.clone().leak(),
            layout: function_layout,
            closure_builder: None, //location
        };
        self.closure_stack.pop();
        (
            if trampoline_of.is_none() {
                if !callable.closure_vars.is_empty() {
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        Value::Closure(ec.eval_closure_with_env(new_callable_index, frame))
                    })
                } else {
                    Box::new(move |ec: &mut ExecutionContext, _| {
                        Value::Closure(ec.eval_closure_no_env(new_callable_index))
                    })
                }
            } else if !callable.closure_vars.is_empty() {
                Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    Value::Trampoline(
                        ec.eval_closure_with_env_trapolined(new_callable_index, frame),
                    )
                })
            } else {
                Box::new(move |ec: &mut ExecutionContext, _| {
                    Value::Trampoline(ec.eval_closure_no_env(new_callable_index))
                })
            },
            function_data,
        )
    }

    //only call when is_tail_recursive is true
    fn to_tco(body: &[Expr], fname: &str) -> Vec<Expr> {
        let mut new_body = vec![];
        for value in body {
            match value {
                Expr::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    let new_if = Expr::If {
                        cond: cond.clone(),
                        then: LambdaCompiler::to_tco(then, fname),
                        otherwise: LambdaCompiler::to_tco(otherwise, fname),
                    };
                    new_body.push(new_if);
                }
                Expr::FuncCall { func, args } => {
                    let trampoline_call = Expr::TrampolineCall {
                        func: func.clone(),
                        args: args.clone(),
                    };

                    match &**func {
                        Expr::Var { name, .. } if name == fname => {
                            let fdecl = Expr::FuncDecl(FuncDecl {
                                params: vec![],
                                body: vec![trampoline_call],
                                is_tco_trampoline: true,
                            });
                            new_body.push(fdecl);
                        }
                        t => new_body.push(t.clone()),
                    }
                }
                t => {
                    new_body.push(t.clone());
                }
            }
        }
        new_body
    }

    fn is_tail_recursive(body: &[Expr], fname: &str) -> TailRecursivity {
        for value in body.iter() {
            match value {
                Expr::Let { value, .. } => {
                    if let TailRecursivity::TailRecursive =
                        LambdaCompiler::is_tail_recursive(&[*value.clone()], fname)
                    {
                        return TailRecursivity::NotTailRecursive;
                    }
                }
                Expr::FuncCall { func, .. } => {
                    match &**func {
                        //if we find a function call to the same function, and it's the last expression (it is because it's either the only function or the follow up to the last let expr), otherwise it's not
                        //because after the function call, more expressions need to be evaluated. we can do an early return.
                        Expr::Var { name, .. } if name == fname => {
                            return TailRecursivity::TailRecursive
                        }
                        //it's a function call to some other function, not self recursive.
                        Expr::Var { .. } => return TailRecursivity::NotSelfRecursive,
                        _ => {
                            //if it's a call to some other kind of function, then we could do more analysis...
                            //but for now we just assume it's not tail recursive
                            return TailRecursivity::NotTailRecursive;
                        }
                    }
                }
                Expr::If {
                    then,
                    otherwise,
                    cond,
                } => {
                    //if the condition itself is tail recursive, it means the function is not because more exprs will be evaluated depending on the branch.
                    if let TailRecursivity::TailRecursive =
                        LambdaCompiler::is_tail_recursive(&[*cond.clone()], fname)
                    {
                        return TailRecursivity::NotTailRecursive;
                    }

                    let then_branch = LambdaCompiler::is_tail_recursive(then, fname);
                    let otherwise_branch = LambdaCompiler::is_tail_recursive(otherwise, fname);

                    match (then_branch, otherwise_branch) {
                        //if either branch is not tail recursive, then the whole thing is not tail recursive
                        (TailRecursivity::NotTailRecursive, _)
                        | (_, TailRecursivity::NotTailRecursive) => {
                            return TailRecursivity::NotTailRecursive
                        }
                        (TailRecursivity::NotSelfRecursive, TailRecursivity::NotSelfRecursive) => {
                            return TailRecursivity::NotSelfRecursive
                        }
                        (TailRecursivity::TailRecursive, TailRecursivity::TailRecursive) => {
                            return TailRecursivity::TailRecursive
                        }
                        (TailRecursivity::TailRecursive, TailRecursivity::NotSelfRecursive) => {
                            return TailRecursivity::TailRecursive
                        }
                        (TailRecursivity::NotSelfRecursive, TailRecursivity::TailRecursive) => {
                            return TailRecursivity::TailRecursive
                        }
                    }
                }
                Expr::Tuple { first, second } => {
                    let first_branch = LambdaCompiler::is_tail_recursive(&[*first.clone()], fname);
                    let second_branch =
                        LambdaCompiler::is_tail_recursive(&[*second.clone()], fname);

                    match (first_branch, second_branch) {
                        //if either branch is not tail recursive, then the whole thing is not tail recursive
                        (TailRecursivity::NotSelfRecursive, TailRecursivity::NotSelfRecursive) => {
                            return TailRecursivity::NotSelfRecursive
                        }
                        _ => return TailRecursivity::NotTailRecursive,
                    }
                }

                Expr::Var { .. } => return TailRecursivity::NotSelfRecursive,
                Expr::Int { .. } => return TailRecursivity::NotSelfRecursive,
                Expr::Bool { .. } => return TailRecursivity::NotSelfRecursive,
                Expr::String { .. } => return TailRecursivity::NotSelfRecursive,

                Expr::First { value } => {
                    match LambdaCompiler::is_tail_recursive(&[*value.clone()], fname) {
                        TailRecursivity::TailRecursive | TailRecursivity::NotTailRecursive => {
                            return TailRecursivity::NotTailRecursive
                        }
                        TailRecursivity::NotSelfRecursive => {
                            return TailRecursivity::NotSelfRecursive
                        }
                    }
                }
                Expr::Second { value } => {
                    match LambdaCompiler::is_tail_recursive(&[*value.clone()], fname) {
                        TailRecursivity::TailRecursive | TailRecursivity::NotTailRecursive => {
                            return TailRecursivity::NotTailRecursive
                        }
                        TailRecursivity::NotSelfRecursive => {
                            return TailRecursivity::NotSelfRecursive
                        }
                    }
                }
                Expr::Print { value } => {
                    match LambdaCompiler::is_tail_recursive(&[*value.clone()], fname) {
                        TailRecursivity::TailRecursive | TailRecursivity::NotTailRecursive => {
                            return TailRecursivity::NotTailRecursive
                        }
                        TailRecursivity::NotSelfRecursive => {
                            return TailRecursivity::NotSelfRecursive
                        }
                    }
                }
                Expr::FuncDecl(_) => {
                    //this is very weird, a recursive function that returns another function...
                    //I don't know the interpreter can deal with that beyond trampolines, just return not recursive...
                    return TailRecursivity::NotTailRecursive;
                }
                Expr::BinOp { left, right, .. } => {
                    let lhs_branch = LambdaCompiler::is_tail_recursive(&[*left.clone()], fname);
                    let rhs_branch = LambdaCompiler::is_tail_recursive(&[*right.clone()], fname);

                    match (lhs_branch, rhs_branch) {
                        //if either branch is not tail recursive, then the whole thing is not tail recursive
                        (TailRecursivity::NotSelfRecursive, TailRecursivity::NotSelfRecursive) => {
                            return TailRecursivity::NotSelfRecursive
                        }
                        _ => return TailRecursivity::NotTailRecursive,
                    }
                }
                Expr::TrampolineCall { .. } => {
                    panic!("Should not find a trampoline call at this stage")
                }
            }
        }

        //We don't know and maybe don't care whether this is recursive or not recursive, but we consider it won't be tail recursive
        TailRecursivity::NotTailRecursive
    }

    pub fn compile(mut self, ast: Vec<Expr>) -> CompilationResult {
        let mut lambdas: Vec<LambdaFunction> = vec![];
        let mut function_data = FunctionData {
            name: Some(self.intern_var_name("___root_fn___")),
            let_bindings_so_far_idx: BTreeSet::new(),
            let_bindings_so_far: vec![],
            parameters: vec![],
            closure: vec![],
            callable_index: 0,
            trampoline_of: None,
        };
        for expr in ast.into_iter() {
            let (lambda, fdata) = self.compile_internal(expr, function_data.clone(), None);
            function_data = fdata;
            lambdas.push(lambda);
        }

        let function = self.join_lambdas(lambdas);
        //trampolinize the returned value
        let trampolinized: LambdaFunction =
            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                let result = function(ec, frame);
                ec.run_trampoline(result, frame)
            });

        let mut function_layout = BTreeMap::new();

        let params_len = function_data.parameters.len();
        //first the parameters
        for (i, p) in function_data.parameters.iter().enumerate() {
            function_layout.insert(*p, StackPosition(i));
        }
        //then let bindings
        for (i, p) in function_data.let_bindings_so_far.iter().enumerate() {
            function_layout.insert(*p, StackPosition(i + params_len));
        }

        let callable = Callable {
            name: function_data.name,
            body: trampolinized,
            closure_builder: None,
            closure_vars: &[],
            layout: function_layout,
            parameters: function_data.parameters.clone().leak(),
            trampoline_of: None,
        };

        CompilationResult {
            main: callable,
            strings: self.var_names,
            functions: self.all_functions,
        }
    }

    fn compile_binexp_opt(
        &mut self,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        op: BinaryOp,
        function_data: FunctionData,
    ) -> Option<(LambdaFunction, FunctionData)> {
        macro_rules! comparison_op {
            ($lhs:expr, $rhs:expr, $ec:expr, $op:tt) => {
                match ($lhs, $rhs) {
                    (Value::Int(lhs), Value::Int(rhs)) => {
                        Value::Bool(lhs $op rhs)
                    },
                    (Value::Str(StringPointer(lhs)), Value::Str(StringPointer(rhs))) => {
                        let lhs = &$ec.string_values[lhs as usize];
                        let rhs = &$ec.string_values[rhs as usize];
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
                        Value::Bool(lhs $op rhs)
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
        let result: (LambdaFunction, FunctionData) = match e {
            //LHS is anything, RHS is int
            (lhs, Expr::Int { value }) => {
                let (index_lhs, lhs_fdata) =
                    self.compile_internal(lhs.clone(), function_data, None);
                let int_value = Value::Int(*value);

                macro_rules! variable_int_binop {
                    ($op:tt, $operation_name:ident) => {
                        (
                            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                let lhs = ec.run_lambda_trampoline(&index_lhs, frame);
                                $operation_name!(lhs, int_value, ec, $op)
                            }),
                            lhs_fdata,
                        )
                    };
                }
                dispatch_bin_op!(variable_int_binop, op)
            }
            //LHS could be anything, RHS is bool
            (lhs, Expr::Bool { value }) => {
                let (index_lhs, lhs_fdata) =
                    self.compile_internal(lhs.clone(), function_data, None);
                let bool_value = Value::Bool(*value);

                macro_rules! variable_int_binop {
                    ($op:tt, $operation_name:ident) => {
                        (
                            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                let lhs = ec.run_lambda_trampoline(&index_lhs, frame);
                                $operation_name!(lhs, bool_value, ec, $op)
                            }),
                            lhs_fdata,
                        )
                    };
                }
                dispatch_bin_op!(variable_int_binop, op)
            }
            //LHS is int, RHS could be anything
            (Expr::Int { value }, rhs) => {
                let (index_rhs, rhs_fdata) =
                    self.compile_internal(rhs.clone(), function_data, None);
                let int_value = Value::Int(*value);

                macro_rules! variable_int_binop {
                    ($op:tt, $operation_name:ident) => {
                        (
                            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                let rhs = ec.run_lambda_trampoline(&index_rhs, frame);
                                $operation_name!(int_value, rhs, ec, $op)
                            }),
                            rhs_fdata,
                        )
                    };
                }
                dispatch_bin_op!(variable_int_binop, op)
            }
            //LHS is bool, RHS could be anything
            (Expr::Bool { value }, rhs) => {
                let (index_rhs, rhs_fdata) =
                    self.compile_internal(rhs.clone(), function_data, None);
                let bool_value = Value::Bool(*value);

                macro_rules! variable_int_binop {
                    ($op:tt, $operation_name:ident) => {
                        (
                            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                let rhs = ec.run_lambda_trampoline(&index_rhs, frame);
                                $operation_name!(bool_value, rhs, ec, $op)
                            }),
                            rhs_fdata,
                        )
                    };
                }
                dispatch_bin_op!(variable_int_binop, op)
            }
            //both sides are function calls, just trigger both
            //@TODO mauybe this opt is unecessary/incompatible with the fastcall stuff
            /*( Expr::FuncCall {func: func_lhs, args:args_lhs }, Expr::FuncCall { func: func_rhs, args: args_rhs } ) => {

                let (callee_lhs, args_lhs, _, called_name_lhs, call_lhs) = self.compile_call_data(func_lhs,  args_lhs, function_data);
                let (callee_rhs, args_rhs, _, called_name_rhs, call_rhs) = self.compile_call_data(func_rhs,  args_rhs, call_lhs);

                macro_rules! call_both_sides {
                    ($op:tt, $operation_name:ident) => {
                        (Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                            let call_lhs = ec.eval_generic_call(&callee_lhs, &args_lhs, called_name_lhs);
                            let call_lhs = ec.run_trampoline(call_lhs);

                            let call_rhs = ec.eval_generic_call(&callee_rhs, &args_rhs, called_name_rhs);
                            let call_rhs = ec.run_trampoline(call_rhs);

                            $operation_name!(call_lhs, call_rhs, ec, $op)
                        }), call_rhs)
                    }
                }

                dispatch_bin_op!(call_both_sides, op)
            }*/
            //we could do constant folding here
            _ => return None,
        };
        Some(result)
    }

    fn compile_binexp(
        &mut self,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        op: BinaryOp,
        function_data: FunctionData,
    ) -> (LambdaFunction, FunctionData) {
        //tries to return an optimized version that does less eval_calls
        let optimized =
            self.compile_binexp_opt(lhs.clone(), rhs.clone(), op.clone(), function_data.clone());
        if let Some(opt) = optimized {
            return opt;
        }

        let (evaluate_lhs, fdata_lhs) = self.compile_internal(*lhs, function_data, None);
        let (evaluate_rhs, fdata_rhs) = self.compile_internal(*rhs, fdata_lhs, None);

        macro_rules! int_binary_numeric_op {
            ($op:tt) => {
                (Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    let lhs = ec.run_lambda_trampoline(&evaluate_lhs, frame);
                    let rhs = ec.run_lambda_trampoline(&evaluate_rhs, frame);
                    match (&lhs, &rhs) {
                        (Value::Int(lhs), Value::Int(rhs)) => {
                            Value::Int(lhs $op rhs)
                        }
                        _ => panic!("Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}", op = stringify!($op)),
                    }
                }), fdata_rhs)
            };
        }

        macro_rules! binary_comparison_op {
            ($op:tt) => {

                if (self.strict_equals) {
                    (Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        let lhs = ec.run_lambda_trampoline(&evaluate_lhs, frame);
                        let rhs = ec.run_lambda_trampoline(&evaluate_rhs, frame);

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
                    }), fdata_rhs)
                } else {
                    (Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        let lhs = ec.run_lambda_trampoline(&evaluate_lhs, frame);
                        let rhs = ec.run_lambda_trampoline(&evaluate_rhs, frame);

                        return Value::Bool(lhs $op rhs);
                    }), fdata_rhs)
                }
            };
        }

        match op {
            BinaryOp::Add => (
                Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    ec.binop_add(&evaluate_lhs, &evaluate_rhs, frame)
                }),
                fdata_rhs,
            ),
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
            BinaryOp::And => (
                Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    let lhs = ec.run_lambda_trampoline(&evaluate_lhs, frame);
                    let rhs = ec.run_lambda_trampoline(&evaluate_rhs, frame);
                    match (&lhs, &rhs) {
                    (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(*lhs && *rhs),
                    _ => panic!("Type error: Cannot apply binary operator && (and) on these values: {lhs:?} and {rhs:?}"),
                }
                }),
                fdata_rhs,
            ),
            BinaryOp::Or => (
                Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    let lhs = ec.run_lambda_trampoline(&evaluate_lhs, frame);
                    let rhs = ec.run_lambda_trampoline(&evaluate_rhs, frame);
                    match (&lhs, &rhs) {
                    (Value::Bool(lhs), Value::Bool(rhs)) => Value::Bool(*lhs || *rhs),
                    _ => panic!("Type error: Cannot apply binary operator || (or) on these values: {lhs:?} and {rhs:?}"),
                }
                }),
                fdata_rhs,
            ),
        }
    }
}
