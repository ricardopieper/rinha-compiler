use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{StdoutLock, Write};
use std::ops::{Index, IndexMut};

use smallvec::SmallVec;

use crate::ast::BinaryOp;

use crate::hir::{Expr, FuncDecl};

pub type LambdaFunction = Box<dyn Fn(&mut ExecutionContext, &mut CallFrame) -> Value>;

pub type ClosureStorage = SmallVec<[Value; 4]>;

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
#[repr(u8)]
pub enum ValueType {
    Integer = 0b0000,
    Boolean = 0b0001,
    String = 0b0010,
    TuplePtr = 0b0011,
    SmallTuple = 0b0100,
    Closure = 0b0101,
    Trampoline = 0b0110,
}


impl ValueType {
    pub const fn tag(self) -> u64 {
        (self as u8 as u64) << 60
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Value(u64);

impl Value {

    #[inline(always)]
    pub const fn get_tag(self) -> u64 {
        self.0 & 0xf000000000000000u64
    }

    #[inline(always)]
    pub const fn is_int(self) -> bool {
        self.get_tag() == ValueType::Integer.tag()
    }

    #[inline(always)]
    pub const fn is_bool(self) -> bool {
        self.get_tag() == ValueType::Boolean.tag()
    }

    #[inline(always)]
    pub const fn is_closure(self) -> bool {
        self.get_tag() == ValueType::Closure.tag()
    }

    #[inline(always)]
    pub const fn is_trampoline(self) -> bool {
        self.get_tag() == ValueType::Trampoline.tag()
    }

    #[inline(always)]
    pub const fn is_small_tuple(self) -> bool {
        self.get_tag() == ValueType::SmallTuple.tag()
    }

    #[inline(always)]
    pub const fn is_tuple(self) -> bool {
        self.get_tag() == ValueType::TuplePtr.tag()
    }

    #[inline(always)]
    pub const fn is_str(self) -> bool {
        self.get_tag() == ValueType::String.tag()
    }


    pub fn new_int(i: i32) -> Self {
        let as_u32: u32 = unsafe { std::mem::transmute(i) };
        Value(as_u32 as u64)
    }

    pub fn read_int(self) -> i32 {
        let lower_half = self.0 as u32;
        unsafe { std::mem::transmute(lower_half) }
    }

    pub fn new_bool(i: bool) -> Self {
        let i: u64 = i.into();
        Value(i | (ValueType::Boolean.tag()))
    }

    pub fn read_bool(self) -> bool {
        self.0 & 1 > 0
    }

    pub fn fits_in_30_bits(i: i32) -> bool {
        i <= 0x1fffffff && i >= -0x20000000
    }

    //Assumes fits_in_30_bits
    pub fn new_small_tuple(lhs: i32, rhs: i32) -> Self {
        fn pack_30_bits(i: i32) -> u32 {
            let as_u32: u32 = unsafe { std::mem::transmute(i) };
            as_u32 & 0x3fffffffu32
        }

        let packed_lhs = pack_30_bits(lhs) as u64;
        let packed_rhs = pack_30_bits(rhs) as u64;

        let tag = ValueType::SmallTuple.tag();

        let result = (packed_lhs << 30) | packed_rhs as u64 | tag;

        Self(result)
    }

    pub fn read_small_tuple(self) -> (i32, i32) {
        pub fn unpack_30_bits(i: u32) -> i32 {
            //this i number has no leftmost bit as 1, it's not a negative number here because it's a 30-bit negative number transmuted into u32
            //therefore when I read this 0b00xxxxxx number, it's guaranteed to retain its sign
            let shift_until_sign = (i as i32) << 2;

            //shift_until_sign means the leftmost, 30th bit is placed in the leftmost position of the i32
            //so the value is like 0b100010010....00, the 2 ending 0s were added by the shift

            let sign_extension = shift_until_sign >> 2;
            //now the 1 in the leftmost position gets copied 2 times, like 0b11100010010...
            //which is exactly what we want. If it's a positive number, 0 will be copied... which also works
            sign_extension
        }

        let lhs = ((self.0 & 0x0fffffffc0000000) >> 30) as u32;
        let rhs = (self.0 & 0x000000003fffffff) as u32;

        (unpack_30_bits(lhs), unpack_30_bits(rhs))
    }

    pub fn make_string_ptr(str: &'static String) -> Self {
        let as_u64: u64 = unsafe { std::mem::transmute(str) };
        Value(as_u64 | ValueType::String.tag())
    }

    //assumes checked tag is string, otherwise this will explode
    //Maybe this should be unsafe
    pub fn get_string(self) -> &'static String {
        let cleared_tag = 0x00ffffffffffffff & self.0;
        unsafe { std::mem::transmute(cleared_tag) }
    }

    pub fn make_tuple_ptr(str: &'static (Value, Value)) -> Self {
        let as_u64: u64 = unsafe { std::mem::transmute(str) };
        Value(as_u64 | ValueType::TuplePtr.tag())
    }

    pub fn get_tuple(self) -> &'static (Value, Value) {
        let cleared_tag = 0x00ffffffffffffff & self.0;
        unsafe { std::mem::transmute(cleared_tag) }
    }
    pub fn make_closure(
        //why? because I need a ptr, not a wide 16byte ptr
        cls: &'static Closure,
    ) -> Self {
        let as_u64: u64 = unsafe { std::mem::transmute(cls) };
        Value(as_u64 | ValueType::Closure.tag())
    }

    pub fn make_trampoline(
        //why? because I need a ptr, not a wide 16byte ptr
        cls: &'static Closure,
    ) -> Self {
        let as_u64: u64 = unsafe { std::mem::transmute(cls) };
        Value(as_u64 | ValueType::Trampoline.tag())
    }

    pub fn get_closure(self) -> &'static Closure {
        let cleared_tag = 0x00ffffffffffffff & self.0;
        unsafe { std::mem::transmute(cleared_tag) }
    }
    /*
    pub fn make_closure(
        //why? because I need a ptr, not a wide 16byte ptr
        str: &'static &'static dyn Fn(&mut ExecutionContext, &mut CallFrame) -> Value,
    ) -> Self {
        let as_u64: u64 = unsafe { std::mem::transmute(str) };
        Value(as_u64 | ValueType::Closure.tag())
    }

    pub fn get_closure(
        self,
    ) -> &'static &'static dyn Fn(&mut ExecutionContext, &mut CallFrame) -> Value {
        let cleared_tag = 0x00ffffffffffffff & self.0;
        unsafe { std::mem::transmute(cleared_tag) }
    }
     */
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        if self.is_int() {
            let i = self.read_int();
            write!(f, "{}", i)
        }
        else if self.is_bool() {
            let b = self.read_bool();
            write!(f, "{}", b)
        }
        else if self.is_str() {
            let s = self.get_string();
            write!(f, "{}", s)
        }
        else if self.is_closure() {
            write!(f, "<function>")
        }
        else if self.is_trampoline() {
            write!(f, "<trampoline>")
        }
        else if self.is_small_tuple() {
            let (lhs, rhs) = self.read_small_tuple();
            write!(f, "(")?;
            write!(f, "{}", lhs)?;
            write!(f, ", ")?;
            write!(f, "{}", rhs)?;
            write!(f, ")")
        }
        else if self.is_tuple() {
            let (lhs, rhs) = self.get_tuple();
            write!(f, "(")?;
            lhs.fmt(f)?;
            write!(f, ", ")?;
            rhs.fmt(f)?;
            write!(f, ")")
        }
        else {
            write!(f, "<unknown>")
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
        //self.backing_store.clear();
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
    pub closure_ptr: &'static Closure,
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
    pub stdout: StdoutLock<'a>
    //most things are done in the "stack" for some definition of stack which is very loose here....
    //this heap is for things like dynamic tuples that could very well be infinite.
    //This makes the Value enum smaller than storing boxes directly
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
        //let let_bindings = program.strings.iter().map(|s| vec![] ).collect();
        let initial_stack = {
            let mut vec = StackData {
                backing_store: vec![],
            };
            vec.reset(program.main.layout.len());
            vec
        };
        let stdout = std::io::stdout();
        let stdout = stdout.lock();
        (
            Self {
                reusable_frames: vec![],
                functions: &program.functions,
                closure_environments: vec![],
                string_values: program.strings.clone(),
                stdout
            },
            CallFrame {
                closure_ptr: Box::leak(Box::new(Closure {
                    callable_index: 0,
                    closure_env_index: usize::MAX,
                })),
                stack_data: initial_stack,
            },
        )
    }

    fn leak<T: 'static>(&self, data: T) -> &'static T {
        let boxed = Box::new(data);
        let leaked: &'static T = Box::leak(boxed);
        leaked
    }

    #[inline(always)]
    fn eval_closure_with_env(
        &mut self,
        callable_index: usize,
        frame: &mut CallFrame,
    ) -> &'static Closure {
        let function = self.functions.get(callable_index).unwrap();

        let closure = if let Some(builder) = &function.closure_builder {
            let current_id = self.closure_environments.len();
            let vec = builder(self, frame);
            self.closure_environments.push(vec);

            Closure {
                callable_index,
                closure_env_index: current_id,
            }
        } else {
            Closure {
                callable_index,
                closure_env_index: usize::MAX,
            }
        };

        self.leak(closure)
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
        Value(0) //we can return whatever here
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
        if !condition_result.is_bool() {
            panic!("Type error: Cannot evaluate condition on this value: {condition_result:?}")
        }
        let b = condition_result.read_bool(); 
        if b {
            evaluate_then(self, frame)
        } else {
            evaluate_otherwise(self, frame)
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

        if lhs.is_int() && rhs.is_int() {
            Value::new_int(lhs.0 as i32 + rhs.0 as i32)
        }
        else if lhs.is_str() || rhs.is_str() {
            use std::fmt::Write;
            let mut buf = String::new();

            write!(&mut buf, "{}", lhs).unwrap();
            write!(&mut buf, "{}", rhs).unwrap();
            let str: &'static String = Box::leak(Box::new(buf));
            Value::make_string_ptr(str)
        }
        else {
            panic!(
                "Type error: Cannot apply binary operator + on these values: {lhs:?} and {rhs:?}"
            )
        }
    }

    #[inline(always)]
    fn fastcall(
        &mut self,
        callee_function: &'static Closure,
        arguments: &[LambdaFunction],
        frame: &mut CallFrame,
    ) -> Value {
        let Closure { callable_index, .. } = callee_function;

        let mut new_frame = self.make_new_frame(callee_function, arguments, frame);

        let function = &self.functions[*callable_index].body;
        let function_result = function(self, &mut new_frame);
    
        if function_result.is_trampoline() {
            //if the trampoline returned is not for ourselves, then we evaluate it
            let cls = function_result.get_closure();

            let Closure {
                callable_index: trampoline_callable_index,
                ..
            } = cls;

            if trampoline_callable_index != callable_index {
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
        if !callee_function.is_closure() {
            panic!("Call to non-function value {called_name}: {callee_function:?}")
        }

        let cls = callee_function.get_closure();
        let Closure { callable_index, .. } = cls;
        let callable = &self.functions[*callable_index];

        if callable.parameters.len() != arguments.len() {
            panic!("Wrong number of arguments for function {called_name}")
        }

        let mut new_frame = self.make_new_frame(cls, arguments, frame);

        let function = &callable.body;

        let function_result = function(self, &mut new_frame);

        if function_result.is_trampoline() {
            //if the trampoline returned is not for ourselves, then we evaluate it
            let closure = function_result.get_closure();
            let Closure {
                callable_index: trampoline_callable_index,
                ..
            } = closure;
            if trampoline_callable_index != callable_index {
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
        
        if !maybe_trampoline.is_trampoline() {
            return maybe_trampoline;
        };
        
        let mut current = maybe_trampoline;

        while current.is_trampoline() {
            let cls = current.get_closure();
            let Closure { callable_index, .. } = cls;
            frame.closure_ptr = cls;
            let function = &self.functions[*callable_index].body;
            current = function(self, frame);
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
        closure_ptr: &'static Closure,
        arguments: &[LambdaFunction],
        current_frame: &mut CallFrame,
    ) -> CallFrame {
        let Closure { callable_index, .. } = closure_ptr;
        let callable = &self.functions[*callable_index];

        let stack_data = if let Some(mut s) = self.reusable_frames.pop() {
            s.reset(callable.layout.len());
            s
        } else {
            StackData::new(callable.layout.len())
        };

        let mut new_frame = CallFrame {
            stack_data, // { backing_store: () },
            closure_ptr,
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

        if f.is_int() && s.is_int() {
            let f_int = f.read_int();
            let s_int = s.read_int();

            if Value::fits_in_30_bits(f_int) && Value::fits_in_30_bits(s_int) {
                return Value::new_small_tuple(f_int, s_int);
            }
        }

        let leaked: &'static _ = Box::leak(Box::new((f, s)));
        Value::make_tuple_ptr(leaked)
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
                        
                        write!(&mut ec.stdout, "{}\n", value_to_print).unwrap();

                        value_to_print
                    }),
                    fdata,
                )
            }
            Expr::Int { value } => (Box::new(move |_, _| Value::new_int(value)), function_data),
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
                let throw_at_heap = Box::new(value);
                let leaked: &'static String = Box::leak(throw_at_heap);
                let val = Value::make_string_ptr(leaked);
                (Box::new(move |_, _| val), function_data)
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
                let (evaluate_value, second_fdata) =
                    self.compile_internal(*value, function_data, current_let);
                (
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        let value = ec.run_lambda_trampoline(&evaluate_value, frame);
                        if value.is_tuple() {
                            let (a, _) = value.get_tuple();
                            *a
                        } else if value.is_small_tuple() {
                            let (a, _) = value.read_small_tuple();
                            Value::new_int(a)
                        } else {
                            panic!(
                                "Type error: Cannot evaluate first on this value: {}",
                                value
                            )
                        }
                    }),
                    second_fdata,
                )
            }
            Expr::Second { value, .. } => {
                let (evaluate_value, second_fdata) =
                    self.compile_internal(*value, function_data, current_let);
                (
                    Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        let value = ec.run_lambda_trampoline(&evaluate_value, frame);
                        if value.is_tuple() {
                            let (_, b) = value.get_tuple();
                            *b
                        } else if value.is_small_tuple() {
                            let (_, b) = value.read_small_tuple();
                            Value::new_int(b)
                        } else {
                            panic!(
                                "Type error: Cannot evaluate first on this value: {}",
                                value
                            )
                        }
                    }),
                    second_fdata,
                )
            }
            Expr::Bool { value, .. } => {
                (Box::new(move |_, _| Value::new_bool(value)), function_data)
            }
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
            VariableType::RecursiveFunction { .. } => Box::new(move |_, frame: &mut CallFrame| {
                let var = frame.closure_ptr;
                Value::make_closure(var)
            }),
            VariableType::StackPosition(pos) => Box::new(move |_, frame: &mut CallFrame| {
                let var = &frame.stack_data;
                var[pos]
            }),
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
                    } = closure_env;
                    // Can we do this in-frame? No because we can call a function that deep into its stack frame returns a closure all the way to us, at that stage
                    // we have no context where the values might have come from, they might be stack data already popped, so it will construct a closure instance and we get it here

                    let env = &ec.closure_environments[*closure_env_index];
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
                        Value::make_closure(ec.eval_closure_with_env(new_callable_index, frame))
                    })
                } else {
                    let closure: &'static Closure = Box::leak(Box::new(Closure {
                        callable_index: new_callable_index,
                        closure_env_index: usize::MAX,
                    }));
                    Box::new(move |_, _| {
                        Value::make_closure(closure)
                    })
                }
            } else if !callable.closure_vars.is_empty() {
                Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    Value::make_trampoline(ec.eval_closure_with_env(new_callable_index, frame))
                })
            } else {
                let closure: &'static Closure = Box::leak(Box::new(Closure {
                    callable_index: new_callable_index,
                    closure_env_index: usize::MAX,
                }));
                Box::new(move |_, _| {
                    Value::make_trampoline(closure)
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
        lhs: Expr,
        rhs: Expr,
        op: BinaryOp,
        function_data: FunctionData,
    ) -> Option<(LambdaFunction, FunctionData)> {
        let e = (lhs, rhs);
        let result: (LambdaFunction, FunctionData) = match e {
            //LHS is anything, RHS is int
            (lhs, Expr::Int { value: int_value }) => {
                let (index_lhs, lhs_fdata) =
                    self.compile_internal(lhs.clone(), function_data, None);

                macro_rules! rhs_const_arith {
                    ($op:tt) => {
                        (
                            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                let lhs = ec.run_lambda_trampoline(&index_lhs, frame);
                                
                                if lhs.is_int() {
                                    Value::new_int(lhs.read_int() $op int_value)
                                } else {
                                    panic!("Type error: Cannot evaluate {} on this value: {}", stringify!($op), lhs)
                                }
                            }),
                            lhs_fdata,
                        )
                    };
                }

                macro_rules! rhs_const_compare_int {
                    ($op:tt) => {
                        (
                            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                let lhs = ec.run_lambda_trampoline(&index_lhs, frame);
                                
                                if lhs.is_int() {
                                    Value::new_bool(lhs.read_int() $op int_value)
                                } else {
                                    panic!("Type error: Cannot evaluate {} on this value:{}", stringify!($op), lhs);
                                }
                            }),
                            lhs_fdata,
                        )
                    };
                }

                match op {
                    BinaryOp::Add => (
                        Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                            let lhs = ec.run_lambda_trampoline(&index_lhs, frame);
                            //let lhs_type = ValueType::from_u8(lhs.get_tag());

                            if lhs.is_int() {
                                Value::new_int(lhs.read_int() + int_value)
                            } else if lhs.is_str() {
                                let mut s = lhs.get_string().to_string();
                                s.push_str(&int_value.to_string());
                                Value::make_string_ptr(ec.leak(s))
                            } else {
                                panic!(
                                    "Type error: Cannot evaluate + on this value: {}",
                                    lhs
                                )
                            }
                        }),
                        lhs_fdata,
                    ),
                    BinaryOp::Sub => rhs_const_arith!(-),
                    BinaryOp::Mul => rhs_const_arith!(*),
                    BinaryOp::Div => rhs_const_arith!(/),
                    BinaryOp::Rem => rhs_const_arith!(%),
                    BinaryOp::Eq => rhs_const_compare_int!(==),
                    BinaryOp::Neq => rhs_const_compare_int!(!=),
                    BinaryOp::Lt => rhs_const_compare_int!(<),
                    BinaryOp::Gt => rhs_const_compare_int!(>),
                    BinaryOp::Lte => rhs_const_compare_int!(<=),
                    BinaryOp::Gte => rhs_const_compare_int!(>=),
                    BinaryOp::And => panic!("&& cannot be applied to int"),
                    BinaryOp::Or => panic!("|| cannot be applied to int"),
                }
            }
            (Expr::Int { value: int_value }, rhs) => {
                let (index_rhs, rhs_fdata) =
                    self.compile_internal(rhs.clone(), function_data, None);

                macro_rules! lhs_const_arith {
                    ($op:tt) => {
                        (
                            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                let rhs = ec.run_lambda_trampoline(&index_rhs, frame);
                                if rhs.is_int() {
                                    Value::new_int(int_value $op rhs.read_int())
                                } else {
                                    panic!("Type error: Cannot evaluate {} on this value: {}", stringify!($op), rhs)
                                }
                            }),
                            rhs_fdata,
                        )
                    };
                }

                macro_rules! lhs_const_compare_int {
                    ($op:tt) => {
                        (
                            Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                                let rhs = ec.run_lambda_trampoline(&index_rhs, frame);
                                if rhs.is_int() {
                                    Value::new_bool(int_value $op rhs.read_int())
                                } else {
                                    panic!("Type error: Cannot evaluate {} on this value: {}", stringify!($op), rhs)
                                }
                            }),
                            rhs_fdata,
                        )
                    };
                }

                match op {
                    BinaryOp::Add => (
                        Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                            let rhs = ec.run_lambda_trampoline(&index_rhs, frame);
                            if rhs.is_int() {
                                Value::new_int(int_value + rhs.read_int())
                            } else if rhs.is_str() {
                                let mut s = rhs.get_string().to_string();
                                s.push_str(&int_value.to_string());
                                Value::make_string_ptr(ec.leak(s))
                            } else {
                                panic!(
                                    "Type error: Cannot evaluate + on this value: {}",
                                    rhs
                                )
                            }
                        }),
                        rhs_fdata,
                    ),
                    BinaryOp::Sub => lhs_const_arith!(-),
                    BinaryOp::Mul => lhs_const_arith!(*),
                    BinaryOp::Div => lhs_const_arith!(/),
                    BinaryOp::Rem => lhs_const_arith!(%),
                    BinaryOp::Eq => lhs_const_compare_int!(==),
                    BinaryOp::Neq => lhs_const_compare_int!(!=),
                    BinaryOp::Lt => lhs_const_compare_int!(<),
                    BinaryOp::Gt => lhs_const_compare_int!(>),
                    BinaryOp::Lte => lhs_const_compare_int!(<=),
                    BinaryOp::Gte => lhs_const_compare_int!(>=),
                    BinaryOp::And => panic!("&& cannot be applied to int"),
                    BinaryOp::Or => panic!("|| cannot be applied to int"),
                }
            }
            
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
        let optimized = self.compile_binexp_opt(
            *lhs.clone(),
            *rhs.clone(),
            op.clone(),
            function_data.clone(),
        );
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

                    if lhs.is_int() && rhs.is_int() {
                        Value::new_int(lhs.read_int() $op rhs.read_int())
                    } else {
                        panic!("Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}", op = stringify!($op));
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
                        let same_type = lhs.get_tag() == rhs.get_tag();

                        if (same_type && (rhs.is_int() || rhs.is_bool())) {
                            return Value::new_bool(lhs.0 $op rhs.0);
                        } 
                        if (same_type && (rhs.is_str())) {
                            let lhs = lhs.get_string();
                            let rhs = rhs.get_string();
                            return Value::new_bool(lhs $op rhs);
                        } 
                        panic!("Type error: Cannot apply binary operator {op} on these values: {lhs:?} and {rhs:?}", op = stringify!($op));
                    }), fdata_rhs)
                } else {
                    (Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                        let lhs = ec.run_lambda_trampoline(&evaluate_lhs, frame);
                        let rhs = ec.run_lambda_trampoline(&evaluate_rhs, frame);

                        return Value::new_bool(lhs.0 $op rhs.0);
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
                    if lhs.is_bool() && rhs.is_bool() {
                        Value::new_bool(lhs.read_bool() && rhs.read_bool())
                    } else {
                        panic!("Type error: Cannot apply binary operator && (and) on these values: {lhs:?} and {rhs:?}")
                    }
                }),
                fdata_rhs,
            ),
            BinaryOp::Or => (
                Box::new(move |ec: &mut ExecutionContext, frame: &mut CallFrame| {
                    let lhs = ec.run_lambda_trampoline(&evaluate_lhs, frame);
                    let rhs = ec.run_lambda_trampoline(&evaluate_rhs, frame);
                   
                    if lhs.is_bool() && rhs.is_bool() {
                        Value::new_bool(lhs.read_bool() || rhs.read_bool())
                    } else {
                        panic!("Type error: Cannot apply binary operator || (or) on these values: {lhs:?} and {rhs:?}")
                    }
                }),
                fdata_rhs,
            ),
        }
    }
}

#[cfg(test)]
pub mod test {

    use crate::lambda_compiler::Value;

    #[test]
    pub fn small_tuple_test_packing_and_unpacking() {
        pub fn do_test(x: i32, y: i32) {
            let tuple = Value::new_small_tuple(x, y);
            
            if !tuple.is_small_tuple() {
                panic!("Type is not SmallTuple")
            }
            let (lhs, rhs) = tuple.read_small_tuple();
            assert!(lhs == x);
            assert!(rhs == y);
        }

        do_test(0, 0);
        do_test(-1, 1);
        do_test(536870911, -536870912);
        do_test(12386, -9875);
    }

    #[test]
    pub fn test_integer() {
        pub fn do_test(x: i32) {
            let value = Value::new_int(x);
            if !value.is_int() {
                panic!("Type is not Int")
            }
            let result = value.read_int();
            assert!(result == x);
        }

        do_test(0);
        do_test(-1);
        do_test(536870911);
        do_test(-536870912);
        do_test(i32::MAX);
        do_test(i32::MIN);
    }

    #[test]
    pub fn test_bool() {
        pub fn do_test(x: bool) {
            let value = Value::new_bool(x);
           
            if !value.is_bool() {
                panic!("Type is not Boolean")
            }
            let result = value.read_bool();
            assert!(result == x);
        }

        do_test(true);
        do_test(false);
    }

    #[test]
    pub fn test_string() {
        pub fn do_test(x: &'static String) {
            let value = Value::make_string_ptr(x);
           
            if !value.is_str() {
                panic!("Type is not String")
            }
            let result = value.get_string();
            assert!(result == x);
        }

        let s = "Hello".to_string();
        let boxed = Box::new(s);
        let leaked: &'static String = Box::leak(boxed);
        do_test(leaked);
    }

    #[test]
    pub fn test_tuple() {
        pub fn do_test(x: &'static (Value, Value)) -> &'static (Value, Value) {
            let value = Value::make_tuple_ptr(x);
           
            if !value.is_tuple() {
                panic!("Type is not TuplePtr")
            }
            let result = value.get_tuple();
            assert!(result == x);
            return result;
        }

        let s = "Hello".to_string();
        let boxed = Box::new(s);
        let leaked: &'static String = Box::leak(boxed);

        let tuple = (Value::new_int(0), Value::make_string_ptr(leaked));
        let boxed = Box::new(tuple);
        let leaked: &'static (Value, Value) = Box::leak(boxed);

        let (l, r) = do_test(leaked);

        assert!(l.read_int() == 0);
        assert!(r.get_string() == "Hello");
    }
}
