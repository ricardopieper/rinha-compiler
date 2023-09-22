use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, HashMap},
    env::Args,
};

use crate::{
    ast::BinaryOp,
    hir::{Expr, FuncDecl},
};

pub trait FreeVariableAnalysis {
    fn free_vars(&self) -> Vec<String>;
}

pub trait TypeSubstitutable {
    fn apply(&self, subs: &Substitutions) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeFunction {
    Function,
    Bool,
    Int,
    Tuple,
    String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeVariable(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MonoType {
    TypeVariable(TypeVariable),
    TypeFunctionApplication(TypeFunction, Vec<MonoType>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnificationError {
    InfiniteType,
    TypeMismatch,
    PolymorphicOverloadOperatorFail,
    ArgCountMismatch,
}

impl MonoType {
    pub fn contains(&self, other: &TypeVariable) -> bool {
        match self {
            MonoType::TypeVariable(a) => a == other,
            MonoType::TypeFunctionApplication(_, args) => {
                args.iter().find(|x| x.contains(other)).is_some()
            }
        }
    }

    //Robinson's unify
    pub fn unify(&self, other: &MonoType) -> Result<Substitutions, UnificationError> {
        let empty = Ok(Substitutions::new());
        match (self, other) {
            (MonoType::TypeVariable(TypeVariable(a)), MonoType::TypeVariable(TypeVariable(b)))
                if a == b =>
            {
                empty
            }
            (MonoType::TypeVariable(a), b) if b.contains(a) => Err(UnificationError::InfiniteType),
            (MonoType::TypeVariable(a), b) => {
                let mut subs = Substitutions::new();
                subs.insert(a.0.to_string(), b.clone());
                Ok(subs)
            }
            (a, b @ MonoType::TypeVariable(_)) => b.unify(a),
            (
                MonoType::TypeFunctionApplication(f1, args1),
                MonoType::TypeFunctionApplication(f2, args2),
            ) => {
                if f1 != f2 {
                    return Err(UnificationError::TypeMismatch);
                }
                if args1.len() != args2.len() {
                    return Err(UnificationError::ArgCountMismatch);
                }
                let mut subs = Substitutions::new();
                for (a, b) in args1.iter().zip(args2.iter()) {
                    let a = a.apply(&subs);
                    let b = b.apply(&subs);
                    let new_subs = a.unify(&b)?;
                    subs.extend(new_subs);
                }

                Ok(subs)
            }
        }
    }
}

impl FreeVariableAnalysis for MonoType {
    fn free_vars(&self) -> Vec<String> {
        match self {
            MonoType::TypeVariable(a) => vec![a.0.clone()],
            MonoType::TypeFunctionApplication(_, mono) => {
                mono.iter().flat_map(|m| m.free_vars()).collect()
            }
        }
    }
}

pub type Substitutions = BTreeMap<String, MonoType>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PolyType {
    Mono(MonoType),
    TypeQuantifier(String, Box<PolyType>), //forall notation
}

impl FreeVariableAnalysis for PolyType {
    fn free_vars(&self) -> Vec<String> {
        match self {
            PolyType::Mono(m) => m.free_vars(),
            PolyType::TypeQuantifier(quantifier, poly) => {
                let free_vars = poly.free_vars();
                let except_quantifier: Vec<_> =
                    free_vars.into_iter().filter(|x| x != quantifier).collect();
                except_quantifier
            }
        }
    }
}

impl PolyType {
    //Va Vb -> a -> b  becomes 't0 -> 't1
    //This means a polymorphic type becomes a instantiated type, but the types of this instantiated type are still to be inferred.
    //It just means a specific instantiation that needs to be figured out.
    pub fn instantiate(&self, typer: &Typer) -> MonoType {
        return self.instantiate_internal(typer, &mut BTreeMap::new());
    }

    pub fn original(self) -> (PolyType, Option<MonoType>) {
        let clone = self.clone();
        match self {
            PolyType::Mono(m) => {
                let cloned_m = m.clone();
                (clone, Some(cloned_m))
            }
            PolyType::TypeQuantifier(_, _) => (self, None),
        }
    }

    fn instantiate_internal(
        &self,
        typer: &Typer,
        mappings: &mut BTreeMap<String, TypeVariable>,
    ) -> MonoType {
        match self {
            PolyType::Mono(mono) => match mono {
                MonoType::TypeVariable(tv) => mappings
                    .get(&tv.0)
                    .map(|x| MonoType::TypeVariable(x.clone()))
                    .unwrap_or_else(|| MonoType::TypeVariable(tv.clone())),
                MonoType::TypeFunctionApplication(func, args) => {
                    let instantiated_args: Vec<_> = args
                        .iter()
                        .map(|x| PolyType::Mono(x.clone()).instantiate_internal(typer, mappings))
                        .collect();

                    MonoType::TypeFunctionApplication(func.clone(), instantiated_args)
                }
            },
            PolyType::TypeQuantifier(quantifier, ty) => {
                mappings.insert(quantifier.clone(), typer.create_type_variable());
                ty.instantiate_internal(typer, mappings)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Context {
    //the option is to hold the original instantiation for deferred inference
    bindings: BTreeMap<String, (PolyType, Option<MonoType>)>,
    instantiations: Vec<(String, PolyType, MonoType)>,
    current_function: Option<(String, MonoType)>,
    coming_from_let: bool
}

impl Context {
    pub fn generalize(&self, ty: &MonoType) -> PolyType {
        let ty_vars: BTreeSet<String> = ty.free_vars().into_iter().collect();
        let context_vars: BTreeSet<String> = self.free_vars().into_iter().collect();
        let free_vars = ty_vars.difference(&context_vars);
        let mut current = PolyType::Mono(ty.clone());
        for q in free_vars {
            current = PolyType::TypeQuantifier(q.to_string(), current.into());
        }
        current
    }

    pub fn get(&self, name: &str) -> Option<&(PolyType, Option<MonoType>)> {
        return self.bindings.get(name);
    }

    pub fn with(&self, name: &str, ty: PolyType, original: Option<MonoType>) -> Context {
        let mut this_bindings = self.bindings.clone();
        this_bindings.insert(name.to_string(), (ty, original));
        Context {
            bindings: this_bindings,
            instantiations: vec![],
            current_function: None, coming_from_let: false
        }
    }

    pub fn with_many(&self, new_bindings: &[(&str, (PolyType, Option<MonoType>))]) -> Context {
        let mut this_bindings = self.bindings.clone();
        for (name, ty) in new_bindings {
            this_bindings.insert(name.to_string(), ty.clone());
        }
        Context {
            bindings: this_bindings,
            instantiations: vec![],
            current_function: None, coming_from_let: false
        }
    }
}

impl FreeVariableAnalysis for Context {
    //finds variables that are not bound to any forall quantifier
    fn free_vars(&self) -> Vec<String> {
        self.bindings
            .values()
            .flat_map(|(poly, _)| poly.free_vars())
            .collect()
    }
}

impl TypeSubstitutable for Context {
    fn apply(&self, subs: &Substitutions) -> Self {
        let mut new_bindings = BTreeMap::new();

        for (k, (v, original)) in self.bindings.iter() {
            new_bindings.insert(k.to_string(), (v.apply(subs), original.clone()));
        }

        return Context {
            bindings: new_bindings,
            instantiations: vec![],
            current_function: None, coming_from_let: false
        };
    }
}

impl TypeSubstitutable for PolyType {
    fn apply(&self, subs: &Substitutions) -> Self {
        match self {
            PolyType::Mono(m) => PolyType::Mono(m.apply(subs)),
            PolyType::TypeQuantifier(quantifier, poly) => {
                PolyType::TypeQuantifier(quantifier.to_string(), poly.apply(subs).into())
            }
        }
    }
}

impl TypeSubstitutable for MonoType {
    fn apply(&self, subs: &Substitutions) -> Self {
        match self {
            MonoType::TypeVariable(var) => {
                let subs_for_var = subs.get(&var.0);
                if let Some(v) = subs_for_var {
                    v.clone()
                } else {
                    MonoType::TypeVariable(var.clone())
                }
            }
            MonoType::TypeFunctionApplication(function, args) => MonoType::TypeFunctionApplication(
                function.clone(),
                args.iter().map(|a| a.apply(subs)).collect(),
            ),
        }
    }
}

impl TypeSubstitutable for Substitutions {
    fn apply(&self, subs: &Substitutions) -> Self {
        let mut new_subs = Substitutions::new();
        for (k, v) in self.iter() {
            new_subs.insert(k.clone(), v.clone());
        }
        for (k, v) in subs.iter() {
            println!("Replacing {v:?}, subs: {self:?}");
            //this will replace previous values if they already exit
            let substituted = v.apply(self);
            new_subs.insert(k.to_string(), substituted);
        }

        new_subs
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SubstitutionArg {
    Mono(MonoType),
    Poly(PolyType),
    Context(Context),
}

#[derive(Debug)]
pub struct DeferredBinOp {
    possible_signatures: Vec<MonoType>, //possible signatures of the operation
    signature: MonoType,                //still contains type variables
}

pub struct Typer {
    next_var: Cell<usize>,
    operators: Vec<(BinaryOp, MonoType, MonoType, MonoType)>,
    pub deferred_ops: Vec<DeferredBinOp>,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Defer {
    NoDefer,
    Named(String),
    Unnamed,
}

impl Defer {
    pub fn or(self, other: Defer) -> Defer {
        match (self, other) {
            (Defer::NoDefer, d) => d,
            (Defer::Named(s), d) => Defer::Named(s),
            (d1 @ Defer::Unnamed, d2) => d2.or(d1),
        }
    }

    pub fn defers(&self) -> bool {
        match self {
            Defer::NoDefer => false,
            Defer::Named(_) => true,
            Defer::Unnamed => true,
        }
    }
}

impl Typer {
    pub fn new() -> Self {
        let mut operators = vec![];
        let int = MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]);
        let string = MonoType::TypeFunctionApplication(TypeFunction::String, vec![]);
        let bool = MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]);

        operators.push((BinaryOp::Add, int.clone(), int.clone(), int.clone()));
        operators.push((BinaryOp::Add, int.clone(), string.clone(), string.clone()));
        operators.push((BinaryOp::Add, string.clone(), int.clone(), string.clone()));
        operators.push((
            BinaryOp::Add,
            string.clone(),
            string.clone(),
            string.clone(),
        ));

        operators.push((BinaryOp::Sub, int.clone(), int.clone(), int.clone()));
        operators.push((BinaryOp::Mul, int.clone(), int.clone(), int.clone()));
        operators.push((BinaryOp::Div, int.clone(), int.clone(), int.clone()));
        operators.push((BinaryOp::Rem, int.clone(), int.clone(), int.clone()));

        operators.push((BinaryOp::Eq, int.clone(), int.clone(), bool.clone()));
        operators.push((BinaryOp::Eq, string.clone(), string.clone(), bool.clone()));
        operators.push((BinaryOp::Eq, bool.clone(), bool.clone(), bool.clone()));

        operators.push((BinaryOp::Neq, int.clone(), int.clone(), bool.clone()));
        operators.push((BinaryOp::Neq, string.clone(), string.clone(), bool.clone()));
        operators.push((BinaryOp::Neq, bool.clone(), bool.clone(), bool.clone()));

        operators.push((BinaryOp::Lt, int.clone(), int.clone(), bool.clone()));
        operators.push((BinaryOp::Gt, int.clone(), int.clone(), bool.clone()));
        operators.push((BinaryOp::Lte, int.clone(), int.clone(), bool.clone()));
        operators.push((BinaryOp::Gte, int.clone(), int.clone(), bool.clone()));

        operators.push((BinaryOp::And, bool.clone(), bool.clone(), bool.clone()));
        operators.push((BinaryOp::Or, bool.clone(), bool.clone(), bool.clone()));

        Typer {
            next_var: Cell::new(0),
            operators,
            deferred_ops: vec![],
        }
    }

    pub fn create_type_variable(&self) -> TypeVariable {
        let next = format!("'t{}", self.next_var.get());
        self.next_var.set(self.next_var.get() + 1);
        TypeVariable(next)
    }

    pub fn log(&self, s: String) {
        println!("{}", s)
    }

    pub fn algorithm_w(
        &mut self,
        type_env: &Context,
        expr: &Expr,
    ) -> Result<(Substitutions, MonoType, Defer), UnificationError> {
        self.log(format!(
            "current ctx: {type_env:#?}\nExpr being inferred: {expr:#?}\n\n\n\n"
        ));
        let result: Result<(Substitutions, MonoType, Defer), UnificationError> = match expr {
            Expr::Var { name } => {

                //check if this is a recursive definition
               // if let Some((current_function_name, args)) = &type_env.current_function {
               //     if current_function_name == name {
                        //@TODO finish
                      //  todo!()
                        //return MonoType::TypeFunctionApplication(TypeFunction::Function, 
                        //    args.clone()
                        //)
                 //   }
               /// }

                //VAR in algorithm W
                let Some((var_type, original)) = type_env.get(name) else {
                    panic!("Undefined variable {name}")
                };

                let instantiation = var_type.instantiate(self);

                self.log(format!(
                    "Type {var_type:?} got instantiated into {instantiation:?} on a VAR access"
                ));
                match original {
                    Some(_) => {
                        Ok((Substitutions::new(), instantiation, Defer::Named(name.clone())))
                    }
                    None => Ok((Substitutions::new(), instantiation, Defer::NoDefer))
                }
            }
            Expr::FuncDecl(FuncDecl { params, body, .. }) => {
                //ABS in algorithm w
                //create a new type for this parameter (it seems the video doesn't handle multiple vars, as the body would be another lambda expr)
                //maybe we can extend W to support multiple params later
                //The algorithm, when translated to this code's terms,
                //says let (s1, t1) = W(context + [name: function_type]) in (S1, S1*function_type -> t1)

                //QUESTION: Shouldn't it be:

                //let mut function_args_and_ret: Vec<_> = params.iter().map(|_| MonoType::TypeVariable(self.create_type_variable())).collect();
                //one more for the ret
                //function_args_and_ret.push(MonoType::TypeVariable(self.create_type_variable()));

                //maybe this would be a top-bottom strategy like alg M?
                // let actual_function = PolyType::Mono(
                //    MonoType::TypeFunctionApplication(TypeFunction::Function, function_args_and_ret)
                //);

                //on function declarations, if we have a current function then we are in a let binding and this function can be recursive,
                //else we just do the normal alg W

                //if type_env.coming_from_let {
                //    if let Some((name, params)) = type_env.current_function {

                //    }
                //}

                let params_args: Vec<MonoType> = params
                    .iter()
                    .map(|_| MonoType::TypeVariable(self.create_type_variable()))
                    .collect();
                let args: Vec<_> = params_args
                    .iter()
                    .zip(params)
                    .map(|(mono, p)| {
                        (
                            p.as_str(),
                            (PolyType::Mono(mono.clone()), Some(mono.clone())),
                        )
                    })
                    .collect();

                //  Vec<(&str, PolyType)> = params.iter().map(|p| (p.as_str(), PolyType::Mono(MonoType::TypeVariable(self.create_type_variable())))).collect();

                //so let's build the new context
                let new_ctx = type_env.with_many(&args);

                let (s1, t1, defer) = self.algorithm_w(&new_ctx, body)?;
                //unify t1 and the argument we have on context
                //this execution of alg w learned new substitutions s1, and inferred the body of t1 as having some type, let's apply them

                let mut function_signature = params_args;
                function_signature.push(t1);

                let function_b_to_t1 =
                    MonoType::TypeFunctionApplication(TypeFunction::Function, function_signature)
                        .apply(&s1);

                Ok((s1, function_b_to_t1, defer))
            }
            Expr::FuncCall { func, args } => {
                //var returns defer::named(f)
                let (callee_subs, callee_type, defer_call) = self.algorithm_w(&type_env, func)?;

                match defer_call.clone() {
                    Defer::NoDefer | Defer::Unnamed => {
                        let mut type_env_after_args = type_env.apply(&callee_subs);
                        let mut function_args = vec![];
                        let mut function_args_subs = None;
                        for arg in args {
                            let (s2, t2, defer) = self.algorithm_w(&type_env_after_args, arg)?;
                            function_args.push(t2);
                            type_env_after_args = type_env_after_args.apply(&s2);
                            match function_args_subs {
                                None => function_args_subs = Some(s2),
                                Some(s) => function_args_subs = Some(s.apply(&s2)),
                            }
                        }
                        let function_args_subs = match function_args_subs {
                            None => Substitutions::new(),
                            Some(s) => s,
                        };

                        let new_type_ret = MonoType::TypeVariable(self.create_type_variable());
                        let mut function_sig = function_args;
                        function_sig.push(new_type_ret.clone());
                        let s3 = callee_type.apply(&function_args_subs).unify(
                            &MonoType::TypeFunctionApplication(TypeFunction::Function, function_sig),
                        )?;

                        Ok((
                            callee_subs.apply(&function_args_subs).apply(&s3),
                            new_type_ret.apply(&s3),
                            defer_call,
                        ))
                    }
                    Defer::Named(s) => {
                        //maybe it should try resolving the defer, if it cannot, then proceed with the normal function application above
                        //find s in let bindings 
                        //this should necessarily work
                        let Some((_, Some(original))) = type_env.get(&s) else {
                            panic!("Undefined variable or without original {s}")
                        };

                        let mut type_env_after_args = type_env.apply(&callee_subs);
                        let mut function_args = vec![];
                        let mut function_args_subs = None;
                        for arg in args {
                            let (s2, t2, defer) = self.algorithm_w(&type_env_after_args, arg)?;
                            function_args.push(t2);
                            type_env_after_args = type_env_after_args.apply(&s2);
                            match function_args_subs {
                                None => function_args_subs = Some(s2),
                                Some(s) => function_args_subs = Some(s.apply(&s2)),
                            }
                        }
                        let function_args_subs = match function_args_subs {
                            None => Substitutions::new(),
                            Some(s) => s,
                        };

                        let mut unification = None;

                        let mut deferred_solved = vec![];

                        let new_type_ret = MonoType::TypeVariable(self.create_type_variable()); //t6

                        for (i, DeferredBinOp { possible_signatures, signature }) in self.deferred_ops.iter().enumerate() {
                            //signature here refers to the original signature of the binop, it would be t0 -> t1 -> t2. Do not confuse with signature of function we're calling
                            //calee_type is t5 -> t4 -> t3
                            //original is t0 -> t1 -> t2 too

                            //I need t0 -> t5, t1 -> t4, t2 -> t3
                            let subs = original.unify(&callee_type)?;
                            println!("subs: {subs:?}");
                            //now this is in t5 -> t4 -> t3 format
                            let binop_new_signature = signature.apply(&subs);
                            println!("binop_new_signature: {binop_new_signature:?}");

                            //create a function signature that matches the arguments, and has a new type variable for the return type (which is still to be inferred)
                            let mut function_sig = function_args.clone();
                            function_sig.push(new_type_ret.clone()); // int -> string -> t6
                            
                            //binop_new_signature = t5 -> t4 -> t3 
                            //function sig = int -> string -> t6

                            let s3 = binop_new_signature.apply(&function_args_subs).unify(
                                &MonoType::TypeFunctionApplication(TypeFunction::Function, function_sig),
                            )?;

                            //s3 has t5 -> int, t4 -> string, t3 -> t6
                            //
                            //transform into int -> string -> t6
                            let binop_args = binop_new_signature.apply(&s3);
                            println!("binop_args: {binop_args:?}");
                            
                            //now we find the first that unifies
                            for sig in possible_signatures.iter() {

                                let unification_result = binop_args.unify(sig);
                                match unification_result {
                                    Ok(subs) => {
                                        unification = Some((binop_args.apply(&subs), subs));
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                            if unification.is_some() {
                                deferred_solved.push(i);
                                break;
                            }
                            
                        }

                        for s in deferred_solved {
                            self.deferred_ops.remove(s);
                        }

                        match unification {
                            Some((_new_signature, deferred_subs)) => {
                                Ok((
                                    callee_subs.apply(&function_args_subs).apply(&deferred_subs),
                                    new_type_ret.apply(&deferred_subs),
                                    Defer::NoDefer
                                ))
                            }
                            None =>{
                                Err(UnificationError::PolymorphicOverloadOperatorFail)
                            },
                        }
                    }
                }
                
            }

            Expr::Let { name, value, next } => {

                let mut new_type_env = type_env.clone();

                if let Expr::FuncDecl(FuncDecl { params, body, .. }) = &**value {
                    //we will need to 
                    let mut args: Vec<_> = params
                        .iter()
                        .map(|_| MonoType::TypeVariable(self.create_type_variable()))
                        .collect();
                
                    let return_type = MonoType::TypeVariable(self.create_type_variable());
                    args.push(return_type);

                    new_type_env.current_function = Some((name.clone(), MonoType::TypeFunctionApplication(TypeFunction::Function, args)));
                    new_type_env.coming_from_let = true;
                }


                let (s1, t1, defer_f) = self.algorithm_w(type_env, value)?;
                //in order to continue we need to generalize the value

                let new_type_env = type_env.apply(&s1);

                let generalized = new_type_env.generalize(&t1);

                let new_type_env = new_type_env.with(
                    name,
                    generalized.clone(),
                    if defer_f.defers() {
                        Some(t1.clone())
                    } else {
                        None
                    },
                );

                self.log(format!(
                    "Type t1 {t1:?} got generalized to {generalized:?} in a let binding"
                ));

                let (s2, t2, defer) = self.algorithm_w(&new_type_env, next)?;

                Ok((
                    s1.apply(&s2),
                    t2,
                    if defer_f.defers() {
                        Defer::Named(name.clone())
                    } else {
                        Defer::NoDefer
                    },
                ))
            }
            Expr::Int { value } => Ok((
                Substitutions::new(),
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]),
                Defer::NoDefer,
            )),
            Expr::Bool { value } => Ok((
                Substitutions::new(),
                MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]),
                Defer::NoDefer,
            )),

            Expr::String { value } => Ok((
                Substitutions::new(),
                MonoType::TypeFunctionApplication(TypeFunction::String, vec![]),
                Defer::NoDefer,
            )),
            Expr::If {
                cond,
                then,
                otherwise,
            } => {
                let (cond_s, cond_t, defer1) = self.algorithm_w(type_env, cond)?;
                //any LETs in cond are not used in then/otherwise, they don't escape their scope

                //cond_ty must unify with bool
                cond_t.unify(&MonoType::TypeFunctionApplication(
                    TypeFunction::Bool,
                    vec![],
                ))?;

                let (then_s, then_t, defer2) = self.algorithm_w(&type_env.apply(&cond_s), &then)?;

                let (otherwise_s, otherwise_t, defer3) =
                    self.algorithm_w(&type_env.apply(&cond_s).apply(&then_s), &otherwise)?;
                //then and otherwise must unify
                let s = then_t.unify(&otherwise_t)?;
                Ok((
                    (&cond_s.apply(&then_s).apply(&otherwise_s)).apply(&s),
                    then_t,
                    defer1.or(defer2).or(defer3),
                ))
            }
            Expr::Tuple { first, second } => {
                //let bindings in first don't flow into second, (? maybe they do change the type environment in some other way with substitutions?)
                let (first_s, first_t, d1) = self.algorithm_w(type_env, first)?;
                let (second_s, second_t, d2) =
                    self.algorithm_w(&type_env.apply(&first_s), second)?;

                Ok((
                    first_s.apply(&second_s),
                    MonoType::TypeFunctionApplication(TypeFunction::Tuple, vec![first_t, second_t]),
                    d1.or(d2),
                ))
            }
            Expr::First { value } => {
                //value is supposed to be a tuple here
                let (s1, tuple, d1) = self.algorithm_w(type_env, value)?;

                //create 2 type variables for each side
                let t1 = self.create_type_variable();
                let t2 = self.create_type_variable();

                let generic_tuple_type = MonoType::TypeFunctionApplication(
                    TypeFunction::Tuple,
                    vec![
                        MonoType::TypeVariable(t1.clone()),
                        MonoType::TypeVariable(t2),
                    ],
                );

                let s = tuple.unify(&generic_tuple_type)?;

                //applies substitutions from the unification above
                let new_s = s1.apply(&s);

                //find in the resulting substitutions the t1 type
                let t1_type = new_s.get(&t1.0).unwrap();

                Ok((s1, t1_type.clone(), d1))
            }
            Expr::Second { value } => {
                let (s1, tuple, d1) = self.algorithm_w(type_env, value)?;

                //create 2 type variables for each side
                let t1 = self.create_type_variable();
                let t2 = self.create_type_variable();

                let generic_tuple_type = MonoType::TypeFunctionApplication(
                    TypeFunction::Tuple,
                    vec![
                        MonoType::TypeVariable(t1),
                        MonoType::TypeVariable(t2.clone()),
                    ],
                );

                let s = tuple.unify(&generic_tuple_type)?;

                let new_s = s1.apply(&s);
                //find in substitutions the t1 type
                let t2_type = new_s.get(&t2.0).unwrap();

                Ok((s1, t2_type.clone(), d1))
            }
            Expr::Print { value } => {
                let (s1, t1, d1) = self.algorithm_w(type_env, value)?;
                Ok((s1, t1, d1))
            }
            Expr::BinOp { op, left, right } => {
                let (sleft, tleft, d1) = self.algorithm_w(type_env, left)?;

                let new_env = type_env.apply(&sleft);
                let (sright, tright, d2) = self.algorithm_w(&new_env, right)?;

                let result_type = MonoType::TypeVariable(self.create_type_variable());
                let desired_signature = MonoType::TypeFunctionApplication(
                    TypeFunction::Function,
                    vec![tleft, tright, result_type.clone()],
                );

                let supported_sigs = self
                    .operators
                    .iter()
                    .filter(|(supported_op, _, _, _)| supported_op == op);
                //for each supported signature for this operator, we defer the inference when more information becomes available

                let mut deferred_op_signatures = vec![];

                for (_, lhs, rhs, result) in supported_sigs {
                    deferred_op_signatures.push(MonoType::TypeFunctionApplication(
                        TypeFunction::Function,
                        vec![lhs.clone(), rhs.clone(), result.clone()],
                    ));
                }

                let desired_sig_replaced = desired_signature.apply(&sright);
                self.deferred_ops.push(DeferredBinOp {
                    possible_signatures: deferred_op_signatures,
                    signature: desired_sig_replaced.clone(),
                });

                Ok((sleft.apply(&sright), result_type, Defer::Unnamed))
                //Ok(( ))
            }
        };

        match result {
            Ok((s, ty, defer)) => {
                self.log(format!(
                    "Alg W finished, type: {ty:?}, subs: {s:#?}, bindings: {b:#?}",
                    b = type_env.bindings
                ));
                Ok((s, ty, defer))
            }
            Err(e) => {
                self.log(format!("Alg W error! {e:?}"));
                Err(e)
            }
        }
    }
}

#[cfg(test)]
pub mod tests {
    use std::{cell::Cell, collections::BTreeMap};

    use crate::{typing::{
        Context, MonoType, PolyType, Substitutions, TypeFunction, TypeSubstitutable, TypeVariable,
    }, ast::Term};

    #[cfg(test)]
    use pretty_assertions::assert_eq;


    use super::{Defer, Typer, UnificationError};

    fn to_hir(term: Term) -> crate::hir::Expr {
        return crate::hir::ast_to_hir(term);
    }

    fn run_w(text: &str) -> Result<(Substitutions, MonoType, Defer), UnificationError> {
        let file = crate::parser::parse_or_report("test_file", text).unwrap();
        let hir = to_hir(file.expression);
        let mut typer = Typer::new();
        let ctx = Context {
            bindings: BTreeMap::new(),
            instantiations: vec![],
            current_function: None, coming_from_let: false
        };
        let result = typer.algorithm_w(&ctx, &hir);

        println!("Deferred ops: {def:#?}", def = typer.deferred_ops);

        result
    }

    #[test]
    pub fn type_variable_substitution() {
        let ty = MonoType::TypeVariable(TypeVariable("a".to_string()));
        let subst = Substitutions::from([(
            "a".to_string(),
            MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]),
        )]);

        let result = ty.apply(&subst);
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![])
        );
    }

    #[test]
    pub fn type_function_application_substitution() {
        let ty = MonoType::TypeFunctionApplication(
            crate::typing::TypeFunction::Function,
            vec![MonoType::TypeVariable(TypeVariable("a".to_string()))],
        );
        let subst = Substitutions::from([(
            "a".to_string(),
            MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]),
        )]);

        let result = ty.apply(&subst);
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(
                crate::typing::TypeFunction::Function,
                vec![MonoType::TypeFunctionApplication(
                    TypeFunction::Bool,
                    vec![]
                )]
            )
        );
    }

    #[test]
    pub fn poly_type_quantifier_substitution() {
        //forall a: a -> a
        let ty = PolyType::TypeQuantifier(
            "a".to_string(),
            PolyType::Mono(MonoType::TypeVariable(TypeVariable("a".to_string()))).into(),
        );

        let subst = Substitutions::from([(
            "a".to_string(),
            MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]),
        )]);

        let result = ty.apply(&subst);
        assert_eq!(
            result,
            PolyType::TypeQuantifier(
                "a".to_string(),
                PolyType::Mono(MonoType::TypeFunctionApplication(
                    TypeFunction::Bool,
                    vec![]
                ))
                .into()
            )
        );
    }

    #[test]
    pub fn context_substitution() {
        let ctx = Context {
            instantiations: vec![],
            current_function: None, coming_from_let: false,
            bindings: BTreeMap::from([
                (
                    "x".to_string(),
                    PolyType::Mono(MonoType::TypeFunctionApplication(
                        TypeFunction::Bool,
                        vec![],
                    ))
                    .original(),
                ), //remains unchanged
                (
                    "y".to_string(),
                    PolyType::TypeQuantifier(
                        "a".to_string(),
                        PolyType::Mono(MonoType::TypeFunctionApplication(
                            TypeFunction::Bool,
                            vec![],
                        ))
                        .into(),
                    )
                    .original(),
                ), //unchanged too
                (
                    "y".to_string(),
                    PolyType::TypeQuantifier(
                        "a".to_string(),
                        PolyType::Mono(MonoType::TypeVariable(TypeVariable("a".to_string())))
                            .into(),
                    )
                    .original(),
                ),
            ]),
        };

        let subst = Substitutions::from([(
            "a".to_string(),
            MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]),
        )]);

        let result = ctx.apply(&subst);

        let expected = Context {
            instantiations: vec![],
            current_function: None, coming_from_let: false,
            bindings: BTreeMap::from([
                (
                    "x".to_string(),
                    PolyType::Mono(MonoType::TypeFunctionApplication(
                        TypeFunction::Bool,
                        vec![],
                    ))
                    .original(),
                ), //remains unchanged
                (
                    "y".to_string(),
                    PolyType::TypeQuantifier(
                        "a".to_string(),
                        PolyType::Mono(MonoType::TypeFunctionApplication(
                            TypeFunction::Bool,
                            vec![],
                        ))
                        .into(),
                    )
                    .original(),
                ), //unchanged too
                (
                    "y".to_string(),
                    PolyType::TypeQuantifier(
                        "a".to_string(),
                        PolyType::Mono(MonoType::TypeFunctionApplication(
                            TypeFunction::Bool,
                            vec![],
                        ))
                        .into(),
                    )
                    .original(),
                ),
            ]),
        };
        assert_eq!(expected, result);
    }

    #[test]
    pub fn combining_substitutions() {
        let subst1 = Substitutions::from([(
            "x".to_string(),
            MonoType::TypeVariable(TypeVariable("y".to_string())),
        )]);

        let subst2 = Substitutions::from([(
            "z".to_string(),
            MonoType::TypeFunctionApplication(
                TypeFunction::Function,
                vec![
                    MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]),
                    MonoType::TypeVariable(TypeVariable("x".to_string())),
                ],
            ),
        )]);

        let result = subst1.apply(&subst2);

        let expected = BTreeMap::from([
            (
                "x".to_string(),
                MonoType::TypeVariable(TypeVariable("y".to_string())),
            ),
            (
                "z".to_string(),
                MonoType::TypeFunctionApplication(
                    TypeFunction::Function,
                    vec![
                        MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]),
                        MonoType::TypeVariable(TypeVariable("y".to_string())),
                    ],
                ),
            ),
        ]);

        assert_eq!(expected, result);
    }

    #[test]
    pub fn poly_type_instantiation() {
        let poly = PolyType::TypeQuantifier(
            "TIn".to_string(),
            PolyType::TypeQuantifier(
                "TOut".to_string(),
                PolyType::Mono(MonoType::TypeFunctionApplication(
                    TypeFunction::Function,
                    vec![
                        MonoType::TypeVariable(TypeVariable("TIn".to_string())),
                        MonoType::TypeVariable(TypeVariable("TOut".to_string())),
                    ],
                ))
                .into(),
            )
            .into(),
        );

        let mut typer = Typer::new();
        let instantiated = poly.instantiate(&mut typer);
        let expected = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeVariable(TypeVariable("'t0".to_string())),
                MonoType::TypeVariable(TypeVariable("'t1".to_string())),
            ],
        );

        assert_eq!(instantiated, expected);
    }

    #[test]
    pub fn mono_type_generalization_already_bound_variable_in_context_does_not_generalize() {
        let ctx = Context {
            instantiations: vec![],
            current_function: None, coming_from_let: false,
            bindings: BTreeMap::from([(
                "x".to_string(),
                PolyType::Mono(MonoType::TypeVariable(TypeVariable("'t0".to_string()))).original(),
            )]),
        };

        let generalized = ctx.generalize(&MonoType::TypeVariable(TypeVariable("'t0".to_string())));

        assert_eq!(
            generalized,
            PolyType::Mono(MonoType::TypeVariable(TypeVariable("'t0".to_string())))
        );
    }

    #[test]
    pub fn mono_type_generalization_unbound_variable_in_context_generalizes() {
        let ctx = Context {
            instantiations: vec![],
            current_function: None, coming_from_let: false,
            bindings: BTreeMap::from([(
                "x".to_string(),
                PolyType::Mono(MonoType::TypeVariable(TypeVariable("'t0".to_string()))).original(),
            )]),
        };

        let generalized = ctx.generalize(&MonoType::TypeVariable(TypeVariable("'t1".to_string())));

        assert_eq!(
            generalized,
            PolyType::TypeQuantifier(
                "'t1".to_string(),
                PolyType::Mono(MonoType::TypeVariable(TypeVariable("'t1".to_string()))).into()
            )
        );
    }

    #[test]
    pub fn unify_primitive() -> Result<(), UnificationError> {
        let primitives = [
            TypeFunction::Int,
            TypeFunction::Bool,
            TypeFunction::String,
            TypeFunction::Tuple,
        ];
        for p in primitives {
            let t1 = MonoType::TypeFunctionApplication(p.clone(), vec![]);
            let t2 = MonoType::TypeFunctionApplication(p, vec![]);
            let result = t1.unify(&t2)?;
            assert_eq!(result.len(), 0);
        }

        Ok(())
    }

    #[test]
    pub fn same_type_variable() -> Result<(), UnificationError> {
        let t1 = MonoType::TypeVariable(TypeVariable("a".to_string()));
        let t2 = MonoType::TypeVariable(TypeVariable("a".to_string()));
        let result = t1.unify(&t2)?;
        assert_eq!(result.len(), 0);

        Ok(())
    }

    #[test]
    pub fn type_variable_occurs_check_triggers() {
        let t1 = MonoType::TypeVariable(TypeVariable("a".to_string()));
        let t2 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![MonoType::TypeVariable(TypeVariable("a".to_string()).into())],
        );
        let result = t1.unify(&t2).expect_err("Expected infinite type error");
        assert_eq!(result, UnificationError::InfiniteType);
    }

    #[test]
    pub fn type_variable_substitution_in_unification() -> Result<(), UnificationError> {
        let t1 = MonoType::TypeVariable(TypeVariable("a".to_string()));
        let t2 = MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]);
        let result = t1.unify(&t2)?;
        assert!(result["a"] == MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]));
        Ok(())
    }

    #[test]
    pub fn type_variable_substitution_function_in_unification() -> Result<(), UnificationError> {
        let t1 = MonoType::TypeVariable(TypeVariable("a".to_string()));
        let t2 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]),
                MonoType::TypeFunctionApplication(TypeFunction::String, vec![]),
            ],
        );

        let result = t1.unify(&t2)?;
        assert!(result["a"] == t2);
        Ok(())
    }

    #[test]
    pub fn type_variable_substitution_another_variable_in_unification(
    ) -> Result<(), UnificationError> {
        let t1 = MonoType::TypeVariable(TypeVariable("a".to_string()));
        let t2 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]),
                MonoType::TypeFunctionApplication(TypeFunction::String, vec![]),
                MonoType::TypeVariable(TypeVariable("b".to_string()).into()),
            ],
        );
        let result = t1.unify(&t2)?;
        assert!(result["a"] == t2);
        Ok(())
    }

    #[test]
    pub fn unification_different_types_fail() {
        let t1 = MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]);
        let t2 = MonoType::TypeFunctionApplication(TypeFunction::String, vec![]);
        let result = t1.unify(&t2).expect_err("Expected type mismatch error");
        assert_eq!(result, UnificationError::TypeMismatch);
    }

    #[test]
    pub fn unification_argcount_fail() {
        let t1 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]),
                MonoType::TypeVariable(TypeVariable("b".to_string()).into()),
            ],
        );
        let t2 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]),
                MonoType::TypeFunctionApplication(TypeFunction::String, vec![]),
                MonoType::TypeVariable(TypeVariable("b".to_string()).into()),
            ],
        );
        let result = t1
            .unify(&t2)
            .expect_err("Expected argument count type error");
        assert_eq!(result, UnificationError::ArgCountMismatch);
    }

    #[test]
    pub fn function_unification() -> Result<(), UnificationError> {
        let t1 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]),
                MonoType::TypeVariable(TypeVariable("a".to_string()).into()),
            ],
        );
        let t2 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeVariable(TypeVariable("a".to_string()).into()),
                MonoType::TypeVariable(TypeVariable("a".to_string()).into()),
            ],
        );
        let result = t1.unify(&t2)?;
        assert!(result["a"] == MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]));
        Ok(())
    }

    #[test]
    pub fn function_unification_only_variables() -> Result<(), UnificationError> {
        let t1 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeVariable(TypeVariable("a".to_string()).into()),
                MonoType::TypeVariable(TypeVariable("b".to_string()).into()),
                MonoType::TypeVariable(TypeVariable("c".to_string()).into()),
            ],
        );
        let t2 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeVariable(TypeVariable("d".to_string()).into()),
                MonoType::TypeVariable(TypeVariable("e".to_string()).into()),
                MonoType::TypeVariable(TypeVariable("f".to_string()).into()),
            ],
        );
        let result = t1.unify(&t2)?;
        println!("{result:#?}");
        Ok(())
    }

    #[test]
    pub fn function_unification_conflicting_requirements() {
        //should a be int or bool? neither, this should fail
        let t1 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![]),
                MonoType::TypeVariable(TypeVariable("a".to_string()).into()),
            ],
        );
        let t2 = MonoType::TypeFunctionApplication(
            TypeFunction::Function,
            vec![
                MonoType::TypeVariable(TypeVariable("a".to_string()).into()),
                MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![]),
            ],
        );
        let result = t1.unify(&t2).expect_err("Expected type mismatch error");
        assert_eq!(result, UnificationError::TypeMismatch);
    }

    //Algorithm W tests
    #[test]
    pub fn trivial_typing_example() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w("0")?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn literal_binding() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w("let x = 0; x")?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn function_returns_int() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w(
            "
let x = fn() => {
    1
};
x()",
        )?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn identity_function() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w(
            "
let x = fn(x) => {
    x
};
x(10)",
        )?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn simple_binop() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w("1 + 2")?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn simple_binop_string() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w("1 + \"2\"")?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::String, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn no_binary_op_works() {
        let result = run_w("1 == \"2\"");
        assert_eq!(result, Err(UnificationError::TypeMismatch));
    }

    #[test]
    pub fn bool_op() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w("\"1\" == \"2\"")?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn function_with_bin_op_and_args_plus() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w(
            "
let x = fn(a, b) => {
    a + b
};
x(10, 20)",
        )?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn function_with_bin_op_and_args_equals() -> Result<(), UnificationError> {
        let (_, result, _defer) = run_w(
            "
let equals = fn(a, b) => {
    a == b
};
equals(10, 20)",
        )?;
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn function_with_bin_op_and_args_str_concat() -> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let plus = fn(a, b) => {
    a + b
};
plus(10, \"20\")",
        )?;

        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::String, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn function_with_bin_op_and_args_str_concat_str_first() -> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let plus = fn(a, b) => {
    a + b
};
plus(\"10\", 20)",
        )?;

        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::String, vec![])
        );
        Ok(())
    }


    #[test]
    pub fn tuple_first() -> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let t = (1, true);
first(t)
")?;

        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn tuple_second() -> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let t = (1, true);
second(t)
")?;

        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Bool, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn tuple_infer_function() -> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let one = fn() => { 1 };
let t = (1, one);
second(t)
")?;

        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
            MonoType::TypeFunctionApplication(TypeFunction::Function, vec![
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
            ])
        );
        Ok(())
    }


    #[test]
    pub fn tuple_infer_plus() -> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let plus = fn(a, b) => {
    a + b
};
let t = (1, 2);
let f = first(t);
let s = second(t);
plus(f, s)
")?;

        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }


    #[test]
    pub fn tuple_infer_plus_but_theres_another_plus() -> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let plus = fn(a, b) => {
    a + b
};
let plus2 = fn(a, b) => {
    a + b
};
let t = (1, 2);
let f = first(t);
let s = second(t);
plus(f, s)
")?;

        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn function_minus_wrong_args_fail() {
        let result = run_w(
            "
let minus = fn(a, b) => {
    a - b
};
minus(10, \"20\")",
        );

    
        assert_eq!(
            result,
            Err(UnificationError::PolymorphicOverloadOperatorFail)
        );
    }

    #[test]
    pub fn function_minus_correct_args_work()-> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let minus = fn(a, b) => {
    a - b
};
minus(10, 20)",
        )?;

    
        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }

    #[test]
    pub fn math()-> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let some_math = fn(a, b) => {
    (a + b) - b
};
some_math(10, 20)",
        )?;

    
        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }
    

    #[test]
    pub fn math_wrong() {
        let result = run_w(
            "
let some_math = fn(a, b) => {
    (a + b) - b
};
some_math(10, \"20\")",
        ); 

        assert_eq!(
            result,
            Err(UnificationError::PolymorphicOverloadOperatorFail)
        );
    }


    #[test]
    pub fn fib()-> Result<(), UnificationError> {
        let (subs, result, _) = run_w(
            "
let fib = fn (n) => {
  if (n < 2) {
    n
  } else {
    fib(n - 1) + fib(n - 2)
  }
};
fib(30)",
        )?;

    
        println!("Subs: {subs:#?}, result: {result:?}");
        assert_eq!(
            result,
                MonoType::TypeFunctionApplication(TypeFunction::Int, vec![])
        );
        Ok(())
    }
}
