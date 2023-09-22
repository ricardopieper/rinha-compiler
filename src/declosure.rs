use crate::hir::{Expr, Type, ClosureType, FuncDecl};

pub type ParamSet = Vec<Type>;


pub struct ReplacementItem {
    pub function_full_name: String,
    pub all_parameters: Vec<String>,
    pub original_function: FuncDecl
}

pub struct Declosurer {

    pending_replace: Vec<ReplacementItem>,
}


impl Declosurer {
    //finds functions declared inside functions, see what they close over, and then
    //generate a function that takes those variables as regular parameters
    //every closure has a path, for instance: __root__::foo::bar, where foo is a root-level function and bar is a nested function inside foo.
    //Therefore the first call to declousurize will be with __root__ as the current_function.

    //The replacements will be name-based, i.e. if a closure closes over a variable x, then the generated function will have a parameter x.
    /* 
    pub fn declosurize(&mut self, e: Expr, current_function: String) -> Expr {

        //first, find all functions declared inside this function that we will need to replace later
        //then after this process we will have a list of functions to replace, we do another pass to replace them

        self.make_replacements(&e, current_function);
        todo!()
    }

    fn make_replacements(&mut self, e: &Expr, current_function: String) {
        match e {
            Expr::FuncDecl(fdecl @ FuncDecl { name, args, body, return_ty }) => {
                let closure_vars = self.find_all_var_refs(&*body);
                let mut new_args = args.clone();
        
                for var in closure_vars.iter() {
                    if let None = args.iter().find(|(name, _) | name == var) {
                        new_args.push((var.clone(), Type::Int));
                    }
                }

                let replaced_function = format!("{}::{}", current_function, name);
                //now we know all the args. Record this function for replacement
                self.pending_replace.push(ReplacementItem {
                    function_full_name: replaced_function.clone(),
                    all_parameters: new_args.iter().map(|(name, _)| name.clone()).collect(),
                    original_function: fdecl.clone()
                });

                self.make_replacements(body, replaced_function);
            }
            _ => {
                
            }
        }
    }

    //recursively find all vars inside the function including nested functions

    pub fn find_all_var_refs(&mut self, e: &Expr) -> Vec<String> {

        match e {
            Expr::Let { name, value, let_type, next } =>  {
                let mut vars = self.find_all_var_refs(&*value);
                vars.append(&mut self.find_all_var_refs(&*next));
                vars
            }
            Expr::Var { name, ty } => {
                vec![name.clone()]
            }
            Expr::Int { value } => vec![],
            Expr::Bool { value } =>vec![],
            Expr::String { value } =>vec![],
            Expr::If { cond, then, otherwise } => {
                let mut vars = self.find_all_var_refs(&*cond);
                vars.append(&mut self.find_all_var_refs(&*then));
                vars.append(&mut self.find_all_var_refs(&*otherwise));
                vars
            }
            Expr::Tuple { first, second } => {
                let mut vars = self.find_all_var_refs(&*first);
                vars.append(&mut self.find_all_var_refs(&*second));
                vars
            }
            Expr::First { value, ty } => self.find_all_var_refs(&*value),
            Expr::Second { value, ty } => self.find_all_var_refs(&*value),
            Expr::Print { value, ty } => self.find_all_var_refs(&*value),
            Expr::FuncDecl(FuncDecl { name, args, body, return_ty }) => {
                let vars = self.find_all_var_refs(&*body);
                vars
            }
            Expr::FuncCall { name, args, ty } => {
                let mut vars = vec![];
                for arg in args {
                    vars.append(&mut self.find_all_var_refs(&*arg));
                }
                vars
            }
            Expr::BinOp { op, left, right, ty } => {
                let mut vars = self.find_all_var_refs(&*left);
                vars.append(&mut self.find_all_var_refs(&*right));
                vars
            }
        }

    }
    */
}