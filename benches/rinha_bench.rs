use std::mem::MaybeUninit;

use criterion::{BenchmarkId, Criterion};
use dyn_stack::{StackReq, DynStack};
use lambda_rinha::{
    hir::ast_to_hir,
    lambda_compiler::{CompilationResult, ExecutionContext, LambdaCompiler, Value},
    parser,
};

fn compile(text: &str) -> CompilationResult {
    let file = parser::parse_or_report("bench_test", text).unwrap();

    let compiler = LambdaCompiler::new();
    let hir = ast_to_hir(file.expression);

    compiler.compile(hir)
}

const PERF_PROGRAM: &str = "
let iter = fn (from, to, call, prev) => {
  if (from < to) {
    let res = call(from);
    iter(from + 1, to, call, res)
  } else {
    prev
  }
};

let work = fn(x) => {
  let work_closure = fn(y) => {
    let xx = x * y;
    let tupl = (xx, x);
    let f = first(tupl);
    let s = second(tupl);
    f * s
  };

  iter(0, 500, work_closure, 0)
};

<<<<<<< HEAD
iter(0, 500, work, 0)
=======
let iteration = iter(0, 100, work, 0);

print(iteration)
>>>>>>> actual-stack
";

const FIB_30: &str = "
let fib = fn (n) => {
  if (n < 2) {
    n
  } else {
    fib(n - 1) + fib(n - 2)
  }
};
print (\"fib\" + fib(30))
";

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("rinha-bench");
    group.sample_size(40);
    group.bench_with_input(
        BenchmarkId::new("rinha-bench", "iterative"),
        &PERF_PROGRAM,
        |b, &program| {
            let compiled = compile(&program);
            /*
            let mut buf = [MaybeUninit::uninit();
            StackReq::new::<Value>(2000000)
                .unaligned_bytes_required()];

            let mut stack = DynStack::new(&mut buf);*/

            b.iter(move || {
                let (mut ec, mut frame) = ExecutionContext::new(&compiled);
                (compiled.main.body)(&mut ec, &mut frame);
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("rinha-bench-fib", "fib30"),
        &FIB_30,
        |b, &program| {
            let compiled = compile(program);
            b.iter(|| {
              let (mut ec, mut frame) = ExecutionContext::new(&compiled);
                (compiled.main.body)(&mut ec, &mut frame);
            });
        },
    );
}

fn main() {
    // make Criterion listen to command line arguments
    let mut c = Criterion::default().configure_from_args();
    let _str_unix_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string();
    //let mut c = c.save_baseline(str_unix_time);



    criterion_benchmark(&mut c);
}
