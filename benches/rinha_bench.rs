use criterion::{BenchmarkId, Criterion};
use lambda_rinha::{
    lambda_compiler::{CompilationResult, ExecutionContext, LambdaCompiler},
    parser, hir::ast_to_hir,
};

fn compile(text: &str) -> CompilationResult {
    let file = parser::parse_or_report("bench_test", text).unwrap();

    let compiler = LambdaCompiler::new();
    let hir = ast_to_hir(file.expression);
    let program = compiler.compile(hir);

    return program;
}

const PERF_PROGRAM: &'static str = "
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

  iter(0, 200, work_closure, 0)
};

let iteration = iter(0, 100, work, 0);

print(iteration)
";

pub fn criterion_benchmark(c: &mut Criterion) {

    let mut group = c
      .benchmark_group("rinha-bench");
    group.sample_size(30);
    group.bench_with_input(
        BenchmarkId::new("rinha-bench", "iterative"),
        &PERF_PROGRAM,
        |b, &program| {
            let compiled = compile(&program);

            b.iter(|| {
                let mut ec = ExecutionContext::new(&compiled);
                (compiled.main)(&mut ec);
            });
        },
    );
}

fn main(){
    
  // make Criterion listen to command line arguments
  let mut c = Criterion::default().configure_from_args();
  let str_unix_time = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap()
    .as_secs()
    .to_string();
  //let mut c = c.save_baseline(str_unix_time);

  criterion_benchmark(&mut c);

}