#![feature(box_patterns)]
#![feature(type_changing_struct_update)]
#![recursion_limit = "256"]
#![feature(assert_matches)]
#![feature(let_chains)]

use clap::Parser;
use lalrpop_util::lalrpop_mod;
use miette::IntoDiagnostic;
use owo_colors::OwoColorize;
use crate::ast::File;

// The lalrpop module, it does generate the parser and lexer
// for the language.
lalrpop_mod! {
    #[allow(warnings)]
    /// The parsing module
    pub rinha
}

/// The abstract syntax tree for the language. The abstract
/// syntax tree is the tree that represents the program
/// in a tree form.
pub mod ast;

/// Parser LALRPOP module. It does uses a parse generator to
/// generate a parser and lexer for the language.
pub mod parser;
pub mod hir;

pub mod lambda_compiler;

/// Simple program to run `rinha` language.
#[derive(clap::Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Command {
    #[clap(long, short, default_value = "false")]
    pub pretty: bool,

    /// The file we would like to run, type check, etc
    pub main: String,

    #[clap(long, short, default_value = "true")]
    pub rinha_mode: bool,
}

/// Logger function for the fern logger.
///
/// It does format the log message to a specific format.
fn log(out: fern::FormatCallback, message: &std::fmt::Arguments, record: &log::Record) {
    let style = match record.level() {
        log::Level::Error => owo_colors::Style::new().red().bold(),
        log::Level::Warn => owo_colors::Style::new().yellow().bold(),
        log::Level::Info => owo_colors::Style::new().bright_blue().bold(),
        log::Level::Debug => owo_colors::Style::new().bright_red().bold(),
        log::Level::Trace => owo_colors::Style::new().bright_cyan().bold(),
    };
    let level = record.level().to_string().to_lowercase();
    let level = level.style(style);

    out.finish(format_args!("  {level:>7} {}", message))
}


/// The main function of the program.
fn program() -> miette::Result<()> {
    let start = std::time::Instant::now();
    // Initialize the bupropion handler with miette
    bupropion::BupropionHandlerOpts::install(|| {
        // Build the bupropion handler options, for specific
        // error presenting.
        bupropion::BupropionHandlerOpts::new()
    })
    .into_diagnostic()?;

    // Initialize the logger
    fern::Dispatch::new() // Perform allocation-free log formatting
        .format(log) // Add blanket level filter -
        .level(log::LevelFilter::Debug) // - and per-module overrides
        .level_for("hyper", log::LevelFilter::Info) // Output to stdout, files, and other Dispatch configurations
        .chain(std::io::stdout())
        .apply()
        .into_diagnostic()?;

    // Parse the command line arguments
    let command = Command::parse();
    let file = std::fs::read_to_string(&command.main).into_diagnostic()?;
    if command.rinha_mode {

        //deserialize into file

        let file: File = serde_json::from_str(&file).into_diagnostic()?;
        let compiler = lambda_compiler::LambdaCompiler::new();
        let hir = hir::ast_to_hir(file.expression);
        let program = compiler.compile(hir);
        let mut ee = lambda_compiler::ExecutionContext::new(&program);
        (program.main)(&mut ee);

    } else {
     
        let file = crate::parser::parse_or_report(&command.main, &file)?;
        let end = std::time::Instant::now();
    
        println!("Parse Time: {:?}", end - start);
        let start = std::time::Instant::now();
    
        let compiler = lambda_compiler::LambdaCompiler::new();
        let hir = hir::ast_to_hir(file.expression);
        let program = compiler.compile(hir);
        let end = std::time::Instant::now();
    
        println!("Compile Time: {:?}", end - start);
        let mut ee = lambda_compiler::ExecutionContext::new(&program);
    
        let start = std::time::Instant::now();
        (program.main)(&mut ee);
        let end = std::time::Instant::now();
        println!("Run Time: {:?}", end - start);
        println!("Stats: {:#?}", ee.stats);
    
    }
    Ok(())
}

// The main function wrapper around [`crate::program`].
fn main() {
    // Avoid printing print `Error: ` before the error message
    // to maintain the language beauty!
    if let Err(e) = program() {
        eprintln!("{e:?}");
        std::process::exit(1);
    }
}
