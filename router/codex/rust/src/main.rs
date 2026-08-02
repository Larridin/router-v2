use std::io::{self, Read};
use std::path::PathBuf;

use clap::Parser;
use routerlab_core::{Artifact, Router};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Route one normalized prompt embedding with a frozen artifact")]
struct Arguments {
    #[arg(long)]
    artifact: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Request {
    embedding: Vec<f32>,
    quality_bias: f64,
    eligible_models: Option<Vec<String>>,
}

fn main() {
    if let Err(error) = run() {
        println!("{}", json!({"error": error.to_string()}));
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    let router = Router::new(Artifact::load(arguments.artifact)?)?;
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let request: Request = serde_json::from_str(&input)?;
    let decision = router.route_embedding(
        &request.embedding,
        request.quality_bias,
        request.eligible_models.as_deref(),
    )?;
    println!("{}", serde_json::to_string(&decision)?);
    Ok(())
}
