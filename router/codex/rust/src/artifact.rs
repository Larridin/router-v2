use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

const SCHEMA: &str = "codex-centroid-artifact-v2";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub n_clusters: usize,
    pub top_p: usize,
    pub shrinkage: f64,
    pub temperature: f64,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct Artifact {
    pub artifact_id: String,
    pub models: Vec<String>,
    pub dimension: usize,
    pub centroids: Vec<f32>,
    pub cluster_quality: Vec<f32>,
    pub expected_cost: Vec<f64>,
    pub config: Config,
    pub embedding_model: String,
    pub embedding_source_repo: String,
    pub embedding_revision: String,
}

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("cannot read artifact: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid artifact JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid artifact: {0}")]
    Validation(String),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    artifact_id: String,
    models: Vec<String>,
    embedding: Embedding,
    config: Config,
    expected_cost: Vec<f64>,
    tensors: Tensors,
    provenance: serde_json::Map<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Embedding {
    model_id: String,
    source_repo: String,
    revision: String,
    dimension: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Tensors {
    centroids: Tensor,
    cluster_quality: Tensor,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Tensor {
    file: String,
    shape: Vec<usize>,
    sha256: String,
}

impl Artifact {
    pub fn load(directory: impl AsRef<Path>) -> Result<Self, ArtifactError> {
        let root = directory.as_ref();
        let manifest_bytes = fs::read(root.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        validate_manifest(&manifest)?;

        let identity = serde_json::to_vec(&identity_value(&manifest))?;
        let expected_id = format!("{:x}", Sha256::digest(&identity));
        if manifest.artifact_id != expected_id {
            return Err(ArtifactError::Validation(format!(
                "artifact_id does not match manifest content: expected {expected_id}, got {}",
                manifest.artifact_id
            )));
        }

        let centroids = read_tensor(root, &manifest.tensors.centroids)?;
        let cluster_quality = read_tensor(root, &manifest.tensors.cluster_quality)?;
        validate_unit_rows(&centroids, manifest.embedding.dimension)?;
        Ok(Self {
            artifact_id: manifest.artifact_id,
            models: manifest.models,
            dimension: manifest.embedding.dimension,
            centroids,
            cluster_quality,
            expected_cost: manifest.expected_cost,
            config: manifest.config,
            embedding_model: manifest.embedding.model_id,
            embedding_source_repo: manifest.embedding.source_repo,
            embedding_revision: manifest.embedding.revision,
        })
    }
}

fn validate_manifest(manifest: &Manifest) -> Result<(), ArtifactError> {
    if manifest.schema != SCHEMA {
        return invalid(format!("unknown artifact schema: {}", manifest.schema));
    }
    if manifest
        .provenance
        .get("trained_at")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return invalid("trained_at must be a non-empty string");
    }
    if manifest
        .provenance
        .iter()
        .any(|(key, value)| key != "trained_at" && contains_float(value))
    {
        return invalid("artifact provenance floats must be encoded as strings");
    }
    let unique: HashSet<&str> = manifest.models.iter().map(String::as_str).collect();
    if manifest.models.is_empty()
        || unique.len() != manifest.models.len()
        || manifest.models.iter().any(String::is_empty)
    {
        return invalid("models must be non-empty and unique");
    }
    if manifest.embedding.model_id.is_empty()
        || manifest.embedding.source_repo.is_empty()
        || manifest.embedding.revision.is_empty()
        || manifest.embedding.dimension == 0
    {
        return invalid("embedding identity and dimension must be valid");
    }
    let config = &manifest.config;
    if config.n_clusters == 0
        || config.top_p == 0
        || config.top_p > config.n_clusters
        || !config.shrinkage.is_finite()
        || config.shrinkage <= 0.0
        || !config.temperature.is_finite()
        || config.temperature <= 0.0
    {
        return invalid("centroid config is invalid");
    }
    if manifest.expected_cost.len() != manifest.models.len()
        || manifest
            .expected_cost
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return invalid("expected_cost must be finite, non-negative, and match models");
    }
    validate_tensor(
        &manifest.tensors.centroids,
        "centroids.f32",
        [config.n_clusters, manifest.embedding.dimension],
    )?;
    validate_tensor(
        &manifest.tensors.cluster_quality,
        "cluster_quality.f32",
        [config.n_clusters, manifest.models.len()],
    )?;
    Ok(())
}

fn validate_tensor(tensor: &Tensor, file: &str, shape: [usize; 2]) -> Result<(), ArtifactError> {
    if tensor.file != file || tensor.shape != shape || !is_sha256(&tensor.sha256) {
        return invalid(format!("{file} metadata or shape is invalid"));
    }
    Ok(())
}

fn read_tensor(root: &Path, tensor: &Tensor) -> Result<Vec<f32>, ArtifactError> {
    let bytes = fs::read(root.join(&tensor.file))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != tensor.sha256 {
        return invalid(format!("{} SHA-256 mismatch", tensor.file));
    }
    let expected_values = tensor.shape.iter().try_fold(1_usize, |total, value| {
        total.checked_mul(*value).ok_or_else(|| {
            ArtifactError::Validation(format!("{} shape overflows address space", tensor.file))
        })
    })?;
    let expected_bytes = expected_values
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| {
            ArtifactError::Validation(format!("{} shape overflows byte length", tensor.file))
        })?;
    if bytes.len() != expected_bytes {
        return invalid(format!("{} byte length does not match shape", tensor.file));
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return invalid(format!("{} contains non-finite values", tensor.file));
    }
    Ok(values)
}

fn validate_unit_rows(values: &[f32], columns: usize) -> Result<(), ArtifactError> {
    for row in values.chunks_exact(columns) {
        let norm = row
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > 1e-5 {
            return invalid("centroid rows must have unit length");
        }
    }
    Ok(())
}

fn identity_value(manifest: &Manifest) -> Value {
    let mut provenance = manifest.provenance.clone();
    provenance.remove("trained_at");
    json!({
        "schema": manifest.schema,
        "models": manifest.models,
        "embedding": {
            "model_id": manifest.embedding.model_id,
            "source_repo": manifest.embedding.source_repo,
            "revision": manifest.embedding.revision,
            "dimension": manifest.embedding.dimension,
        },
        "config": {
            "n_clusters": manifest.config.n_clusters,
            "top_p": manifest.config.top_p,
            "shrinkage": identity_float(manifest.config.shrinkage),
            "temperature": identity_float(manifest.config.temperature),
            "seed": manifest.config.seed,
        },
        "expected_cost": manifest.expected_cost.iter().map(|value| identity_float(*value)).collect::<Vec<_>>(),
        "tensors": {
            "centroids": {
                "file": manifest.tensors.centroids.file,
                "shape": manifest.tensors.centroids.shape,
                "sha256": manifest.tensors.centroids.sha256,
            },
            "cluster_quality": {
                "file": manifest.tensors.cluster_quality.file,
                "shape": manifest.tensors.cluster_quality.shape,
                "sha256": manifest.tensors.cluster_quality.sha256,
            },
        },
        "provenance": provenance,
    })
}

fn identity_float(value: f64) -> String {
    format!("f64:{:016x}", value.to_bits())
}

fn contains_float(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.is_f64(),
        Value::Array(values) => values.iter().any(contains_float),
        Value::Object(values) => values.values().any(contains_float),
        _ => false,
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ArtifactError> {
    Err(ArtifactError::Validation(message.into()))
}
