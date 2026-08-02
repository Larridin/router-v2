use std::fs;

use serde_json::json;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use routerlab_core::Artifact;

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn artifact_id(manifest: &serde_json::Value) -> String {
    let mut identity = manifest.clone();
    identity.as_object_mut().unwrap().remove("artifact_id");
    identity["provenance"]
        .as_object_mut()
        .unwrap()
        .remove("trained_at");
    for field in ["shrinkage", "temperature"] {
        let value = identity["config"][field].as_f64().unwrap();
        identity["config"][field] = json!(format!("f64:{:016x}", value.to_bits()));
    }
    for value in identity["expected_cost"].as_array_mut().unwrap() {
        let number = value.as_f64().unwrap();
        *value = json!(format!("f64:{:016x}", number.to_bits()));
    }
    sha256(&serde_json::to_vec(&identity).unwrap())
}

fn fixture() -> TempDir {
    let root = TempDir::new().unwrap();
    let centroids = floats(&[1.0, 0.0, 0.0, 1.0]);
    let quality = floats(&[0.2, 0.9, 0.8, 0.3]);
    fs::write(root.path().join("centroids.f32"), &centroids).unwrap();
    fs::write(root.path().join("cluster_quality.f32"), &quality).unwrap();
    let identity = json!({
        "config": {"n_clusters": 2, "seed": 42, "shrinkage": 5.0, "temperature": 0.05, "top_p": 1},
        "embedding": {"dimension": 2, "model_id": "test/embedder", "revision": "v1", "source_repo": "test/source"},
        "expected_cost": [0.1, 0.4],
        "models": ["model-a", "model-b"],
        "provenance": {},
        "schema": "codex-centroid-artifact-v2",
        "tensors": {
            "centroids": {"file": "centroids.f32", "shape": [2, 2], "sha256": sha256(&centroids)},
            "cluster_quality": {"file": "cluster_quality.f32", "shape": [2, 2], "sha256": sha256(&quality)}
        }
    });
    let artifact_id = artifact_id(&identity);
    let mut manifest = identity;
    manifest["artifact_id"] = json!(artifact_id);
    manifest["provenance"] = json!({"trained_at": "2026-08-01T12:00:00Z"});
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    root
}

fn rewrite_artifact_id(root: &TempDir, mutate: impl FnOnce(&mut serde_json::Value)) {
    let manifest_path = root.path().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    mutate(&mut manifest);
    manifest["artifact_id"] = json!(artifact_id(&manifest));
    fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
}

#[test]
fn loads_validated_little_endian_artifact() {
    let root = fixture();

    let artifact = Artifact::load(root.path()).unwrap();

    assert_eq!(artifact.models, ["model-a", "model-b"]);
    assert_eq!(artifact.dimension, 2);
    assert_eq!(artifact.centroids, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(artifact.cluster_quality, [0.2, 0.9, 0.8, 0.3]);
}

#[test]
fn rejects_corrupt_tensor() {
    let root = fixture();
    fs::write(root.path().join("centroids.f32"), [0_u8; 16]).unwrap();

    let error = Artifact::load(root.path()).unwrap_err();

    assert!(error.to_string().contains("SHA-256 mismatch"));
}

#[test]
fn rejects_duplicate_model_before_scoring() {
    let root = fixture();
    let manifest_path = root.path().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["models"] = json!(["model-a", "model-a"]);
    fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let error = Artifact::load(root.path()).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("models must be non-empty and unique")
    );
}

#[test]
fn rejects_unknown_manifest_fields_even_with_matching_identity() {
    let root = fixture();
    rewrite_artifact_id(&root, |manifest| manifest["unknown"] = json!(true));

    let error = Artifact::load(root.path()).unwrap_err();

    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn rejects_non_unit_centroids_even_when_tensor_hash_matches() {
    let root = fixture();
    let centroids = floats(&[2.0, 0.0, 0.0, 1.0]);
    fs::write(root.path().join("centroids.f32"), &centroids).unwrap();
    rewrite_artifact_id(&root, |manifest| {
        manifest["tensors"]["centroids"]["sha256"] = json!(sha256(&centroids));
    });

    let error = Artifact::load(root.path()).unwrap_err();

    assert!(error.to_string().contains("unit length"));
}

#[test]
fn rejects_tensor_shape_overflow() {
    let root = fixture();
    rewrite_artifact_id(&root, |manifest| {
        manifest["config"]["n_clusters"] = json!(usize::MAX);
        manifest["tensors"]["centroids"]["shape"] = json!([usize::MAX, 2]);
        manifest["tensors"]["cluster_quality"]["shape"] = json!([usize::MAX, 2]);
    });

    let error = Artifact::load(root.path()).unwrap_err();

    assert!(error.to_string().contains("shape overflows"));
}
