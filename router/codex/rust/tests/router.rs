use routerlab_core::{Artifact, Config, Router};

fn artifact() -> Artifact {
    Artifact {
        artifact_id: "a".repeat(64),
        models: vec!["model-a".into(), "model-b".into(), "model-c".into()],
        dimension: 2,
        centroids: vec![1.0, 0.0, 0.0, 1.0],
        cluster_quality: vec![0.2, 0.9, 0.5, 0.8, 0.3, 0.4],
        expected_cost: vec![0.1, 0.4, 0.2],
        config: Config {
            n_clusters: 2,
            top_p: 1,
            shrinkage: 5.0,
            temperature: 0.05,
            seed: 42,
        },
        embedding_model: "test/embedder".into(),
        embedding_source_repo: "test/source".into(),
        embedding_revision: "v1".into(),
    }
}

#[test]
fn routes_quality_and_cost_endpoints_with_manifest_ties() {
    let router = Router::new(artifact()).unwrap();

    let quality = router.route_embedding(&[1.0, 0.0], 1.0, None).unwrap();
    let cost = router.route_embedding(&[1.0, 0.0], 0.0, None).unwrap();

    assert_eq!(quality.model, "model-b");
    assert_eq!(quality.artifact_id, "a".repeat(64));
    assert_eq!(quality.cluster_ids, [0]);
    assert_eq!(cost.model, "model-a");
}

#[test]
fn eligibility_filters_unknown_and_unavailable_models() {
    let router = Router::new(artifact()).unwrap();

    let decision = router
        .route_embedding(
            &[1.0, 0.0],
            1.0,
            Some(&["model-a".into(), "model-c".into()]),
        )
        .unwrap();
    let error = router
        .route_embedding(&[1.0, 0.0], 1.0, Some(&["unknown".into()]))
        .unwrap_err();

    assert_eq!(decision.model, "model-c");
    assert!(error.to_string().contains("unknown eligible model"));
}

#[test]
fn rejects_invalid_embedding_and_bias() {
    let router = Router::new(artifact()).unwrap();

    assert!(router.route_embedding(&[1.0], 0.5, None).is_err());
    assert!(router.route_embedding(&[1.0, 0.0], 1.1, None).is_err());
}

#[test]
fn rejects_artifact_dimension_overflow() {
    let mut malformed = artifact();
    malformed.config.n_clusters = usize::MAX;
    malformed.dimension = 2;

    let error = Router::new(malformed).unwrap_err();

    assert!(error.to_string().contains("overflow"));
}

#[test]
fn rejects_directly_constructed_invalid_artifacts() {
    let mut invalid_cost = artifact();
    invalid_cost.expected_cost[0] = f64::NAN;
    assert!(Router::new(invalid_cost).is_err());

    let mut invalid_centroid = artifact();
    invalid_centroid.centroids[0] = 2.0;
    assert!(Router::new(invalid_centroid).is_err());

    let mut invalid_config = artifact();
    invalid_config.config.temperature = 0.0;
    assert!(Router::new(invalid_config).is_err());

    let mut duplicate_models = artifact();
    duplicate_models.models[1] = duplicate_models.models[0].clone();
    assert!(Router::new(duplicate_models).is_err());
}
