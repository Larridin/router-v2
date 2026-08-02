use std::cmp::Ordering;
use std::collections::HashSet;

use serde::Serialize;
use thiserror::Error;

use crate::Artifact;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Decision {
    pub artifact_id: String,
    pub model: String,
    pub model_index: usize,
    pub utility: f64,
    pub predicted_quality: f64,
    pub expected_cost: f64,
    pub cluster_ids: Vec<usize>,
    pub cluster_weights: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum RouteError {
    #[error("invalid router artifact: {0}")]
    Artifact(String),
    #[error("invalid route request: {0}")]
    Request(String),
}

#[derive(Clone, Debug)]
pub struct Router {
    artifact: Artifact,
}

impl Router {
    pub fn new(artifact: Artifact) -> Result<Self, RouteError> {
        let models = artifact.models.len();
        let clusters = artifact.config.n_clusters;
        let expected_centroids = clusters.checked_mul(artifact.dimension).ok_or_else(|| {
            RouteError::Artifact("centroid dimensions overflow address space".to_owned())
        })?;
        let expected_quality = clusters.checked_mul(models).ok_or_else(|| {
            RouteError::Artifact("quality dimensions overflow address space".to_owned())
        })?;
        let unique_models: HashSet<&str> = artifact.models.iter().map(String::as_str).collect();
        if !is_sha256(&artifact.artifact_id)
            || models == 0
            || unique_models.len() != models
            || artifact.models.iter().any(String::is_empty)
            || artifact.dimension == 0
            || artifact.centroids.len() != expected_centroids
            || artifact.cluster_quality.len() != expected_quality
            || artifact.expected_cost.len() != models
            || artifact.config.top_p == 0
            || artifact.config.top_p > clusters
            || !artifact.config.shrinkage.is_finite()
            || artifact.config.shrinkage <= 0.0
            || !artifact.config.temperature.is_finite()
            || artifact.config.temperature <= 0.0
            || artifact.embedding_model.is_empty()
            || artifact.embedding_source_repo.is_empty()
            || artifact.embedding_revision.is_empty()
            || artifact.centroids.iter().any(|value| !value.is_finite())
            || artifact
                .cluster_quality
                .iter()
                .any(|value| !value.is_finite())
            || artifact
                .expected_cost
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(RouteError::Artifact(
                "tensor dimensions or configuration do not align".to_owned(),
            ));
        }
        if artifact
            .centroids
            .chunks_exact(artifact.dimension)
            .any(|row| {
                let norm = row
                    .iter()
                    .map(|value| f64::from(*value).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (norm - 1.0).abs() > 1e-5
            })
        {
            return Err(RouteError::Artifact(
                "centroid rows must have unit length".to_owned(),
            ));
        }
        Ok(Self { artifact })
    }

    pub fn route_embedding(
        &self,
        embedding: &[f32],
        quality_bias: f64,
        eligible_models: Option<&[String]>,
    ) -> Result<Decision, RouteError> {
        if !quality_bias.is_finite() || !(0.0..=1.0).contains(&quality_bias) {
            return request("quality_bias must be finite and in [0, 1]");
        }
        if embedding.len() != self.artifact.dimension
            || embedding.iter().any(|value| !value.is_finite())
        {
            return request("embedding dimension or values are invalid");
        }
        let norm = embedding
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm <= 0.0 {
            return request("embedding must have non-zero norm");
        }
        let normalized: Vec<f64> = embedding
            .iter()
            .map(|value| f64::from(*value) / norm)
            .collect();
        let dimension = self.artifact.dimension;
        let mut similarities: Vec<(usize, f64)> = self
            .artifact
            .centroids
            .chunks_exact(dimension)
            .enumerate()
            .map(|(index, centroid)| {
                let similarity = centroid
                    .iter()
                    .zip(&normalized)
                    .map(|(left, right)| f64::from(*left) * right)
                    .sum();
                (index, similarity)
            })
            .collect();
        similarities.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        similarities.truncate(self.artifact.config.top_p);
        let maximum = similarities
            .iter()
            .map(|(_, value)| *value / self.artifact.config.temperature)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = similarities
            .iter()
            .map(|(_, value)| (*value / self.artifact.config.temperature - maximum).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= total;
        }

        let model_count = self.artifact.models.len();
        let mut predicted = vec![0.0_f64; model_count];
        for ((cluster, _), weight) in similarities.iter().zip(&weights) {
            for (model, value) in predicted.iter_mut().enumerate() {
                *value += weight
                    * f64::from(self.artifact.cluster_quality[cluster * model_count + model]);
            }
        }
        let eligible = self.eligible_indices(eligible_models)?;
        let quality_values: Vec<f64> = eligible.iter().map(|index| predicted[*index]).collect();
        let cost_values: Vec<f64> = eligible
            .iter()
            .map(|index| self.artifact.expected_cost[*index])
            .collect();
        let quality_scores = minmax(&quality_values);
        let cost_scores = minmax(&cost_values);
        let mut winner_local = 0;
        let mut winner_utility = f64::NEG_INFINITY;
        for local in 0..eligible.len() {
            let utility = quality_bias * quality_scores[local]
                + (1.0 - quality_bias) * (1.0 - cost_scores[local]);
            if utility > winner_utility {
                winner_utility = utility;
                winner_local = local;
            }
        }
        let winner = eligible[winner_local];
        Ok(Decision {
            artifact_id: self.artifact.artifact_id.clone(),
            model: self.artifact.models[winner].clone(),
            model_index: winner,
            utility: winner_utility,
            predicted_quality: predicted[winner],
            expected_cost: self.artifact.expected_cost[winner],
            cluster_ids: similarities.iter().map(|(index, _)| *index).collect(),
            cluster_weights: weights,
        })
    }

    fn eligible_indices(
        &self,
        eligible_models: Option<&[String]>,
    ) -> Result<Vec<usize>, RouteError> {
        let Some(requested) = eligible_models else {
            return Ok((0..self.artifact.models.len()).collect());
        };
        if requested.is_empty() {
            return request("eligible_models must not be empty");
        }
        let requested_set: HashSet<&str> = requested.iter().map(String::as_str).collect();
        if requested_set.len() != requested.len() {
            return request("eligible_models contains duplicates");
        }
        for model in &requested_set {
            if !self.artifact.models.iter().any(|known| known == model) {
                return request(format!("unknown eligible model: {model}"));
            }
        }
        Ok(self
            .artifact
            .models
            .iter()
            .enumerate()
            .filter_map(|(index, model)| requested_set.contains(model.as_str()).then_some(index))
            .collect())
    }
}

fn minmax(values: &[f64]) -> Vec<f64> {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = maximum - minimum;
    if span <= 1e-12 {
        return vec![0.0; values.len()];
    }
    values
        .iter()
        .map(|value| (value - minimum) / span)
        .collect()
}

fn request<T>(message: impl Into<String>) -> Result<T, RouteError> {
    Err(RouteError::Request(message.into()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
