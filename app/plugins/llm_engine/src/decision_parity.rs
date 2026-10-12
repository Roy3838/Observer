/// Decision model (OneJev) smoke + parity test: scores System One branches with `decide()` and
/// prints the letter logits and probabilities as JSON, to diff against transformers.js / PyTorch.
///
/// Branches come from the frontend's own prepareDecision (systemOne.ts) so the prompts match
/// byte for byte, e.g. with Node 23+:
///   node dump_branches.mts app/src/utils/localLlm/systemOne.ts '<prompt>' [image.png] > branches.json
///
/// Required env vars:
///   DECIDE_MODEL_PATH    — OneJev GGUF
///   DECIDE_BRANCHES_JSON — { "branches": DecisionBranch[] } as dumped above
/// Optional:
///   DECIDE_MMPROJ_PATH   — vision projector, needed when the branches carry images
///   DECIDE_IMAGE_TOKENS  — vision tokens per image (default 280, OneJev's minimum in the MCP)
///
/// Run:
///   DECIDE_MODEL_PATH=OneJev-0.8B-F16.gguf DECIDE_BRANCHES_JSON=branches.json \
///   cargo test decision_parity -- --nocapture

#[cfg(test)]
mod decision_parity {
    use std::path::PathBuf;
    use crate::{ContextParams, DecisionBranch, LlmEngine};

    #[derive(serde::Deserialize)]
    struct Dump {
        branches: Vec<DecisionBranch>,
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    #[test]
    fn decision_parity() {
        let (Ok(model), Ok(branches_json)) = (
            std::env::var("DECIDE_MODEL_PATH"),
            std::env::var("DECIDE_BRANCHES_JSON"),
        ) else {
            println!("SKIP: set DECIDE_MODEL_PATH and DECIDE_BRANCHES_JSON");
            return;
        };
        let mmproj = std::env::var("DECIDE_MMPROJ_PATH").ok().map(PathBuf::from);
        let image_tokens: i32 = std::env::var("DECIDE_IMAGE_TOKENS").ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(280);

        let dump: Dump = serde_json::from_str(
            &std::fs::read_to_string(&branches_json).expect("branches json unreadable"),
        ).expect("branches json malformed");

        let mut engine = LlmEngine::new().expect("backend init failed");
        engine.set_use_gpu(false);
        engine.set_context_params(ContextParams {
            image_min_tokens: image_tokens,
            image_max_tokens: image_tokens,
            ..ContextParams::default()
        });
        engine.load_model(PathBuf::from(model), "decision-parity".to_string(), mmproj)
            .expect("model load failed");

        let logits = engine.decide(&dump.branches, || false).expect("decide failed");

        assert_eq!(logits.len(), dump.branches.len());
        let report: Vec<serde_json::Value> = logits.iter().zip(&dump.branches).map(|(row, branch)| {
            assert_eq!(row.len(), branch.n_slots);
            assert!(row.iter().all(|v| v.is_finite()), "non-finite logits: {:?}", row);
            serde_json::json!({ "logits": row, "probs": softmax(row) })
        }).collect();

        println!("DECISION_PARITY {}", serde_json::to_string(&report).unwrap());
    }
}
