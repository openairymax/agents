// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! contract.json 的 serde schema。
//!
//! 同时兼容两种形态：
//! 1. `airymax-agent-standard.md` §1.1 的精简字段集
//!    （agent_id/role/name/description/model/tools/handoffs/...）
//! 2. 现有 `airymax_agents/<role>/contract.json` 的完整字段
//!    （schema_version/agent_id/agent_name/version/role/description/capabilities/
//!     models/required_permissions/cost_profile/trust_metrics/extensions）

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid agent_id: {0} (expected ^[a-z][a-z0-9_]*_v\\d+$)")]
    InvalidAgentId(String),
}

pub type Result<T> = std::result::Result<T, ContractError>;

/// Agent 契约根结构。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    #[serde(default)]
    pub schema_version: String,

    /// 全仓唯一 agent_id，格式 `^[a-z][a-z0-9_]*_v\d+$`（如 `architect_v1`）。
    pub agent_id: String,

    /// 人类可读名（Python 侧字段为 `agent_name` 或 `name`，二者皆兼容）。
    #[serde(alias = "agent_name", alias = "name")]
    pub name: String,

    #[serde(default)]
    pub version: String,

    /// 角色，与目录名一致（如 `architect`）。
    pub role: String,

    pub description: String,

    #[serde(default)]
    pub capabilities: Vec<Capability>,

    #[serde(default)]
    pub models: ModelConfig,

    #[serde(default)]
    pub required_permissions: Vec<String>,

    #[serde(default)]
    pub cost_profile: CostProfile,

    #[serde(default)]
    pub trust_metrics: TrustMetrics,

    /// `airymax-agent-standard.md` §1.1 精简字段（可选）。
    #[serde(default)]
    pub model: Option<String>,

    #[serde(default)]
    pub tools: Vec<String>,

    #[serde(default)]
    pub handoffs: Vec<String>,

    /// 自定义扩展字段。
    #[serde(default)]
    pub extensions: HashMap<String, serde_json::Value>,
}

impl Contract {
    /// 从 JSON 字符串解析。
    pub fn from_json(s: &str) -> Result<Self> {
        let c: Contract = serde_json::from_str(s)?;
        c.validate()?;
        Ok(c)
    }

    /// 从文件读取并解析。
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_json(&s)
    }

    /// 校验 agent_id 格式：`^[a-z][a-z0-9_]*_v\d+$`。
    pub fn validate(&self) -> Result<()> {
        if !is_valid_agent_id(&self.agent_id) {
            return Err(ContractError::InvalidAgentId(self.agent_id.clone()));
        }
        Ok(())
    }
}

/// 简化版正则校验：`^[a-z][a-z0-9_]*_v\d+$`。
fn is_valid_agent_id(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_lowercase() {
        return false;
    }
    let mut i = 1;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'_' {
            // 检查后跟 "v" + 数字
            if i + 2 >= bytes.len() || bytes[i + 1] != b'v' || !bytes[i + 2].is_ascii_digit() {
                return false;
            }
            // 剩余必须全是数字
            for &b in &bytes[i + 3..] {
                if !b.is_ascii_digit() {
                    return false;
                }
            }
            return true;
        }
        if !(c.is_ascii_lowercase() || c.is_ascii_digit()) {
            return false;
        }
        i += 1;
    }
    false
}

/// 单个能力声明。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
    #[serde(default)]
    pub estimated_tokens: u32,
    #[serde(default)]
    pub avg_duration_ms: u32,
    #[serde(default)]
    pub success_rate: f32,
}

/// 模型配置。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub system1: String,
    #[serde(default)]
    pub system2: String,
}

/// 成本画像。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostProfile {
    #[serde(default)]
    pub token_per_task_avg: u32,
    #[serde(default)]
    pub api_cost_per_task: f32,
    #[serde(default)]
    pub maintenance_level: String,
}

/// 信任度量。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrustMetrics {
    #[serde(default)]
    pub install_count: u32,
    #[serde(default)]
    pub rating: f32,
    #[serde(default)]
    pub verified_provider: bool,
    #[serde(default)]
    pub last_audit: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_contract() {
        let s = r#"{
            "agent_id": "architect_v1",
            "name": "Architect Agent",
            "role": "architect",
            "description": "test"
        }"#;
        let c = Contract::from_json(s).unwrap();
        assert_eq!(c.agent_id, "architect_v1");
        assert_eq!(c.role, "architect");
    }

    #[test]
    fn parse_full_contract() {
        let s = r#"{
            "schema_version": "1.0.0",
            "agent_id": "product_manager_v1",
            "agent_name": "Product Manager Agent",
            "version": "1.0.0",
            "role": "product_manager",
            "description": "PRD",
            "capabilities": [{"name": "prd_generation"}],
            "models": {"system1": "gpt-4o-mini", "system2": "gpt-4o"},
            "required_permissions": ["read"],
            "cost_profile": {"token_per_task_avg": 6000},
            "trust_metrics": {"install_count": 0}
        }"#;
        let c = Contract::from_json(s).unwrap();
        assert_eq!(c.agent_id, "product_manager_v1");
        assert_eq!(c.name, "Product Manager Agent");
        assert_eq!(c.models.system2, "gpt-4o");
    }

    #[test]
    fn rejects_invalid_agent_id_no_version() {
        let s = r#"{"agent_id":"architect","name":"x","role":"architect","description":"x"}"#;
        let err = Contract::from_json(s).unwrap_err();
        assert!(matches!(err, ContractError::InvalidAgentId(_)));
    }

    #[test]
    fn rejects_invalid_agent_id_uppercase() {
        let s = r#"{"agent_id":"Architect_v1","name":"x","role":"x","description":"x"}"#;
        assert!(Contract::from_json(s).is_err());
    }

    #[test]
    fn agent_id_format_check() {
        assert!(is_valid_agent_id("architect_v1"));
        assert!(is_valid_agent_id("product_manager_v3"));
        assert!(!is_valid_agent_id("architect"));
        assert!(!is_valid_agent_id("Architect_v1"));
        assert!(!is_valid_agent_id("architect_v"));
        assert!(!is_valid_agent_id("architect_v1a"));
        assert!(!is_valid_agent_id(""));
    }
}
