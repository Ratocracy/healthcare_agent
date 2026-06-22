"""Streamlit dashboard for reviewing healthcare AI pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml


CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: str, base_dir: Path = Path(".")) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def flatten_summary(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        rule_analysis = record.get("rule_based_analysis", {})
        safety = record.get("safety_review", {})
        rows.append(
            {
                "case_id": record.get("case_id"),
                "timestamp": record.get("timestamp"),
                "severity": rule_analysis.get("severity"),
                "num_findings": len(rule_analysis.get("prioritized_findings", [])),
                "safe": safety.get("safe"),
                "num_safety_issues": len(safety.get("issues", [])),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Healthcare AI Audit Dashboard", layout="wide")
st.title("Healthcare AI Audit Dashboard")
st.caption("Review final messages, rule-based findings, safety review results, and audit trail details.")

config = load_config()
paths = config.get("paths", {})
audit_path = resolve_path(paths.get("audit_trail_path", "outputs/audit_trail.json"))
outbox_path = resolve_path(paths.get("outbox_path", "outputs/outbox.json"))

audit_records = load_json_list(audit_path)
outbox_records = load_json_list(outbox_path)

if not audit_records:
    st.warning(f"No audit records found at `{audit_path}`. Run `python main.py --config config.yaml` first.")
    st.stop()

summary_df = flatten_summary(audit_records)

with st.sidebar:
    st.header("Filters")
    severity_options = sorted(x for x in summary_df["severity"].dropna().unique())
    selected_severities = st.multiselect("Severity", options=severity_options, default=severity_options)
    safe_filter = st.selectbox("Safety status", options=["All", "Safe", "Unsafe"])

filtered = summary_df.copy()
if selected_severities:
    filtered = filtered[filtered["severity"].isin(selected_severities)]
if safe_filter == "Safe":
    filtered = filtered[filtered["safe"] == True]
elif safe_filter == "Unsafe":
    filtered = filtered[filtered["safe"] == False]

st.subheader("Run Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cases", len(filtered))
col2.metric("Unsafe", int((filtered["safe"] == False).sum()))
col3.metric("Avg Findings", round(filtered["num_findings"].mean(), 2) if len(filtered) else 0)
col4.metric("Safety Issues", int(filtered["num_safety_issues"].sum()) if len(filtered) else 0)

st.dataframe(filtered, use_container_width=True, hide_index=True)

case_ids = filtered["case_id"].dropna().astype(str).tolist()
if not case_ids:
    st.info("No cases match the selected filters.")
    st.stop()

selected_case_id = st.selectbox("Select a case", options=case_ids)
selected_record = next(r for r in audit_records if str(r.get("case_id")) == selected_case_id)
rule_analysis = selected_record.get("rule_based_analysis", {})
safety_review = selected_record.get("safety_review", {})
inputs = selected_record.get("inputs_used", {})

st.subheader(f"Case {selected_case_id}")

left, right = st.columns([1, 1])

with left:
    st.markdown("### Patient Inputs")
    st.json(inputs)

    st.markdown("### Rule-Based Findings")
    findings = rule_analysis.get("prioritized_findings", [])
    if findings:
        st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
    else:
        st.info("No prioritized abnormal findings.")

    st.markdown("### Follow-up Questions")
    questions = rule_analysis.get("followup_questions", [])
    if questions:
        for q in questions:
            st.write(f"- {q}")
    else:
        st.info("No generated follow-up questions.")

with right:
    st.markdown("### Initial Draft")
    st.write(selected_record.get("draft_model", {}).get("initial_draft") or "Not saved.")

    st.markdown("### Safety Review")
    st.write(f"**Safe:** {safety_review.get('safe')}")
    issues = safety_review.get("issues", [])
    if issues:
        for issue in issues:
            st.error(issue)
    else:
        st.success("No safety issues reported.")

    st.markdown("### Final Output")
    st.write(selected_record.get("final_output", ""))

st.markdown("### Decision Summary")
st.write(selected_record.get("decision_summary", ""))

with st.expander("Raw audit trail JSON"):
    st.json(selected_record)

with st.expander("Outbox records"):
    st.json(outbox_records)
