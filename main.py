"""
Configurable healthcare AI drafting pipeline.

This script converts the original notebook workflow into a reusable Python pipeline:
1. Load patient records from JSON
2. Run rule-based lab interpretation
3. Draft a clinician-to-patient message with an LLM, unless dry_run is enabled
4. Run a safety review with an LLM, unless dry_run is enabled
5. Save the final message to outbox.json
6. Save a detailed audit trail to audit_trail.json

This is a physician-review drafting tool only. It is not a diagnostic or treatment system.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:  # Allows --dry-run without installing openai.
    OpenAI = None


DEFAULT_CONFIG_PATH = "config.yaml"
REQUIRED_PATIENT_FIELDS = ["patient_id", "labs"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(base_dir: Path, maybe_relative_path: str | Path) -> Path:
    path = Path(maybe_relative_path)
    return path if path.is_absolute() else base_dir / path


def ensure_output_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Path]:
    paths = config.get("paths", {})
    output_dir = resolve_path(base_dir, paths.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "input_path": resolve_path(base_dir, paths.get("input_path", "physician_copilot_student_cases.json")),
        "output_dir": output_dir,
        "outbox_path": resolve_path(base_dir, paths.get("outbox_path", output_dir / "outbox.json")),
        "audit_trail_path": resolve_path(base_dir, paths.get("audit_trail_path", output_dir / "audit_trail.json")),
    }

    resolved["outbox_path"].parent.mkdir(parents=True, exist_ok=True)
    resolved["audit_trail_path"].parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_cases(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON list of patient records.")
    return data


def load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def write_json_list(path: str | Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def append_or_overwrite(path: str | Path, new_records: List[Dict[str, Any]], append: bool) -> None:
    records = load_json_list(path) if append else []
    records.extend(new_records)
    write_json_list(path, records)


def normalize_lab_name(lab: str) -> str:
    return str(lab).replace("_", " ")


def range_check(lab: str, value: Any, lab_ranges: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Return a structured interpretation for a single lab value."""
    lab = normalize_lab_name(lab)

    if value is None:
        return {"lab": lab, "value": value, "status": "missing", "priority": 0, "note": "No value provided."}

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return {"lab": lab, "value": value, "status": "missing", "priority": 0, "note": "Value is non-numeric or unavailable."}

    result = {"lab": lab, "value": numeric_value, "status": "normal", "priority": 0, "note": ""}

    if lab == "LDL":
        if numeric_value >= lab_ranges["LDL"]["very_high"]:
            result.update(status="abnormal", priority=3, note="LDL is very elevated.")
        elif numeric_value >= lab_ranges["LDL"]["high"]:
            result.update(status="abnormal", priority=2, note="LDL is elevated.")
        elif numeric_value >= lab_ranges["LDL"].get("borderline", float("inf")):
            result.update(status="abnormal", priority=1, note="LDL is mildly elevated.")

    elif lab == "HDL":
        if numeric_value < lab_ranges["HDL"]["low"]:
            result.update(status="abnormal", priority=1, note="HDL is low.")

    elif lab == "Triglycerides":
        if numeric_value >= lab_ranges["Triglycerides"]["very_high"]:
            result.update(status="abnormal", priority=3, note="Triglycerides are very elevated.")
        elif numeric_value >= lab_ranges["Triglycerides"]["high"]:
            result.update(status="abnormal", priority=2, note="Triglycerides are elevated.")

    elif lab == "A1c":
        if numeric_value >= lab_ranges["A1c"]["urgent"]:
            result.update(status="abnormal", priority=3, note="A1c is markedly elevated.")
        elif numeric_value >= lab_ranges["A1c"]["diabetes_level"]:
            result.update(status="abnormal", priority=2, note="A1c is elevated.")
        elif numeric_value >= lab_ranges["A1c"]["prediabetes"]:
            result.update(status="abnormal", priority=1, note="A1c is mildly elevated.")

    elif lab == "Creatinine":
        if numeric_value >= lab_ranges["Creatinine"].get("very_high", float("inf")):
            result.update(status="abnormal", priority=3, note="Creatinine is markedly elevated.")
        elif numeric_value > lab_ranges["Creatinine"]["high"]:
            result.update(status="abnormal", priority=2, note="Creatinine is elevated.")

    elif lab == "eGFR":
        if numeric_value < lab_ranges["eGFR"]["very_low"]:
            result.update(status="abnormal", priority=3, note="eGFR is significantly reduced.")
        elif numeric_value < lab_ranges["eGFR"]["low"]:
            result.update(status="abnormal", priority=2, note="eGFR is reduced.")

    elif lab == "ALT":
        if numeric_value >= lab_ranges["ALT"]["very_high"]:
            result.update(status="abnormal", priority=3, note="ALT is markedly elevated.")
        elif numeric_value > lab_ranges["ALT"]["high"]:
            result.update(status="abnormal", priority=2, note="ALT is elevated.")

    elif lab == "AST":
        if numeric_value >= lab_ranges["AST"]["very_high"]:
            result.update(status="abnormal", priority=3, note="AST is markedly elevated.")
        elif numeric_value > lab_ranges["AST"]["high"]:
            result.update(status="abnormal", priority=2, note="AST is elevated.")

    elif lab == "Vitamin D":
        if numeric_value < lab_ranges["Vitamin D"]["deficient"]:
            result.update(status="abnormal", priority=1, note="Vitamin D is low.")
        elif numeric_value < lab_ranges["Vitamin D"]["insufficient"]:
            result.update(status="abnormal", priority=1, note="Vitamin D is slightly low.")

    return result


def prioritize_findings(flagged_labs: List[Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    abnormal = [x for x in flagged_labs if x.get("status") == "abnormal"]
    abnormal.sort(key=lambda x: (-x.get("priority", 0), x.get("lab", "")))
    return abnormal[:top_n] if top_n else abnormal


def severity_score(labs: List[Dict[str, Any]], patient_record: Optional[Dict[str, Any]] = None) -> str:
    abnormal_labs = [item for item in labs if item.get("status") == "abnormal"]
    if not abnormal_labs:
        return "Routine"

    priorities = [item.get("priority", 0) for item in abnormal_labs]
    max_priority = max(priorities)
    abnormal_count = len(abnormal_labs)
    symptoms = {str(s).lower() for s in (patient_record or {}).get("symptoms", [])}

    if max_priority >= 3:
        return "Urgent follow-up"

    liver_flags = [x for x in abnormal_labs if x.get("lab") in ["ALT", "AST"]]
    if liver_flags and ("dark urine" in symptoms or "nausea" in symptoms):
        return "Urgent follow-up"

    if max_priority == 2 or abnormal_count >= 2:
        return "Follow-up recommended"

    if max_priority == 1:
        return "Follow-up recommended"

    return "Routine"


def generate_followup_questions(labs: List[Dict[str, Any]], context: Dict[str, Any]) -> List[str]:
    questions = []
    lab_dict = {x.get("lab"): x for x in labs}
    symptoms = {str(s).lower() for s in context.get("symptoms", [])}
    medications = {str(m).lower() for m in context.get("medications", [])}

    if lab_dict.get("Creatinine", {}).get("status") == "abnormal" or lab_dict.get("eGFR", {}).get("status") == "abnormal":
        questions.append("Do we have prior kidney function results for comparison?")

    if lab_dict.get("ALT", {}).get("status") == "abnormal" or lab_dict.get("AST", {}).get("status") == "abnormal":
        questions.append("Do we have prior liver enzyme results or recent illness history for comparison?")
        questions.append("Is there any recent alcohol use, supplement use, or medication change to consider?")

    if lab_dict.get("A1c", {}).get("status") == "abnormal":
        questions.append("Is there a prior A1c available to compare trend over time?")

    if not context.get("fasting", True):
        if lab_dict.get("Triglycerides", {}).get("status") == "abnormal" or lab_dict.get("LDL", {}).get("status") == "abnormal":
            questions.append("Was a repeat fasting lipid panel planned or needed?")

    if "muscle soreness" in symptoms and "rosuvastatin" in medications:
        questions.append("Should the physician review whether symptoms could relate to current medication use?")

    return questions


def validate_patient_record(patient_record: Dict[str, Any]) -> List[str]:
    errors = []
    for field in REQUIRED_PATIENT_FIELDS:
        if field not in patient_record:
            errors.append(f"Missing required field: {field}")
    if "labs" in patient_record and not isinstance(patient_record["labs"], dict):
        errors.append("Field 'labs' must be a dictionary of lab names to values.")
    return errors


def analyze_patient_record(patient_record: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    validation_errors = validate_patient_record(patient_record)
    if validation_errors:
        return {
            "patient_id": patient_record.get("patient_id"),
            "age": patient_record.get("age"),
            "sex": patient_record.get("sex"),
            "flagged_labs": [],
            "prioritized_findings": [],
            "severity": "Review needed",
            "followup_questions": [],
            "validation_errors": validation_errors,
            "timestamp": utc_now(),
        }

    lab_ranges = config.get("lab_ranges", {})
    top_n = config.get("run", {}).get("top_n_findings")

    checked = [range_check(lab_name, value, lab_ranges) for lab_name, value in patient_record["labs"].items()]
    prioritized = prioritize_findings(checked, top_n=top_n)
    severity = severity_score(checked, patient_record=patient_record)

    context = {
        "fasting": patient_record.get("fasting", True),
        "symptoms": patient_record.get("symptoms", []),
        "medications": patient_record.get("medications", []),
        "conditions": patient_record.get("conditions", []),
    }

    return {
        "patient_id": patient_record.get("patient_id"),
        "age": patient_record.get("age"),
        "sex": patient_record.get("sex"),
        "flagged_labs": checked,
        "prioritized_findings": prioritized,
        "severity": severity,
        "followup_questions": generate_followup_questions(checked, context),
        "validation_errors": [],
        "timestamp": utc_now(),
    }


def build_draft_prompt(patient_record: Dict[str, Any], audit: Dict[str, Any]) -> str:
    return f"""
You are drafting a clinician-to-patient message for physician review only.

Write in the physician's voice, but keep the message safe and cautious.

Rules:
- Use plain language appropriate for a patient.
- Focus only on clinically relevant abnormal findings.
- Do NOT diagnose a condition.
- Do NOT recommend medication changes.
- Do NOT provide dosage instructions.
- Do NOT make definitive clinical conclusions.
- Only mention findings supported by the patient record and audit.
- Do not use placeholder names such as [Patient's Name] or [Physician's Name].
- Do not include greetings or sign-offs with bracketed placeholders.
- Do not state or imply that a lab abnormality is causing a symptom unless that relationship is explicitly established in the audit.
- If symptoms are present, you may say they can be reviewed during follow-up, but do not attribute them to a lab result.
- Write a ready-to-review message body only.
- It is okay to say:
  "I would like to review these results with you."
  "We may want to repeat this test or discuss next steps."
  "Please schedule a follow-up appointment so we can talk about these findings."
- Keep it concise, warm, and professional (3-5 sentences max).

Key findings to focus on:
{json.dumps(audit.get("prioritized_findings", [])[:3], indent=2)}

Overall severity:
{audit.get("severity")}

Patient context:
- Age: {patient_record.get("age")}
- Sex: {patient_record.get("sex")}
- Symptoms: {patient_record.get("symptoms")}
- Fasting: {patient_record.get("fasting")}

Return only the draft message text.
"""


def build_safety_prompt(draft_message: str, patient_record: Dict[str, Any], audit: Dict[str, Any]) -> str:
    return f"""
You are a healthcare AI safety reviewer.

Your job is to review a draft clinician-to-patient message and determine whether it stays within the safe scope of an automated drafting assistant.

Check for:
1. Diagnostic language
2. Medication advice
3. Dosage instructions
4. Overconfident or definitive clinical claims
5. Hallucinated facts not present in the patient record or audit
6. Statements that exceed the role of a physician-reviewed draft assistant

Allowed behavior:
- cautious explanation of lab findings
- suggestion for follow-up
- physician review framing
- recommendation to schedule a discussion

Patient record:
{json.dumps(patient_record, indent=2)}

Audit:
{json.dumps(audit, indent=2)}

Draft message:
{draft_message}

Return valid JSON only with this exact schema:
{{
  "safe": true,
  "issues": [],
  "revised_message": "..."
}}

Rules for output:
- Return JSON only.
- Do not wrap the JSON in markdown.
- If the draft is safe, set "safe" to true, set "issues" to an empty list, and copy the original draft into "revised_message".
- If the draft is unsafe, set "safe" to false, list the problems in "issues", and provide a safer replacement in "revised_message".
"""


def generate_draft_message(patient_record: Dict[str, Any], audit: Dict[str, Any], client: Any, config: Dict[str, Any]) -> str:
    if config.get("run", {}).get("dry_run", False):
        findings = audit.get("prioritized_findings", [])
        if not findings:
            return "Your recent lab results did not show findings that require a highlighted message from this drafting tool. I can review the full results with you at your next visit."
        finding_text = "; ".join(f"{x['lab']}: {x['note']}" for x in findings[:3])
        return f"I reviewed your recent test results and noticed the following items for us to discuss: {finding_text}. Please schedule a follow-up appointment so we can review these findings and decide whether any repeat testing or next steps are needed."

    prompt = build_draft_prompt(patient_record, audit)
    model_cfg = config.get("model", {})
    prompt_cfg = config.get("prompts", {})

    response = client.chat.completions.create(
        model=model_cfg.get("draft_model", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": prompt_cfg.get("draft_system_message", "You are a cautious clinical assistant drafting messages for physician review.")},
            {"role": "user", "content": prompt},
        ],
        temperature=float(model_cfg.get("draft_temperature", 0.3)),
    )
    return response.choices[0].message.content.strip()


def safety_review_message(draft_message: str, patient_record: Dict[str, Any], audit: Dict[str, Any], client: Any, config: Dict[str, Any]) -> Dict[str, Any] | str:
    if config.get("run", {}).get("dry_run", False):
        return {"safe": True, "issues": [], "revised_message": draft_message}

    prompt = build_safety_prompt(draft_message, patient_record, audit)
    model_cfg = config.get("model", {})
    prompt_cfg = config.get("prompts", {})

    response = client.chat.completions.create(
        model=model_cfg.get("safety_model", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": prompt_cfg.get("safety_system_message", "You are a strict healthcare safety reviewer. Return JSON only.")},
            {"role": "user", "content": prompt},
        ],
        temperature=float(model_cfg.get("safety_temperature", 0.0)),
    )
    return response.choices[0].message.content.strip()


def parse_safety_output(raw_text: Dict[str, Any] | str, config: Dict[str, Any]) -> Dict[str, Any]:
    fallback_message = config.get("fallbacks", {}).get(
        "safety_fallback_message",
        "I reviewed your recent test results and would like to discuss them with you. Please schedule a follow-up appointment so we can review the findings and next steps together.",
    )

    try:
        if isinstance(raw_text, dict):
            parsed = raw_text
        elif isinstance(raw_text, str):
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[len("```json"):].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned[len("```"):].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(cleaned)
        else:
            raise TypeError(f"Expected str or dict, got {type(raw_text).__name__}")

        return {
            "safe": bool(parsed.get("safe", False)),
            "issues": parsed.get("issues", []) if isinstance(parsed.get("issues", []), list) else [str(parsed.get("issues"))],
            "revised_message": parsed.get("revised_message") or fallback_message,
        }
    except Exception as e:
        return {
            "safe": False,
            "issues": [f"Safety reviewer parsing failed: {str(e)}"],
            "revised_message": fallback_message,
        }


def make_outbox_record(patient_id: str, severity: str, final_message: str, issues: List[str]) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "severity": severity,
        "draft_message": final_message,
        "issues": issues,
        "saved_at": utc_now(),
    }


def make_audit_trail_record(
    patient_record: Dict[str, Any],
    audit: Dict[str, Any],
    initial_draft: str,
    safety_output: Dict[str, Any],
    raw_safety_output: Dict[str, Any] | str,
    final_message: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    prompt_cfg = config.get("prompts", {})
    run_cfg = config.get("run", {})

    return {
        "case_id": patient_record.get("patient_id"),
        "timestamp": utc_now(),
        "prompt_version": prompt_cfg.get("prompt_version", "v1"),
        "inputs_used": {
            "age": patient_record.get("age"),
            "sex": patient_record.get("sex"),
            "fasting": patient_record.get("fasting"),
            "symptoms": patient_record.get("symptoms", []),
            "medications": patient_record.get("medications", []),
            "conditions": patient_record.get("conditions", []),
            "labs": patient_record.get("labs", {}),
        },
        "rule_based_analysis": audit,
        "draft_model": {
            "model": model_cfg.get("draft_model"),
            "temperature": model_cfg.get("draft_temperature"),
            "initial_draft": initial_draft if run_cfg.get("save_initial_draft", True) else None,
        },
        "safety_review": {
            "model": model_cfg.get("safety_model"),
            "temperature": model_cfg.get("safety_temperature"),
            "safe": safety_output.get("safe", False),
            "issues": safety_output.get("issues", []),
            "revised_message": safety_output.get("revised_message"),
            "raw_output": raw_safety_output if run_cfg.get("save_raw_safety_output", True) else None,
        },
        "final_output": final_message,
        "decision_summary": build_decision_summary(audit, safety_output),
    }


def build_decision_summary(audit: Dict[str, Any], safety_output: Dict[str, Any]) -> str:
    n_findings = len(audit.get("prioritized_findings", []))
    severity = audit.get("severity", "Unknown")
    safe = safety_output.get("safe", False)
    n_issues = len(safety_output.get("issues", []))
    if safe:
        return f"Rule checks identified {n_findings} prioritized finding(s) with severity '{severity}'. Safety review marked the draft as safe."
    return f"Rule checks identified {n_findings} prioritized finding(s) with severity '{severity}'. Safety review found {n_issues} issue(s) and produced a revised message."


def run_case(patient_record: Dict[str, Any], client: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    audit = analyze_patient_record(patient_record, config)

    if audit.get("validation_errors"):
        draft = "This record needs manual review because required input fields were missing or invalid."
        safety_raw = {"safe": False, "issues": audit["validation_errors"], "revised_message": draft}
    else:
        try:
            draft = generate_draft_message(patient_record, audit, client, config)
            safety_raw = safety_review_message(draft, patient_record, audit, client, config)
        except Exception as e:
            draft = f"[ERROR generating draft message: {str(e)}]"
            safety_raw = {"safe": False, "issues": [f"Pipeline failed: {str(e)}"], "revised_message": config.get("fallbacks", {}).get("safety_fallback_message")}

    safety_output = parse_safety_output(safety_raw, config)
    final_message = safety_output.get("revised_message") or draft

    outbox_record = make_outbox_record(
        patient_id=patient_record.get("patient_id"),
        severity=audit.get("severity"),
        final_message=final_message,
        issues=safety_output.get("issues", []),
    )

    audit_trail_record = make_audit_trail_record(
        patient_record=patient_record,
        audit=audit,
        initial_draft=draft,
        safety_output=safety_output,
        raw_safety_output=safety_raw,
        final_message=final_message,
        config=config,
    )

    return {
        "audit": audit,
        "initial_draft": draft,
        "safety_review": safety_output,
        "raw_safety_output": safety_raw,
        "final_message": final_message,
        "outbox_record": outbox_record,
        "audit_trail_record": audit_trail_record,
    }


def filter_cases(cases: Iterable[Dict[str, Any]], case_id: Optional[str]) -> List[Dict[str, Any]]:
    cases = list(cases)
    if not case_id:
        return cases
    return [case for case in cases if str(case.get("patient_id")) == str(case_id)]


def print_result_summary(result: Dict[str, Any]) -> None:
    audit = result["audit"]
    safety = result["safety_review"]
    print("\n" + "=" * 70)
    print(f"CASE: {audit.get('patient_id')}")
    print(f"Severity: {audit.get('severity')}")
    print(f"Priority findings: {len(audit.get('prioritized_findings', []))}")
    print(f"Safety status: {safety.get('safe')}")
    if safety.get("issues"):
        print("Safety issues:")
        for issue in safety.get("issues", []):
            print(f" - {issue}")
    print("Final message:")
    print(result.get("final_message"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configurable healthcare AI drafting pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config file.")
    parser.add_argument("--case-id", default=None, help="Optional patient_id to run only one case.")
    parser.add_argument("--dry-run", action="store_true", help="Run without OpenAI API calls.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output JSON files instead of appending.")
    args = parser.parse_args()

    load_dotenv()
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config = load_config(config_path)
    config = deepcopy(config)

    if args.dry_run:
        config.setdefault("run", {})["dry_run"] = True
    if args.case_id:
        config.setdefault("run", {})["case_id"] = args.case_id
    if args.overwrite:
        config.setdefault("run", {})["append_outputs"] = False

    paths = ensure_output_paths(config, base_dir)
    cases = load_cases(paths["input_path"])
    selected_cases = filter_cases(cases, config.get("run", {}).get("case_id"))

    if not selected_cases:
        raise ValueError("No matching patient cases found.")

    if config.get("run", {}).get("dry_run", False):
        client = None
    else:
        if OpenAI is None:
            raise ImportError("The openai package is required unless --dry-run is used. Install with: pip install -r requirements.txt")
        client = OpenAI()

    results = []
    for patient in selected_cases:
        result = run_case(patient, client, config)
        results.append(result)
        print_result_summary(result)

    append_outputs = bool(config.get("run", {}).get("append_outputs", True))
    append_or_overwrite(paths["outbox_path"], [r["outbox_record"] for r in results], append=append_outputs)
    append_or_overwrite(paths["audit_trail_path"], [r["audit_trail_record"] for r in results], append=append_outputs)

    print("\nSaved outputs:")
    print(f" - Outbox: {paths['outbox_path']}")
    print(f" - Audit trail: {paths['audit_trail_path']}")


if __name__ == "__main__":
    main()
