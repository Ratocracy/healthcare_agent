# Healthcare Agent Refactor

## Files

```text
main.py                 # Main pipeline
config.yaml             # Paths, model settings, lab ranges, run settings
streamlit_app.py        # Dashboard for audit/output review
requirements.txt        # Python dependencies
outputs/outbox.json     # Final draft messages after running pipeline
outputs/audit_trail.json# Full decision trail after running pipeline
```

## Setup

Create a `.env` file in this folder:

```text
OPENAI_API_KEY=your_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place your patient case file in this folder or update `paths.input_path` in `config.yaml`.

Expected input format:

```json
[
  {
    "patient_id": "P001",
    "age": 67,
    "sex": "M",
    "fasting": true,
    "symptoms": [],
    "medications": [],
    "conditions": [],
    "labs": {
      "LDL": 165,
      "HDL": 38,
      "A1c": 6.1
    }
  }
]
```

## Run the pipeline

```bash
python main.py --config config.yaml
```

Run one case:

```bash
python main.py --config config.yaml --case-id P001
```

Run without OpenAI calls for testing:

```bash
python main.py --config config.yaml --dry-run
```

Overwrite outputs instead of appending:

```bash
python main.py --config config.yaml --overwrite
```

## Run the dashboard

```bash
streamlit run streamlit_app.py
```

## Audit trail

Each audit trail record captures:

- Patient inputs used by the system
- Rule-based lab interpretations
- Prioritized findings
- Severity classification
- Follow-up questions
- Initial LLM draft
- Safety review result
- Final physician-review draft
- Decision summary

This gives the project a stronger agentic AI architecture: deterministic tool layer, LLM drafting layer, safety review layer, and review dashboard.

## Safety note

This project is designed as a physician-review drafting assistant only. It should not be used to diagnose, prescribe, or provide direct medical advice to patients without clinician review.
