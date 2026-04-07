# ============================================================
# Prompt Templates — Assignment 3
# All templates are stored here; never hardcode prompts inline.
# ============================================================

# ── Phi-3.5 Mini Chat Format ─────────────────────────────────
PHI35_SYSTEM_TOKEN = "<|system|>"
PHI35_USER_TOKEN   = "<|user|>"
PHI35_ASST_TOKEN   = "<|assistant|>"
PHI35_END_TOKEN    = "<|end|>"

def phi35_format(instruction: str, input_text: str = "", output: str = "") -> str:
    """Format a single example into Phi-3.5 chat template."""
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\nInput:\n{input_text}"
    prompt = (
        f"<|system|>\nYou are a helpful AI assistant.<|end|>\n"
        f"<|user|>\n{user_content}<|end|>\n"
        f"<|assistant|>\n"
    )
    if output:
        prompt += f"{output}<|end|>"
    return prompt


# ── Alpaca Training Format ────────────────────────────────────
ALPACA_PROMPT_TEMPLATE = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_PROMPT_NO_INPUT = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


# ── Teacher Generation Prompts (Imitation Learning) ──────────
# Task 1: JSON Extraction from Unstructured Text
TEACHER_EXTRACTION_PROMPT = """\
You are an expert data extraction assistant. Given the following unstructured text, \
extract the specified information and return it as a valid JSON object.

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation.
- Use null for missing fields.
- Ensure all strings are properly escaped.

Text:
{input_text}

Extract the following fields: {fields}

JSON output:"""

# Task 2: Schema-Constrained Generation
TEACHER_SCHEMA_GEN_PROMPT = """\
You are a structured data generation assistant. Given the JSON schema below, \
generate a realistic and valid JSON object that strictly conforms to the schema.

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation.
- All required fields must be present.
- Value types must exactly match the schema.

Schema:
{schema}

Context/description: {context}

JSON output:"""

# Task 3: Exact-Label Classification with JSON Output
TEACHER_CLASSIFICATION_PROMPT = """\
You are a text classification expert. Classify the following text into exactly one \
of the allowed labels and return the result as a valid JSON object.

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation.
- Use exactly the label string from the allowed set.
- Include a brief "rationale" field (1 sentence).

Allowed labels: {labels}

Text to classify:
{text}

JSON output (format: {{"label": "...", "confidence": 0.0-1.0, "rationale": "..."}}):"""

# Task 4: JSON Repair / Formatting Correction
TEACHER_REPAIR_PROMPT = """\
You are a JSON repair specialist. The following JSON is malformed or improperly formatted. \
Fix all errors and return the corrected, valid JSON.

Rules:
- Return ONLY the corrected JSON, no markdown code fences, no explanation.
- Preserve the original data and intent; only fix syntax errors.
- If a value is ambiguous, use the most reasonable interpretation.

Malformed JSON:
{malformed_json}

Fixed JSON output:"""

# Task 5: Tool-Call Argument Generation
TEACHER_TOOL_CALL_PROMPT = """\
You are an AI assistant that generates function call arguments. Given the function \
signature and the user's request, produce a valid JSON object containing the \
named arguments to call the function.

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation.
- Include only the parameters needed for this specific request.
- Match argument types exactly as specified in the signature.

Function signature:
{signature}

User request: {request}

JSON arguments output:"""


# ── Judge Evaluation Prompts ──────────────────────────────────
JUDGE_PAIRWISE_ALPACA_PROMPT = """\
You are an expert judge evaluating the quality of two AI assistant responses to the same instruction.

Your task: Compare Response A and Response B, and determine which is better according to the rubric below.

=== INSTRUCTION ===
{instruction}

=== RESPONSE A (from {checkpoint_a}) ===
{response_a}

=== RESPONSE B (from {checkpoint_b}) ===
{response_b}

=== EVALUATION RUBRIC ===
Score each response on these dimensions (1-5 scale):
1. Instruction Following: Does the response directly address what was asked?
2. Correctness: Is the information accurate and factually sound?
3. Clarity: Is the response well-written and easy to understand?
4. Completeness: Does the response fully answer the question without unnecessary omission?
5. Structured Output Validity: If structured output is expected, is it valid? (N/A = 3)
6. Hallucination Risk: Does the response avoid making up unsupported facts? (5 = no hallucination)

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object with this exact structure. No markdown, no extra text.

{{
  "prompt_id": "{prompt_id}",
  "checkpoint_a": "{checkpoint_a}",
  "checkpoint_b": "{checkpoint_b}",
  "response_a_scores": {{
    "instruction_following": <int 1-5>,
    "correctness": <int 1-5>,
    "clarity": <int 1-5>,
    "completeness": <int 1-5>,
    "structured_output_validity": <int 1-5>,
    "hallucination_risk": <int 1-5>
  }},
  "response_b_scores": {{
    "instruction_following": <int 1-5>,
    "correctness": <int 1-5>,
    "clarity": <int 1-5>,
    "completeness": <int 1-5>,
    "structured_output_validity": <int 1-5>,
    "hallucination_risk": <int 1-5>
  }},
  "winner": "<A|B|tie>",
  "justification": "<1-2 sentence explanation of the winner choice>"
}}"""

JUDGE_JSON_QUALITY_PROMPT = """\
You are an expert evaluating structured JSON output quality.

=== TASK INSTRUCTION ===
{instruction}

=== MODEL RESPONSE ===
{response}

=== EXPECTED OUTPUT (reference) ===
{expected_output}

Evaluate the response on these dimensions (1-5):
1. JSON Validity: Is the output parseable as valid JSON?
2. Schema Compliance: Are all required fields present with correct types?
3. Factual Accuracy: Are the field values correct/reasonable?
4. Formatting Quality: Is the JSON well-formatted and clean?

Return ONLY a valid JSON object:
{{
  "prompt_id": "{prompt_id}",
  "json_valid": <true|false>,
  "scores": {{
    "validity": <int 1-5>,
    "schema_compliance": <int 1-5>,
    "factual_accuracy": <int 1-5>,
    "formatting_quality": <int 1-5>
  }},
  "error_category": "<null|missing_bracket|wrong_type|extra_fields|truncated|other>",
  "comments": "<brief note>"
}}"""
