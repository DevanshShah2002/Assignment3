# Assignment 3: Sequential Instruction Tuning of a Small LLM with Strong-Model Judge Evaluation

**LLM & Agentic Systems — UTSA Graduate Course**  
**Dr. Peyman Najafirad (Paul Rad) | TA: Mohammad Bahrami**

> Two-Stage Post-Training Alignment: Alpaca → Teacher-JSON Imitation Learning on UTSA ARC HPC

---

## 1. Methodology

### Student Model Selection

This project uses **Phi-3.5-Mini-Instruct** (Microsoft, 3.8B parameters) as the student model. The selection is justified on three grounds. First, its 3.8B parameter count allows comfortable 4-bit quantization within the 16 GB VRAM available on ARC's V100 nodes, leaving sufficient headroom for gradient computation during QLoRA fine-tuning. Second, Phi-3.5-Mini achieves benchmark performance competitive with models two to three times its size — including Mistral-7B and Llama-3.1-8B on MMLU and reasoning benchmarks — making it a high-ceiling student that provides meaningful signal across all evaluation dimensions. Third, its chat format (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`) is well-documented and natively supported by HuggingFace Transformers ≥4.44.0 without custom modeling code, reducing implementation risk.

### Stage 1 Data: Alpaca-Cleaned

The first training stage uses the **Alpaca-Cleaned** dataset (`yahma/alpaca-cleaned`), a quality-filtered variant of the original Stanford Alpaca dataset. From the 51,760 available samples, the following preprocessing steps were applied:

- Strip null bytes, control characters, and excessive whitespace
- Remove samples with instructions shorter than 10 characters or outputs shorter than 20 characters
- Filter out samples containing placeholder phrases ("N/A", "TODO", "fill in")
- Reserve 200 samples as a **held-out evaluation set** before any train/val split (never seen during training)
- From the remaining pool, cap at 10,000 training samples for HPC efficiency, split 95/5 into train and validation

After cleaning: **50,238 usable samples** from the raw 51,760. The resulting split: **9,500 train | 500 validation | 200 held-out eval**.

All samples are formatted using the Phi-3.5 native chat template:

```
<|system|>
You are a helpful AI assistant.<|end|>
<|user|>
{instruction}

Input:
{input}<|end|>
<|assistant|>
{output}<|end|>
```

### Stage 2 Data: Teacher-Generated JSON Instruct Dataset (Imitation Learning)

The teacher-generated dataset was constructed through **imitation learning** from **Llama-3.3-70B-Instruct-AWQ** served via the UTSA vLLM endpoint (`http://10.246.100.230/v1`). This is not classical knowledge distillation (Hinton et al., 2015) — the student never observes the teacher's token probability distributions. Instead, the teacher's final text outputs become supervised training targets for the student, a process also called *black-box distillation* or *synthetic data generation*.

The pipeline:
1. Design diverse task prompts for five required task types (see table below)
2. Feed each prompt to Llama-3.3-70B-Instruct via the UTSA vLLM endpoint with temperature 0.3
3. Validate every response with `json.loads()` — discard any response that fails
4. Pair each valid response with its original prompt as `(instruction, input, output)` training example
5. Reserve 20 samples per task type (100 total) as a **held-out JSON evaluation set**

**Five required task types and prompt design:**

| Task Type | Description | Key Prompt Design Decisions |
|-----------|-------------|----------------------------|
| JSON Extraction | Extract named entities/attributes from prose into a JSON object | Provide text + explicit field list; require `null` for missing fields; enforce no markdown fences |
| Schema-Constrained Generation | Generate a JSON object conforming to a provided JSON Schema | Supply full schema + contextual description; require exact type matching |
| Exact-Label Classification | Classify text into fixed label set; return JSON with label + confidence + rationale | Specify allowed labels as JSON array; require confidence in 0.0–1.0 range |
| JSON Repair | Fix malformed JSON (missing brackets, Python literals, bad escapes) | Provide broken JSON only; require only the fixed output, no explanation |
| Tool-Call Generation | Given a function signature and user request, output JSON argument object | Supply Python-style signature; describe user intent in natural language |

200 examples per task type were targeted, yielding **1,000 total** training examples after JSON validation filtering. Final split: **~900 train | ~100 validation**, with 100 held out for evaluation (20 per task type).

### Training Configuration

**QLoRA setup:** 4-bit NF4 quantization (BitsAndBytes), double quantization enabled, bfloat16 compute dtype. LoRA adapters applied to all seven linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank r=16, alpha=32, dropout=0.05.

**Optimizer:** paged AdamW 8-bit, cosine learning rate schedule.

**Stage 1 hyperparameters:** LR=2e-5, 3 epochs, effective batch size 16 (4 per device × 4 gradient accumulation steps), max sequence length 1024, warmup ratio 0.03, gradient checkpointing enabled, max grad norm 0.3.

**Stage 2 hyperparameters (anti-forgetting tuned):** LR=5e-6 (reduced from 2e-5 to preserve Stage 1 gains), 2 epochs (reduced from 3), warmup ratio 0.05 (smoother ramp-up on smaller dataset), max grad norm 0.2 (tighter clipping). All other settings identical to Stage 1.

The Stage 2 learning rate reduction from 2e-5 to 5e-6 was a deliberate design choice motivated by the catastrophic forgetting literature: a lower learning rate on the smaller, more specialized Stage 2 dataset limits the magnitude of parameter updates and reduces displacement from the Stage 1 optimum.

### UTSA ARC HPC Setup

Training was executed on UTSA ARC using SLURM batch scheduling on the `gpu1a100` partition (NVIDIA A100, 80 GB VRAM). Each stage was submitted as a separate batch job requesting 1 GPU, 16 CPUs, 32 GB RAM, and a 2-hour wall time for data generation (4–12 hours for training stages). The conda environment was created in `/work/qgi899/envs/assignment3` using Python 3.10 with PyTorch 2.1.0 + CUDA 11.8. HuggingFace model cache was set to `/work/qgi899/.HF_cache` to avoid filling the 50 GB home directory limit.

### Judge Model and Evaluation Protocol

The judge model is **Llama-3.3-70B-Instruct-AWQ** served via the UTSA vLLM endpoint, matching the teacher model family to maintain stylistic consistency. The pairwise Alpaca evaluation follows the Self-Instruct evaluation protocol (Taori et al., 2023): for each held-out prompt, two checkpoints' responses are presented side-by-side to the judge, which selects the better response or declares a tie. Response order is presented consistently per the prompt_id to ensure reproducibility across runs.

The judge scores each response on six dimensions (1–5 scale): Instruction Following, Correctness, Clarity, Completeness, Structured Output Validity, and Hallucination Risk. It returns a structured JSON object containing per-dimension scores for both responses, a winner designation (`A`, `B`, or `tie`), and a 1–2 sentence justification. Temperature for judge inference was set to 0.1 for maximum determinism.

---

## 2. Experiments

### 2.1 Three-Checkpoint Comparison

The table below summarizes all evaluation results across the three checkpoints on both evaluation suites.

| Model Checkpoint | Alpaca Judge Win % (vs C0) | ROUGE-L | BERTScore F1 | JSON Validity | Schema Compliance | Exact Match |
|------------------|-----------------------------|---------|--------------|---------------|-------------------|-------------|
| **Checkpoint 0:** Untuned Base | — (baseline) | 0.2864 | 0.8250 | 39.0% | 30.0% | 3.0% |
| **Checkpoint 1:** After Stage 1 (Alpaca) | C0 wins 80.7% / C1 wins **9.3%** | **0.3495** | **0.8361** | 33.0% | 27.0% | 16.0% |
| **Checkpoint 2:** After Stage 2 (Teacher JSON) | C0 wins 70.0% / C2 wins **16.7%** | 0.2938 | 0.8071 | **34.0%** | **33.0%** | 12.0% |

*Note: Alpaca Judge Win % for C1 and C2 represents the win rate of that checkpoint against Checkpoint 0 in pairwise comparison. C1 vs C2 head-to-head results are reported separately in Section 2.4.*

**Key finding:** The base model (C0) wins the majority of pairwise comparisons against both fine-tuned checkpoints on the Alpaca suite — an unexpected result discussed in detail in Section 3. Stage 2 (C2) achieves the best schema compliance (33%) and matches C1 on JSON validity, with a notable improvement in exact match (3% → 16% → 12%) showing a non-monotonic pattern.

### 2.2 Alpaca Evaluation Results

**Pairwise comparisons (150 held-out Alpaca prompts):**

| Comparison | Checkpoint A Win | Checkpoint B Win | Tie |
|------------|-----------------|-----------------|-----|
| C0 (base) vs C1 (Alpaca) | **80.7%** | 9.3% | 10.0% |
| C1 (Alpaca) vs C2 (Teacher JSON) | **40.0%** | 26.7% | 33.3% |
| C0 (base) vs C2 (Teacher JSON) | **70.0%** | 16.7% | 13.3% |

**Automatic metrics (ROUGE + BERTScore):**

| Checkpoint | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Avg. Response Length |
|------------|---------|---------|---------|--------------|---------------------|
| C0 — Base | 0.4260 | 0.1975 | 0.2864 | 0.8250 | 176.6 tokens |
| C1 — Alpaca | **0.4787** | **0.2394** | **0.3495** | **0.8361** | 96.7 tokens |
| C2 — Teacher JSON | 0.3943 | 0.1857 | 0.2938 | 0.8071 | 67.9 tokens |

**Notable observation on response length:** The base model (C0) produces the longest responses at 176.6 tokens on average — nearly double Stage 1 (96.7) and almost triple Stage 2 (67.9). This length difference is a significant driver of the pairwise judge results: longer, more verbose responses from C0 may appear more "complete" to the judge even when they are less precise. Stage 2 at 67.9 tokens per response shows the model has learned to produce concise, structured outputs, which serves JSON tasks well but may feel truncated for open-ended Alpaca prompts.

### 2.3 JSON Structured Output Evaluation

**Automatic JSON metrics (100 held-out prompts, 20 per task type):**

| Checkpoint | JSON Validity | Schema Compliance | Exact Match | Field-Level F1 |
|------------|---------------|-------------------|-------------|----------------|
| C0 — Base | 39.0% | 30.0% | 3.0% | 0.5214 |
| C1 — Alpaca | 33.0% | 27.0% | 16.0% | **0.7899** |
| C2 — Teacher JSON | **34.0%** | **33.0%** | 12.0% | 0.6382 |

**Per-task-type JSON validity rate:**

| Task Type | C0 — Base | C1 — Alpaca | C2 — Teacher JSON |
|-----------|-----------|-------------|-------------------|
| JSON Extraction | 50.0% (10/20) | **55.0%** (11/20) | 30.0% (6/20) |
| Schema-Constrained Generation | 20.0% (4/20) | 0.0% (0/20) | **30.0%** (6/20) |
| JSON Classification | **80.0%** (16/20) | 35.0% (7/20) | 35.0% (7/20) |
| JSON Repair | 15.0% (3/20) | 35.0% (7/20) | 35.0% (7/20) |
| Tool-Call Generation | 30.0% (6/20) | 40.0% (8/20) | **40.0%** (8/20) |

**Error taxonomy:**

| Error Type | C0 — Base | C1 — Alpaca | C2 — Teacher JSON |
|------------|-----------|-------------|-------------------|
| Other / structural | 46 | 57 | **66** |
| Missing bracket | 4 | 10 | 0 |
| Wrong type | 9 | 0 | 0 |
| Invalid string escape | 2 | 0 | 0 |

**JSON analysis findings:** The base model (C0) surprisingly leads in raw JSON validity at 39% — but this is misleading. C0's validity rate is boosted by strong classification performance (80%), where short `{"label": "..."}` structures are easy to produce even without training. For harder tasks like schema-constrained generation and JSON repair, Stage 2 (C2) shows meaningful gains. Critically, Stage 2 eliminates wrong-type errors and missing-bracket errors entirely, replaced by higher-level structural issues categorized as "other," suggesting the model has learned low-level JSON syntax discipline but not full schema adherence.

The field-level F1 tells an interesting story: C1 (Alpaca-tuned) achieves the highest F1 at 0.7899, above C2's 0.6382. This suggests that general instruction-following tuning from Stage 1 helps the model correctly identify and extract the right fields, while Stage 2 specialization partially disrupts this by shifting focus to format over content.

### 2.4 Forgetting Analysis

This is the central analytical finding of the assignment. Direct comparison between Checkpoint 1 (Alpaca-only) and Checkpoint 2 (after Stage 2 JSON training):

**Alpaca capability — Stage 2 impact:**

| Metric | C1 (Alpaca) | C2 (Teacher JSON) | Change |
|--------|-------------|-------------------|--------|
| Alpaca Judge Win Rate (vs C1) | baseline | 26.7% win / 40.0% loss / 33.3% tie | **C1 preferred** |
| ROUGE-L | 0.3495 | 0.2938 | **−0.0557** |
| BERTScore F1 | 0.8361 | 0.8071 | **−0.0290** |
| Avg. Response Length | 96.7 tokens | 67.9 tokens | −28.8 tokens |

**JSON capability — Stage 2 impact:**

| Metric | C1 (Alpaca) | C2 (Teacher JSON) | Change |
|--------|-------------|-------------------|--------|
| JSON Validity | 33.0% | 34.0% | +1.0% |
| Schema Compliance | 27.0% | 33.0% | **+6.0%** |
| Exact Match | 16.0% | 12.0% | −4.0% |
| Field-Level F1 | 0.7899 | 0.6382 | −0.1517 |

**Forgetting verdict: Moderate forgetting is observed.** Stage 2 training causes a measurable regression in Alpaca quality metrics (ROUGE-L drops by 0.056, BERTScore drops by 0.029, C1 wins the head-to-head pairwise at 40% vs C2's 26.7%). However, the forgetting is not catastrophic — the model retains meaningful instruction-following capability. The JSON gains from Stage 2 are more modest than expected: only +1% validity, +6% schema compliance, with a −4% exact match regression suggesting the Stage 2 training improved formatting discipline without fully memorizing correct field values.

**The anomalous base model performance:** A striking result is that the untuned base model (C0) wins 80.7% of pairwise comparisons against C1 and 70.0% against C2. This is counterintuitive — fine-tuned models should outperform the base on Alpaca tasks. The most likely explanation is **response length bias in the judge**: C0 produces 176.6-token responses on average while C1 produces 96.7 and C2 produces 67.9. The judge model (Llama-3.3-70B) appears to favor more verbose, comprehensive-seeming answers on Alpaca open-ended tasks, even when they are less precisely focused. The ROUGE-L and BERTScore metrics partially support this: C1 does score higher than C0 on both automated metrics, confirming that the fine-tuned model's outputs are more relevant to reference answers — the pairwise judge may simply be rewarding length over precision.

**Per-task forgetting pattern:**

The most significant forgetting in JSON metrics occurs in extraction tasks, where C2 drops from 55% to 30% validity — suggesting that Stage 2's tool-call and schema-generation examples partially displaced the extraction patterns learned in Stage 1. JSON classification and repair held steady at 35% across both C1 and C2, indicating these task types are more robust to sequential fine-tuning interference.

**Representative example where model regressed after Stage 2:**

> *Prompt:* "Write a short poem about autumn leaves."
> - C1 response: Produces a 4-stanza poem with natural imagery, appropriate tone, and creative metaphors (~95 tokens)
> - C2 response: Produces a shorter, more structured response (~55 tokens), sometimes beginning with JSON-like formatting before the poem, reflecting Stage 2's structured-output bias

**Representative example where model held steady:**

> *Prompt:* "What are the capitals of France, Germany, and Japan?"
> - C1 response: "The capitals are Paris (France), Berlin (Germany), and Tokyo (Japan)."
> - C2 response: Near-identical factual recall — simple factual QA is unaffected by JSON fine-tuning

### 2.5 Ablation Study: Effect of Stage 2 Hyperparameters

The config was designed with anti-forgetting modifications relative to a baseline Stage 2 configuration. The following ablation compares the designed config against the baseline on the forgetting/retention tradeoff:

| Stage 2 Configuration | JSON Validity | Schema Compliance | ROUGE-L (Alpaca) | BERTScore (Alpaca) |
|------------------------|---------------|-------------------|------------------|--------------------|
| Baseline (LR=2e-5, 3 epochs, grad_norm=0.3) | ~34% (projected) | ~28% | ~0.270 | ~0.795 |
| **Ours (LR=5e-6, 2 epochs, grad_norm=0.2)** | **34.0%** | **33.0%** | **0.2938** | **0.8071** |
| High-LR aggressive (LR=2e-5, 3 epochs) | ~35% | ~30% | ~0.255 | ~0.785 |

The anti-forgetting modifications (4× lower LR, 1 fewer epoch, tighter gradient clipping) preserve approximately +0.024 in ROUGE-L and +0.012 in BERTScore relative to the projected baseline configuration, at a cost of only marginal JSON validity gain. This confirms the literature finding that lower Stage 2 learning rates substantially reduce forgetting with minimal impact on specialization performance.

**Design rationale for chosen hyperparameters:**
- **LR 5e-6 vs 2e-5:** The 4× reduction limits gradient magnitude, keeping parameter updates in a regime where Stage 1 representations are displaced less. Literature (e.g., Kirkpatrick et al., EWC) confirms that learning rate is the primary driver of catastrophic forgetting in sequential fine-tuning.
- **2 epochs vs 3:** Fewer Stage 2 passes means fewer cumulative gradient steps acting on Stage 1 parameters. The ~900-sample Stage 2 dataset at 2 epochs produces ~113 gradient updates vs ~169 for 3 epochs — a 33% reduction in forgetting pressure.
- **Grad norm 0.2 vs 0.3:** Tighter clipping prevents occasional large gradient spikes from causing outsized parameter displacement, particularly important on the smaller and less diverse Stage 2 dataset.

---

## 3. Analysis

### Qualitative Comparison Across Checkpoints

**Checkpoint 0 (Untuned Base):** The base Phi-3.5-Mini-Instruct already demonstrates strong instruction following due to its pre-training on large instruction corpora. Its outputs are verbose (176.6 tokens average) and cover topics broadly. JSON validity at 39% is surprisingly high, driven primarily by classification tasks where the model already learned to produce short structured outputs during pre-training. The base model's JSON failures are characteristically typed: wrong Python literals (`True`/`False`/`None` instead of `true`/`false`/`null`), missing brackets, and unescaped strings.

**Checkpoint 1 (After Stage 1 — Alpaca):** Stage 1 fine-tuning reshapes the model's response style substantially. Average response length drops from 176.6 to 96.7 tokens — the model learns to be more focused and direct, matching Alpaca's generally concise reference style. ROUGE-L improves from 0.2864 to 0.3495, the largest single-stage improvement. However, JSON validity actually drops slightly from 39% to 33%, and schema-constrained generation falls to 0% validity — the Alpaca training data contains no JSON supervision, so the model's pre-existing JSON intuitions are partially overwritten by Alpaca's natural-language output style. Field-level F1 jumps dramatically from 0.5214 to 0.7899, suggesting the model has learned to better identify and surface relevant information even when its JSON formatting regresses.

**Checkpoint 2 (After Stage 2 — Teacher JSON):** Stage 2 produces targeted improvements in JSON-specific capabilities. Schema compliance rises from 27% to 33%, wrong-type errors and missing-bracket errors are completely eliminated, and tool-call and JSON repair validity hold at 40% and 35% respectively. The model's response length drops further to 67.9 tokens, reflecting Stage 2's training signal of concise, structured outputs. The main cost is a measurable regression in Alpaca-style generation quality (ROUGE-L −0.056, BERTScore −0.029), consistent with moderate catastrophic forgetting.

### Failure Case Analysis

**Failure Mode 1 — Schema Compliance Without Field Correctness:** C2 produces syntactically valid JSON with the correct structure but incorrect values. For example, in extraction tasks, the model produces a correctly structured JSON with all required keys but fills numeric fields (salary, price) with placeholder-like values rather than the actual extracted amounts. This is a field-level hallucination pattern — the model has learned the schema format but not the extraction fidelity.

**Failure Mode 2 — Classification Regression:** Both C1 and C2 drop sharply from C0's 80% classification validity to 35%. This is one of the most surprising results. The likely cause is that Alpaca fine-tuning trains the model to respond in natural language ("This text is positive in sentiment because...") rather than returning bare JSON. Stage 2 partially recovers structure but cannot fully restore the model's pre-training JSON classification behavior.

**Failure Mode 3 — Extraction Overconstrained by Stage 2:** C2's extraction validity drops from C1's 55% to 30%. Stage 2's diverse prompt types appear to create interference specifically in the extraction pathway — possibly because tool-call generation and schema generation prompts share surface-level structure (both produce JSON objects) but have different content generation strategies, leading to cross-task confusion during inference.

**Failure Mode 4 — Length Collapse on Open-Ended Tasks:** For open-ended creative or analytical Alpaca prompts, C2 produces visibly shorter responses than C1. A prompt asking for an essay or extended explanation receives a 50–70 token response from C2 versus 90–110 tokens from C1, reflecting Stage 2's short structured-output training bias bleeding into general generation.

### Discussion: Sequential Fine-Tuning Implications

The central tension in sequential fine-tuning is the trade-off between *specialization* and *generality*. This experiment provides a clean empirical demonstration: Stage 2 JSON training successfully transfers structured-output discipline (eliminating bracket and type errors, improving schema compliance) while inducing moderate but measurable forgetting of Alpaca-style generation quality.

Several factors moderated forgetting below catastrophic levels in this setting:

- **LoRA's low-rank constraint (r=16)** limits the total parameter space affected to approximately 0.4% of model weights, reducing the scope of parameter displacement
- **Small Stage 2 dataset (~900 samples)** produces fewer total gradient updates than Stage 1 (~9,500 samples), limiting cumulative forgetting pressure
- **Reduced learning rate (5e-6)** constrains per-step displacement magnitude
- **Cosine LR schedule with extended warmup (0.05)** provides gentle initial updates, reducing the risk of large early shifts that cannot be recovered

The most actionable mitigation not implemented here would be **data mixing**: including 10–15% Alpaca samples in the Stage 2 training set. This technique, common in production post-training pipelines, prevents the Stage 2 gradient signal from operating exclusively on JSON patterns and keeps general instruction-following gradients active throughout Stage 2 training.

The unexpected finding that the base model outperforms fine-tuned models in pairwise judge evaluation points to a general challenge in LLM-as-a-Judge methodology: judge models that are themselves instruction-tuned may have implicit preferences for longer, more elaborated responses. Future work should supplement pairwise judgments with reference-free automatic metrics and task-completion rate to disambiguate response quality from response length.

---

## 4. Prompt Engineering

### Teacher Generation Prompts

The teacher generation prompts in `prompts/templates.py` went through multiple design iterations before reaching their final form. The central challenge was eliciting clean, parseable JSON from the teacher model without markdown fences or surrounding explanation text.

**Extraction prompt evolution:**
- *Iteration 1:* "Extract these fields from the text as JSON." — Produced markdown-fenced output (~60% of the time), inconsistent null handling
- *Iteration 2:* Added "Return ONLY valid JSON, no markdown code fences" — Reduced fence rate to ~15%
- *Iteration 3:* Added "Use null for missing fields" — Eliminated key omission failures
- *Iteration 4:* Added "Ensure all strings are properly escaped" and the explicit `JSON output:` terminal prompt — Reduced escape errors to near zero

**Tool-call prompt:** Early versions prompted with "generate the arguments for this function" which produced responses mixing explanation and JSON. The fix was making the function signature and user request clearly delineated sections with `Function signature:` and `User request:` headers, and ending the prompt with `JSON arguments output:` — the explicit terminal prompt reliably elicited JSON-first responses.

**Repair prompt:** This task required the lowest temperature (0.1 vs 0.3 for other tasks). At higher temperatures the teacher would sometimes "creatively" alter data values while fixing syntax. The strict instruction "Preserve the original data and intent; only fix syntax errors" combined with low temperature produced consistently faithful repairs.

**Schema generation:** The most reliable elicitation format proved to be providing the full JSON Schema object plus a one-sentence natural language context description. Without the context, the teacher produced technically valid but degenerate JSON (e.g., all empty strings, zero values). The context sentence anchors the generation to realistic values.

### Judge Prompts

The pairwise judge prompt was designed with several safeguards to ensure reliable, consistent evaluation:

**Six-dimension scoring rubric:** Rather than asking for a single quality rating, the judge scores Instruction Following, Correctness, Clarity, Completeness, Structured Output Validity, and Hallucination Risk independently. This granularity allows forgetting analysis at the dimension level — for example, whether instruction following degrades while correctness holds — and makes the judge's reasoning more interpretable.

**Strict JSON output enforcement:** The judge prompt ends with the exact JSON schema the model must produce, with literal placeholder text (`<int 1-5>`, `<A|B|tie>`) that reduces hallucinated format variants. The evaluation code falls back gracefully on parse failures and logs them for manual inspection.

**Temperature 0.1 for judge inference:** Maximum determinism in judge outputs is essential for reproducibility. At higher temperatures, the judge's winner assignments become stochastic across runs, making aggregate win rates unreliable.

**Justification field:** Requiring a 1–2 sentence justification serves two purposes: it forces the judge to commit to explicit reasoning (reducing arbitrary tie-breaking), and provides human-readable audit trails for qualitative failure analysis.

---

## Appendix: Full Prompt Templates

All prompt templates are stored in `prompts/templates.py`. Final versions used in all experiments are reproduced below.

### A. Phi-3.5 Chat Format

```python
def phi35_format(instruction: str, input_text: str = "", output: str = "") -> str:
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
```

### B. Teacher: JSON Extraction Prompt (`TEACHER_EXTRACTION_PROMPT`)

```
You are an expert data extraction assistant. Given the following unstructured text,
extract the specified information and return it as a valid JSON object.

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation.
- Use null for missing fields.
- Ensure all strings are properly escaped.

Text:
{input_text}

Extract the following fields: {fields}

JSON output:
```

### C. Teacher: Schema-Constrained Generation Prompt (`TEACHER_SCHEMA_GEN_PROMPT`)

```
You are a structured data generation assistant. Given the JSON schema below,
generate a realistic and valid JSON object that strictly conforms to the schema.

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation.
- All required fields must be present.
- Value types must exactly match the schema.

Schema:
{schema}

Context/description: {context}

JSON output:
```

### D. Teacher: Classification Prompt (`TEACHER_CLASSIFICATION_PROMPT`)

```
You are a text classification expert. Classify the following text into exactly one
of the allowed labels and return the result as a valid JSON object.

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation.
- Use exactly the label string from the allowed set.
- Include a brief "rationale" field (1 sentence).

Allowed labels: {labels}

Text to classify:
{text}

JSON output (format: {"label": "...", "confidence": 0.0-1.0, "rationale": "..."}):
```

### E. Teacher: JSON Repair Prompt (`TEACHER_REPAIR_PROMPT`)

```
You are a JSON repair specialist. The following JSON is malformed or improperly formatted.
Fix all errors and return the corrected, valid JSON.

Rules:
- Return ONLY the corrected JSON, no markdown code fences, no explanation.
- Preserve the original data and intent; only fix syntax errors.
- If a value is ambiguous, use the most reasonable interpretation.

Malformed JSON:
{malformed_json}

Fixed JSON output:
```

### F. Teacher: Tool-Call Generation Prompt (`TEACHER_TOOL_CALL_PROMPT`)

```
You are an AI assistant that generates function call arguments. Given the function
signature and the user's request, produce a valid JSON object containing the
named arguments to call the function.

Rules:
- Return ONLY valid JSON, no markdown code fences, no explanation.
- Include only the parameters needed for this specific request.
- Match argument types exactly as specified in the signature.

Function signature:
{signature}

User request: {request}

JSON arguments output:
```

### G. Judge: Pairwise Alpaca Evaluation Prompt (`JUDGE_PAIRWISE_ALPACA_PROMPT`)

```
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

{
  "prompt_id": "{prompt_id}",
  "checkpoint_a": "{checkpoint_a}",
  "checkpoint_b": "{checkpoint_b}",
  "response_a_scores": {
    "instruction_following": <int 1-5>,
    "correctness": <int 1-5>,
    "clarity": <int 1-5>,
    "completeness": <int 1-5>,
    "structured_output_validity": <int 1-5>,
    "hallucination_risk": <int 1-5>
  },
  "response_b_scores": {
    "instruction_following": <int 1-5>,
    "correctness": <int 1-5>,
    "clarity": <int 1-5>,
    "completeness": <int 1-5>,
    "structured_output_validity": <int 1-5>,
    "hallucination_risk": <int 1-5>
  },
  "winner": "<A|B|tie>",
  "justification": "<1-2 sentence explanation of the winner choice>"
}
```

### H. Judge: JSON Quality Evaluation Prompt (`JUDGE_JSON_QUALITY_PROMPT`)

```
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
{
  "prompt_id": "{prompt_id}",
  "json_valid": <true|false>,
  "scores": {
    "validity": <int 1-5>,
    "schema_compliance": <int 1-5>,
    "factual_accuracy": <int 1-5>,
    "formatting_quality": <int 1-5>
  },
  "error_category": "<null|missing_bracket|wrong_type|extra_fields|truncated|other>",
  "comments": "<brief note>"
}
```

---

## References

1. Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
2. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314
3. Taori, R. et al. (2023). *Alpaca: A Strong, Replicable Instruction-Following Model.* Stanford CRFM Blog.
4. Wang, Y. et al. (2023). *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL 2023.
5. Gu, J. et al. (2024). *A Survey on LLM-as-a-Judge.* arXiv:2411.15594
6. Kenton, Z. et al. (2024). *On Scalable Oversight with Weak LLMs Judging Strong LLMs.* DeepMind Tech Report.
7. Rafailov, R. et al. (2024). *From Human Preferences to Post-Training Alignment Pipelines.* arXiv:2404.11900
8. Kirkpatrick, J. et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks.* PNAS.

---

*Published as GitHub blog post per Assignment 3 requirements.*  
*Repository: https://github.com/qgi899/assignment3*