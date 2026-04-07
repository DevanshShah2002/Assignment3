"""
src/data_prep/generate_teacher_data.py
=======================================
Imitation learning: generate the Stage 2 JSON Instruct dataset by
querying a strong teacher model (Llama-3.1-70B-Instruct via HuggingFace
Inference API or local vLLM endpoint on ARC).

Covers 5 required task types:
  1. JSON extraction from unstructured text
  2. Schema-constrained generation
  3. Exact-label classification with JSON output
  4. JSON repair / formatting correction
  5. Tool-call argument generation

Usage (local):
    python -m src.data_prep.generate_teacher_data --config configs/config.yaml

On ARC (vLLM serving teacher):
    Set teacher_api_base in config.yaml to the vLLM URL.
    Then run via the slurm/generate_teacher_data.slurm script.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from prompts.templates import (
    TEACHER_CLASSIFICATION_PROMPT,
    TEACHER_EXTRACTION_PROMPT,
    TEACHER_REPAIR_PROMPT,
    TEACHER_SCHEMA_GEN_PROMPT,
    TEACHER_TOOL_CALL_PROMPT,
)

random.seed(42)

# ── Parallelism setting ───────────────────────────────────────
MAX_WORKERS = 8  # parallel API calls; reduce to 4 if server rate-limits

# ── Diverse seed data for each task type ─────────────────────

EXTRACTION_SEEDS = [
    {
        "input_text": "John Smith, a 34-year-old software engineer from Austin, Texas, "
                      "joined TechCorp on March 15, 2019. His employee ID is EMP-4892 "
                      "and his annual salary is $98,500.",
        "fields": "name, age, job_title, city, company, join_date, employee_id, salary",
    },
    {
        "input_text": "Invoice #INV-2024-0891 dated November 3rd, 2024. Bill To: Acme Inc., "
                      "123 Main St, Dallas TX 75201. Items: 5x Widget Pro @ $49.99 each, "
                      "2x Gadget Plus @ $129.00 each. Subtotal: $507.95. Tax (8.25%): $41.91. "
                      "Total Due: $549.86. Due Date: November 17, 2024.",
        "fields": "invoice_number, date, company_name, address, items, subtotal, tax_rate, total, due_date",
    },
    {
        "input_text": "Patient: Maria Gonzalez, DOB: 07/22/1981. Admitted: 2024-10-12. "
                      "Diagnosis: Type 2 Diabetes Mellitus (E11.9). Attending physician: Dr. Robert Chen, MD. "
                      "Medications: Metformin 500mg twice daily, Lisinopril 10mg once daily. "
                      "Discharged: 2024-10-14.",
        "fields": "patient_name, date_of_birth, admission_date, diagnosis, diagnosis_code, physician, medications, discharge_date",
    },
    {
        "input_text": "Flight AA2341 departs Dallas/Fort Worth (DFW) at 09:15 AM on December 5, 2024, "
                      "arriving at John F. Kennedy International (JFK) at 1:48 PM. Aircraft: Boeing 737-800. "
                      "Duration: 3h 33min. Class: Economy. Seat: 24C. Gate: B22.",
        "fields": "flight_number, departure_airport, arrival_airport, departure_time, arrival_time, date, aircraft, duration, seat, gate",
    },
    {
        "input_text": "Product: UltraSound X1 Bluetooth Speaker. SKU: SPK-X1-BLK. "
                      "Price: $79.99. Battery: 24-hour playtime, USB-C charging. "
                      "Connectivity: Bluetooth 5.3, AUX. Waterproof: IPX7. Weight: 340g. "
                      "Dimensions: 18 × 7 × 7 cm. Colors available: Black, Navy, Red.",
        "fields": "product_name, sku, price, battery_life, charging_type, bluetooth_version, waterproof_rating, weight_grams, dimensions, colors",
    },
    {
        "input_text": "The conference 'NeurIPS 2024' will be held from December 10-15 at the Vancouver "
                      "Convention Center, Vancouver, Canada. Keynote speakers include Yoshua Bengio "
                      "and Fei-Fei Li. Early registration deadline: September 30. Full registration fee: $1,200.",
        "fields": "conference_name, start_date, end_date, venue, city, country, keynote_speakers, registration_deadline, registration_fee",
    },
    {
        "input_text": "GitHub repository 'awesome-llm-agents' owned by user 'ml-research-lab'. "
                      "Stars: 4,821. Forks: 392. Language: Python (78%), Shell (12%), Dockerfile (10%). "
                      "Last commit: 2024-11-01. License: Apache-2.0. Open issues: 23.",
        "fields": "repo_name, owner, stars, forks, primary_language, last_commit, license, open_issues",
    },
    {
        "input_text": "Restaurant: The Blue Heron. Address: 456 Riverwalk Drive, San Antonio, TX 78205. "
                      "Phone: (210) 555-0142. Hours: Mon-Thu 11am-9pm, Fri-Sat 11am-10pm, Sun Closed. "
                      "Cuisine: American, Seafood. Rating: 4.6/5 (1,203 reviews). "
                      "Price range: $$$. Parking: Valet available.",
        "fields": "name, address, city, state, phone, hours, cuisine_types, rating, review_count, price_range, parking",
    },
]

SCHEMA_GEN_SEEDS = [
    {
        "schema": json.dumps({
            "type": "object",
            "required": ["user_id", "username", "email", "created_at", "role", "is_active"],
            "properties": {
                "user_id": {"type": "integer"},
                "username": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "created_at": {"type": "string", "format": "date-time"},
                "role": {"type": "string", "enum": ["admin", "editor", "viewer"]},
                "is_active": {"type": "boolean"},
            }
        }, indent=2),
        "context": "A recently registered editor user named Alex Thompson from the content management system.",
    },
    {
        "schema": json.dumps({
            "type": "object",
            "required": ["product_id", "name", "price", "category", "in_stock", "tags"],
            "properties": {
                "product_id": {"type": "string", "pattern": "^PRD-[0-9]{6}$"},
                "name": {"type": "string"},
                "price": {"type": "number", "minimum": 0},
                "category": {"type": "string", "enum": ["electronics", "clothing", "books", "home", "sports"]},
                "in_stock": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            }
        }, indent=2),
        "context": "A new electronics product: wireless noise-cancelling headphones at $149.99.",
    },
    {
        "schema": json.dumps({
            "type": "object",
            "required": ["order_id", "customer", "items", "total", "status", "created_at"],
            "properties": {
                "order_id": {"type": "string"},
                "customer": {
                    "type": "object",
                    "required": ["name", "email", "address"],
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "address": {"type": "string"}
                    }
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["product_id", "quantity", "unit_price"],
                        "properties": {
                            "product_id": {"type": "string"},
                            "quantity": {"type": "integer", "minimum": 1},
                            "unit_price": {"type": "number"}
                        }
                    }
                },
                "total": {"type": "number"},
                "status": {"type": "string", "enum": ["pending", "processing", "shipped", "delivered", "cancelled"]},
                "created_at": {"type": "string", "format": "date-time"}
            }
        }, indent=2),
        "context": "An e-commerce order for 3 items placed by Sarah Johnson from Chicago.",
    },
    {
        "schema": json.dumps({
            "type": "object",
            "required": ["experiment_id", "model_name", "hyperparameters", "metrics", "timestamp"],
            "properties": {
                "experiment_id": {"type": "string"},
                "model_name": {"type": "string"},
                "hyperparameters": {
                    "type": "object",
                    "properties": {
                        "learning_rate": {"type": "number"},
                        "batch_size": {"type": "integer"},
                        "epochs": {"type": "integer"},
                        "optimizer": {"type": "string"}
                    }
                },
                "metrics": {
                    "type": "object",
                    "properties": {
                        "train_loss": {"type": "number"},
                        "val_loss": {"type": "number"},
                        "accuracy": {"type": "number"},
                        "f1_score": {"type": "number"}
                    }
                },
                "timestamp": {"type": "string", "format": "date-time"}
            }
        }, indent=2),
        "context": "An ML training run for a text classification model with good results.",
    },
]

CLASSIFICATION_SEEDS = [
    {
        "text": "I absolutely love this product! It exceeded all my expectations and the delivery was super fast. Will definitely order again!",
        "labels": '["positive", "negative", "neutral"]',
    },
    {
        "text": "The service was okay, nothing special. Got what I paid for I guess.",
        "labels": '["positive", "negative", "neutral"]',
    },
    {
        "text": "Breaking: Federal Reserve raises interest rates by 25 basis points amid inflation concerns.",
        "labels": '["politics", "economy", "technology", "sports", "entertainment", "health"]',
    },
    {
        "text": "Scientists discover a potential new treatment for Alzheimer's disease using targeted gene therapy.",
        "labels": '["politics", "economy", "technology", "sports", "entertainment", "health"]',
    },
    {
        "text": "Subject: Urgent - Your account has been compromised! Click here immediately to verify your details and prevent unauthorized access.",
        "labels": '["spam", "phishing", "legitimate", "promotional"]',
    },
    {
        "text": "def calculate_fibonacci(n): return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
        "labels": '["python", "javascript", "java", "c++", "sql", "bash", "other"]',
    },
    {
        "text": "The film is a masterful exploration of grief and memory, elevated by stunning cinematography and career-best performances.",
        "labels": '["very_positive", "positive", "mixed", "negative", "very_negative"]',
    },
    {
        "text": "Please review the attached proposal and provide feedback by end of week.",
        "labels": '["request", "complaint", "inquiry", "notification", "acknowledgment"]',
    },
]

REPAIR_SEEDS = [
    {
        "malformed": """{
  "name": "John Doe",
  "age": 28
  "email": "john@example.com",
  "is_active": true,
  "scores": [92, 87, 95
}""",
    },
    {
        "malformed": """{"product": "Laptop", "price": $999.99, "in_stock": True, "tags": ["electronics" "computers"]}""",
    },
    {
        "malformed": """{
  "user": {
    "id": 1234,
    "name": 'Alice Smith',
    "email": "alice@corp.com"
    "permissions": {
      "read": True,
      "write": false
    },
  },
  "timestamp": "2024-01-15T10:30:00"
}""",
    },
    {
        "malformed": """[{id: 1, "title": "Report Q1", author: "Bob", "published": 2024},
 {"id": 2, "title": "Report Q2, "author": "Carol", "published": 2024}]""",
    },
    {
        "malformed": """{
  "config": {
    "debug": FALSE,
    "max_retries": 3,
    "timeout": None,
    "endpoints": [
      "https://api.example.com/v1"
      "https://api.example.com/v2"
    ]
  }
}""",
    },
    {
        "malformed": """{"employees": [
  {"name": "Tom", "dept": "Engineering", salary: 85000},
  {"name": "Lisa", "dept": "Marketing", "salary": 72000,},
  {"name": "Ray", "dept": "HR" "salary": 68000}
]}""",
    },
]

TOOL_CALL_SEEDS = [
    {
        "signature": "get_weather(city: str, country_code: str, units: str = 'celsius') -> dict",
        "request": "What's the weather like in Tokyo right now? Show me in Fahrenheit.",
    },
    {
        "signature": "send_email(to: list[str], subject: str, body: str, cc: list[str] = None, priority: str = 'normal') -> bool",
        "request": "Send an urgent email to alice@corp.com and bob@corp.com about the Q4 board meeting scheduled for Friday 3pm. Also cc the admin team at admin@corp.com.",
    },
    {
        "signature": "search_database(query: str, table: str, filters: dict = None, limit: int = 10, order_by: str = None) -> list",
        "request": "Find the top 5 customers from California who made purchases over $500 in the orders table, ordered by purchase amount.",
    },
    {
        "signature": "create_calendar_event(title: str, start_datetime: str, end_datetime: str, attendees: list[str], location: str = None, description: str = None, reminder_minutes: int = 15) -> str",
        "request": "Schedule a 90-minute project sync meeting for tomorrow at 2pm with the team: sarah@work.com, mike@work.com, and jen@work.com in Conference Room B. Set a 30-minute reminder.",
    },
    {
        "signature": "translate_text(text: str, source_language: str, target_language: str, formality: str = 'neutral') -> dict",
        "request": "Translate 'Good morning, how can I assist you today?' from English to formal Japanese.",
    },
    {
        "signature": "resize_image(image_path: str, width: int, height: int, maintain_aspect_ratio: bool = True, output_format: str = 'jpg', quality: int = 90) -> str",
        "request": "Resize my profile picture at /uploads/profile.png to 400x400 pixels, keeping the aspect ratio and saving as a high-quality PNG.",
    },
    {
        "signature": "calculate_loan(principal: float, annual_rate: float, term_months: int, payment_frequency: str = 'monthly') -> dict",
        "request": "Calculate the loan details for a $250,000 mortgage at 6.5% annual interest rate over 30 years with monthly payments.",
    },
    {
        "signature": "git_commit(message: str, files: list[str] = None, all_changes: bool = False, author: str = None, amend: bool = False) -> dict",
        "request": "Commit all my changes with the message 'feat: add user authentication module'.",
    },
]


# ── JSON validator ────────────────────────────────────────────

def validate_json(text: str):
    """Try to parse text as JSON. Returns parsed object or None."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── Teacher API client ────────────────────────────────────────

class TeacherClient:
    """Wraps vLLM OpenAI-compatible endpoint. Thread-safe for parallel use."""

    def __init__(self, model: str, api_base=None, token=None):
        self.model = "llama-3.3-70b-instruct-awq"
        from openai import OpenAI
        self.client = OpenAI(
            base_url="http://10.246.100.230/v1",
            api_key=os.getenv("UTSA_API_KEY", "dummy"),
            timeout=60.0,  # fail fast instead of hanging forever
        )

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.3, top_p: float = 0.9) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=max(temperature, 0.01),
                )
                time.sleep(0.3)  # reduced from 2.5s — parallelism handles throughput
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  [Warning] Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return ""


# ── Task generators (all parallelized) ───────────────────────

def build_extraction_examples(client: TeacherClient, cfg: dict, n: int) -> list:
    seeds = (EXTRACTION_SEEDS * (n // len(EXTRACTION_SEEDS) + 1))[:n]
    random.shuffle(seeds)

    def process(seed):
        prompt = TEACHER_EXTRACTION_PROMPT.format(**seed)
        response = client.generate(prompt,
                                   max_new_tokens=cfg["teacher_max_new_tokens"],
                                   temperature=cfg["teacher_temperature"])
        parsed = validate_json(response)
        if parsed is None:
            print("  [Skip] Extraction: invalid JSON")
            return None
        return {
            "task_type": "json_extraction",
            "instruction": f"Extract information from the following text and return it as a JSON object with these fields: {seed['fields']}",
            "input": seed["input_text"],
            "output": json.dumps(parsed, ensure_ascii=False),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return [r for r in ex.map(process, seeds) if r is not None]


def build_schema_gen_examples(client: TeacherClient, cfg: dict, n: int) -> list:
    seeds = (SCHEMA_GEN_SEEDS * (n // len(SCHEMA_GEN_SEEDS) + 1))[:n]
    random.shuffle(seeds)

    def process(seed):
        prompt = TEACHER_SCHEMA_GEN_PROMPT.format(**seed)
        response = client.generate(prompt,
                                   max_new_tokens=cfg["teacher_max_new_tokens"],
                                   temperature=cfg["teacher_temperature"])
        parsed = validate_json(response)
        if parsed is None:
            print("  [Skip] Schema gen: invalid JSON")
            return None
        return {
            "task_type": "schema_constrained_generation",
            "instruction": f"Generate a valid JSON object that strictly conforms to the following schema:\n{seed['schema']}",
            "input": seed["context"],
            "output": json.dumps(parsed, ensure_ascii=False),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return [r for r in ex.map(process, seeds) if r is not None]


def build_classification_examples(client: TeacherClient, cfg: dict, n: int) -> list:
    seeds = (CLASSIFICATION_SEEDS * (n // len(CLASSIFICATION_SEEDS) + 1))[:n]
    random.shuffle(seeds)

    def process(seed):
        prompt = TEACHER_CLASSIFICATION_PROMPT.format(**seed)
        response = client.generate(prompt,
                                   max_new_tokens=256,
                                   temperature=cfg["teacher_temperature"])
        parsed = validate_json(response)
        if parsed is None or "label" not in parsed:
            print("  [Skip] Classification: invalid JSON or missing label")
            return None
        return {
            "task_type": "json_classification",
            "instruction": f"Classify the following text into one of these labels: {seed['labels']}. Return a JSON object with 'label', 'confidence' (0.0–1.0), and 'rationale' fields.",
            "input": seed["text"],
            "output": json.dumps(parsed, ensure_ascii=False),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return [r for r in ex.map(process, seeds) if r is not None]


def build_repair_examples(client: TeacherClient, cfg: dict, n: int) -> list:
    seeds = (REPAIR_SEEDS * (n // len(REPAIR_SEEDS) + 1))[:n]
    random.shuffle(seeds)

    def process(seed):
        prompt = TEACHER_REPAIR_PROMPT.format(malformed_json=seed["malformed"])
        response = client.generate(prompt,
                                   max_new_tokens=cfg["teacher_max_new_tokens"],
                                   temperature=0.1)
        parsed = validate_json(response)
        if parsed is None:
            print("  [Skip] Repair: result still invalid JSON")
            return None
        return {
            "task_type": "json_repair",
            "instruction": "The following JSON is malformed or contains syntax errors. Fix all errors and return valid, properly formatted JSON.",
            "input": seed["malformed"],
            "output": json.dumps(parsed, ensure_ascii=False),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return [r for r in ex.map(process, seeds) if r is not None]


def build_tool_call_examples(client: TeacherClient, cfg: dict, n: int) -> list:
    seeds = (TOOL_CALL_SEEDS * (n // len(TOOL_CALL_SEEDS) + 1))[:n]
    random.shuffle(seeds)

    def process(seed):
        prompt = TEACHER_TOOL_CALL_PROMPT.format(**seed)
        response = client.generate(prompt,
                                   max_new_tokens=512,
                                   temperature=cfg["teacher_temperature"])
        parsed = validate_json(response)
        if parsed is None:
            print("  [Skip] Tool call: invalid JSON")
            return None
        return {
            "task_type": "tool_call_generation",
            "instruction": f"Given the following function signature, generate a JSON object with the named arguments to call the function.\n\nFunction signature:\n{seed['signature']}",
            "input": seed["request"],
            "output": json.dumps(parsed, ensure_ascii=False),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return [r for r in ex.map(process, seeds) if r is not None]


# ── Main ──────────────────────────────────────────────────────

def generate_teacher_data(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dg_cfg = cfg["data_gen"]
    model_cfg = cfg["model"]
    n_per_task = dg_cfg["n_samples_per_task"]

    print(f"[Teacher Gen] Teacher model: {model_cfg['teacher_model']}")
    print(f"[Teacher Gen] Samples per task: {n_per_task} (5 tasks → {5 * n_per_task} total)")
    print(f"[Teacher Gen] Parallel workers: {MAX_WORKERS}")

    client = TeacherClient(
        model=model_cfg["teacher_model"],
        api_base=dg_cfg.get("teacher_api_base"),
        token=os.getenv("HF_TOKEN"),
    )

    # Quick connectivity check before burning time on 1000 samples
    print("[Teacher Gen] Testing API connection...")
    test = client.generate("Reply with this exact JSON: {\"status\": \"ok\"}", max_new_tokens=20)
    if not test:
        print("[Teacher Gen] ERROR: API connection failed — check endpoint and UTSA_API_KEY")
        sys.exit(1)
    print(f"[Teacher Gen] API OK — response: {repr(test)}")

    all_examples = []
    task_builders = [
        ("JSON Extraction",        build_extraction_examples),
        ("Schema-Constrained Gen", build_schema_gen_examples),
        ("JSON Classification",    build_classification_examples),
        ("JSON Repair",            build_repair_examples),
        ("Tool-Call Generation",   build_tool_call_examples),
    ]

    for task_name, builder_fn in task_builders:
        print(f"\n[Teacher Gen] Generating: {task_name} ({n_per_task} samples)...")
        t0 = time.time()
        examples = builder_fn(client, dg_cfg, n_per_task)
        elapsed = time.time() - t0
        print(f"  → Collected {len(examples)} valid examples in {elapsed:.1f}s")
        all_examples.extend(examples)

    random.shuffle(all_examples)
    print(f"\n[Teacher Gen] Total valid examples: {len(all_examples)}")

    # Save full dataset
    os.makedirs("data", exist_ok=True)
    output_path = dg_cfg["output_path"]
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[Teacher Gen] Saved → {output_path}")

    # Build and save eval holdout (20 per task type, never used in training)
    holdout = []
    for task_type in ["json_extraction", "schema_constrained_generation",
                      "json_classification", "json_repair", "tool_call_generation"]:
        task_examples = [e for e in all_examples if e["task_type"] == task_type]
        holdout.extend(task_examples[:20])

    holdout_path = "data/json_eval_holdout.jsonl"
    with open(holdout_path, "w") as f:
        for ex in holdout:
            f.write(json.dumps(ex) + "\n")
    print(f"[Teacher Gen] Saved eval holdout ({len(holdout)} samples) → {holdout_path}")

    return all_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    generate_teacher_data(args.config)