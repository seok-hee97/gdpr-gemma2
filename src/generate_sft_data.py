"""
Generate synthetic GDPR Q&A training data using Upstage Solar API.

Produces instruction/input/output triples in the same format as
`sims2k/GDPR_QA_instruct_dataset`, covering GDPR articles and topics
that are under-represented in the original 316-sample dataset.

Usage:
    python -m src.generate_sft_data                        # default: 101 topics × 30 = 3,030 samples
    python -m src.generate_sft_data --num_per_topic 5      # smaller test run (505 samples)
    python -m src.generate_sft_data --model solar-mini     # cheaper model

Features:
    - Incremental saving: results are appended to output file every batch, safe against crashes
    - Auto-retry on rate limit (429) with exponential backoff
    - Resume support: skips already-generated samples if output file exists (use --fresh to overwrite)
"""
import argparse
import json
import os
import random
import time
from openai import OpenAI
from tqdm import tqdm
from . import config

# ────────────────────────────────────────────────────────────────────
# Topic definitions: GDPR chapters × question types
# ────────────────────────────────────────────────────────────────────

GDPR_TOPICS = [
    # Chapter I — Scope & Definitions
    {"chapter": "I", "theme": "territorial scope of the GDPR (Article 3)", "articles": "3"},
    {"chapter": "I", "theme": "definition of 'personal data' vs 'anonymous data' (Article 4)", "articles": "4"},
    {"chapter": "I", "theme": "material scope — when does the GDPR apply and what are the exemptions (Article 2)", "articles": "2"},
    # Chapter II — Principles
    {"chapter": "II", "theme": "lawfulness, fairness, and transparency principle (Article 5(1)(a))", "articles": "5"},
    {"chapter": "II", "theme": "purpose limitation principle (Article 5(1)(b))", "articles": "5"},
    {"chapter": "II", "theme": "data minimization principle (Article 5(1)(c))", "articles": "5"},
    {"chapter": "II", "theme": "conditions for valid consent (Article 7)", "articles": "7"},
    {"chapter": "II", "theme": "child's consent in relation to information society services (Article 8)", "articles": "8"},
    {"chapter": "II", "theme": "processing of special categories of personal data (Article 9)", "articles": "9"},
    {"chapter": "II", "theme": "lawful bases for processing — consent vs legitimate interest vs contract (Article 6)", "articles": "6"},
    # Chapter III — Data Subject Rights
    {"chapter": "III", "theme": "right of access by the data subject (Article 15)", "articles": "15"},
    {"chapter": "III", "theme": "right to rectification (Article 16)", "articles": "16"},
    {"chapter": "III", "theme": "right to erasure / right to be forgotten (Article 17)", "articles": "17"},
    {"chapter": "III", "theme": "right to restriction of processing (Article 18)", "articles": "18"},
    {"chapter": "III", "theme": "right to data portability (Article 20)", "articles": "20"},
    {"chapter": "III", "theme": "right to object to processing (Article 21)", "articles": "21"},
    {"chapter": "III", "theme": "automated individual decision-making including profiling (Article 22)", "articles": "22"},
    {"chapter": "III", "theme": "transparent information and communication (Articles 12-14)", "articles": "12,13,14"},
    {"chapter": "III", "theme": "restrictions on data subject rights (Article 23)", "articles": "23"},
    # Chapter IV — Controller & Processor
    {"chapter": "IV", "theme": "responsibility of the controller (Article 24)", "articles": "24"},
    {"chapter": "IV", "theme": "data protection by design and by default (Article 25)", "articles": "25"},
    {"chapter": "IV", "theme": "joint controllers (Article 26)", "articles": "26"},
    {"chapter": "IV", "theme": "processor obligations and contract requirements (Article 28)", "articles": "28"},
    {"chapter": "IV", "theme": "records of processing activities (Article 30)", "articles": "30"},
    {"chapter": "IV", "theme": "security of processing (Article 32)", "articles": "32"},
    {"chapter": "IV", "theme": "notification of a personal data breach (Article 33)", "articles": "33"},
    {"chapter": "IV", "theme": "communication of a breach to the data subject (Article 34)", "articles": "34"},
    {"chapter": "IV", "theme": "data protection impact assessment (Article 35)", "articles": "35"},
    {"chapter": "IV", "theme": "prior consultation with the supervisory authority (Article 36)", "articles": "36"},
    {"chapter": "IV", "theme": "designation and role of the Data Protection Officer (Articles 37-39)", "articles": "37,38,39"},
    {"chapter": "IV", "theme": "codes of conduct and certification mechanisms (Articles 40-43)", "articles": "40,42,43"},
    # Chapter V — International Transfers
    {"chapter": "V", "theme": "general principle for transfers of personal data to third countries (Article 44)", "articles": "44"},
    {"chapter": "V", "theme": "adequacy decisions by the European Commission (Article 45)", "articles": "45"},
    {"chapter": "V", "theme": "standard contractual clauses (SCCs) as transfer safeguard (Article 46)", "articles": "46"},
    {"chapter": "V", "theme": "binding corporate rules (Article 47)", "articles": "47"},
    {"chapter": "V", "theme": "derogations for specific situations — Article 49 transfers", "articles": "49"},
    # Chapter VI — Supervisory Authorities
    {"chapter": "VI", "theme": "independence and competence of the supervisory authority (Articles 51-55)", "articles": "51,52,55"},
    {"chapter": "VI", "theme": "tasks and powers of the supervisory authority (Articles 57-58)", "articles": "57,58"},
    # Chapter VIII — Remedies & Penalties
    {"chapter": "VIII", "theme": "right to lodge a complaint with a supervisory authority (Article 77)", "articles": "77"},
    {"chapter": "VIII", "theme": "right to an effective judicial remedy (Articles 78-79)", "articles": "78,79"},
    {"chapter": "VIII", "theme": "right to compensation and liability (Article 82)", "articles": "82"},
    {"chapter": "VIII", "theme": "conditions for imposing administrative fines (Article 83)", "articles": "83"},
    {"chapter": "VIII", "theme": "penalties — up to €20M or 4% of global turnover (Article 84)", "articles": "83,84"},
    # Chapter IX — Specific Processing Situations
    {"chapter": "IX", "theme": "processing and freedom of expression (Article 85)", "articles": "85"},
    {"chapter": "IX", "theme": "processing in the employment context (Article 88)", "articles": "88"},
    {"chapter": "IX", "theme": "safeguards for processing for research purposes (Article 89)", "articles": "89"},
    # Cross-cutting Scenarios
    {"chapter": "SCENARIO", "theme": "a startup collecting user emails for marketing — lawful basis, consent, and transparency obligations", "articles": "6,7,12,13"},
    {"chapter": "SCENARIO", "theme": "a hospital processing patient health records — special categories, security, and breach notification", "articles": "9,32,33,34"},
    {"chapter": "SCENARIO", "theme": "an e-commerce company transferring customer data to a US cloud provider — international transfer safeguards", "articles": "44,46,49"},
    {"chapter": "SCENARIO", "theme": "a social media platform processing children's data for targeted advertising — child consent and profiling", "articles": "8,22,6"},
    {"chapter": "SCENARIO", "theme": "a data subject requests erasure of their data from a search engine — right to be forgotten and balancing test", "articles": "17,21"},
    {"chapter": "SCENARIO", "theme": "a company discovers a data breach affecting 10,000 users — breach notification timeline and communication duties", "articles": "33,34"},
    {"chapter": "SCENARIO", "theme": "an employer monitoring employee emails — lawful basis, proportionality, and the employment processing exception", "articles": "6,88"},
    {"chapter": "SCENARIO", "theme": "a bank using automated credit scoring to reject a loan application — profiling, explanation, and human intervention", "articles": "22,15"},
    {"chapter": "SCENARIO", "theme": "a university conducting research with anonymized student data — research safeguards and pseudonymization", "articles": "89,4"},
    {"chapter": "SCENARIO", "theme": "a multinational appointing a DPO and determining the lead supervisory authority for cross-border processing", "articles": "37,56"},
    # ── Sector-specific Scenarios ──
    {"chapter": "SECTOR", "theme": "a hospital deploying an AI diagnostic tool that processes patient health records — special categories, DPIA, and transparency", "articles": "9,35,13,22"},
    {"chapter": "SECTOR", "theme": "a fintech startup processing credit scores and bank transaction data for loan decisions — profiling, automated decision-making, and lawful basis", "articles": "6,9,22,15"},
    {"chapter": "SECTOR", "theme": "an HR department using employee monitoring software to track productivity — lawful basis, proportionality, and employee consent", "articles": "6,88,5,13"},
    {"chapter": "SECTOR", "theme": "a university collecting student performance data for research — research exemption, pseudonymization, and purpose limitation", "articles": "89,5,4,9"},
    {"chapter": "SECTOR", "theme": "a telecom provider retaining call metadata for law enforcement requests — data retention, legal obligation, and storage limitation", "articles": "6,5,23"},
    {"chapter": "SECTOR", "theme": "an insurance company processing health data to calculate premiums — special categories, explicit consent, and automated profiling", "articles": "9,7,22"},
    {"chapter": "SECTOR", "theme": "a SaaS company acting as data processor for enterprise clients — processor obligations, sub-processors, and data processing agreements", "articles": "28,29,82"},
    {"chapter": "SECTOR", "theme": "an IoT smart home device manufacturer collecting user behavior data — data minimization, privacy by design, and transparency", "articles": "5,25,12,13"},
    {"chapter": "SECTOR", "theme": "an online gaming platform verifying age for child protection — children's consent, parental verification, and information society services", "articles": "8,12,6"},
    {"chapter": "SECTOR", "theme": "a real estate agency sharing tenant screening data with landlords — purpose limitation, data sharing, and legitimate interest assessment", "articles": "5,6,14"},
    {"chapter": "SECTOR", "theme": "a non-profit organization processing donor personal data for fundraising campaigns — consent management, right to object, and direct marketing", "articles": "6,7,21"},
    {"chapter": "SECTOR", "theme": "a retail chain using facial recognition cameras for loss prevention — biometric data, DPIA, and proportionality", "articles": "9,35,5"},
    # ── Advanced Concepts ──
    {"chapter": "ADVANCED", "theme": "sub-processor chains — obligations when a processor engages further sub-processors under Article 28(2) and (4)", "articles": "28"},
    {"chapter": "ADVANCED", "theme": "joint controller disputes — determining responsibilities and liability when multiple controllers jointly determine processing purposes (Article 26)", "articles": "26,82"},
    {"chapter": "ADVANCED", "theme": "the BCR (Binding Corporate Rules) approval process — requirements, content, and supervisory authority coordination (Article 47)", "articles": "47,63"},
    {"chapter": "ADVANCED", "theme": "the one-stop-shop mechanism for cross-border processing — identifying the lead supervisory authority (Articles 56, 60)", "articles": "56,60"},
    {"chapter": "ADVANCED", "theme": "the consistency mechanism and the role of the EDPB in ensuring uniform GDPR application (Articles 63-65)", "articles": "63,64,65"},
    {"chapter": "ADVANCED", "theme": "data protection certification mechanisms — ISO 27701, approved certification bodies, and the role of accreditation (Articles 42-43)", "articles": "42,43"},
    {"chapter": "ADVANCED", "theme": "accountability principle in practice — demonstrating compliance through documentation, audits, and governance (Article 5(2) and 24)", "articles": "5,24"},
    {"chapter": "ADVANCED", "theme": "legitimate interest balancing test — three-part assessment (purpose, necessity, balance) and documentation requirements under Article 6(1)(f)", "articles": "6"},
    # ── Special Data Types ──
    {"chapter": "SPECIAL_DATA", "theme": "processing genetic data under GDPR — definition, special category rules, and member state derogations (Articles 4(13), 9)", "articles": "4,9"},
    {"chapter": "SPECIAL_DATA", "theme": "biometric data for identification purposes — when biometric processing triggers Article 9 and DPIA requirements", "articles": "9,4,35"},
    {"chapter": "SPECIAL_DATA", "theme": "processing children's data in educational settings — age verification, parental consent, and child-friendly privacy notices (Article 8)", "articles": "8,12,13"},
    {"chapter": "SPECIAL_DATA", "theme": "processing data of deceased persons — GDPR scope limitations and member state rules (Recital 27)", "articles": "1,2"},
    {"chapter": "SPECIAL_DATA", "theme": "health data in clinical research — lawful basis, explicit consent vs public interest, and ethics committee oversight (Articles 9, 89)", "articles": "9,89"},
    {"chapter": "SPECIAL_DATA", "theme": "criminal conviction data — processing restrictions and the distinction from special categories under Article 10", "articles": "10,9"},
    # ── Practical Workflows ──
    {"chapter": "WORKFLOW", "theme": "step-by-step guide to conducting a Data Protection Impact Assessment (DPIA) — when required, methodology, and supervisory authority consultation", "articles": "35,36"},
    {"chapter": "WORKFLOW", "theme": "72-hour breach notification timeline — detection, assessment, authority notification, and data subject communication workflow", "articles": "33,34"},
    {"chapter": "WORKFLOW", "theme": "building a Record of Processing Activities (ROPA) — required fields, format, and maintenance best practices", "articles": "30"},
    {"chapter": "WORKFLOW", "theme": "implementing a consent management system — granularity, withdrawal mechanism, record-keeping, and cookie consent under ePrivacy", "articles": "7,6"},
    {"chapter": "WORKFLOW", "theme": "drafting a GDPR-compliant privacy notice — required information under Articles 13 and 14, layered approach, and plain language", "articles": "12,13,14"},
    {"chapter": "WORKFLOW", "theme": "conducting a third-party vendor GDPR assessment — due diligence checklist, DPA requirements, and ongoing monitoring", "articles": "28,32"},
    {"chapter": "WORKFLOW", "theme": "handling a data subject access request (DSAR) — identity verification, response timeline, exemptions, and format requirements", "articles": "15,12"},
    # ── Enforcement & Case Law ──
    {"chapter": "ENFORCEMENT", "theme": "analysis of major GDPR fines — Meta, Google, Amazon cases and the criteria supervisory authorities use under Article 83(2)", "articles": "83"},
    {"chapter": "ENFORCEMENT", "theme": "GDPR enforcement trends — most commonly violated articles, sectors with highest fines, and cross-border enforcement challenges", "articles": "83,58,60"},
    {"chapter": "ENFORCEMENT", "theme": "the right to compensation under GDPR — conditions for liability, joint controller/processor liability, and burden of proof (Article 82)", "articles": "82"},
    {"chapter": "ENFORCEMENT", "theme": "cross-border complaint handling — cooperation between supervisory authorities and the dispute resolution mechanism (Articles 60, 65)", "articles": "60,65,77"},
    {"chapter": "ENFORCEMENT", "theme": "judicial remedies against controllers, processors, and supervisory authorities — court jurisdiction and effective remedy (Articles 78-79)", "articles": "78,79"},
    # ── Cross-Regulation ──
    {"chapter": "CROSS_REG", "theme": "GDPR and the ePrivacy Directive — cookie consent, electronic communications, and the overlap between Article 6 GDPR and Article 5(3) ePrivacy", "articles": "6,7"},
    {"chapter": "CROSS_REG", "theme": "GDPR and the EU AI Act — how AI system providers must comply with both data protection and AI-specific transparency/risk obligations", "articles": "22,35,13"},
    {"chapter": "CROSS_REG", "theme": "GDPR and the Digital Services Act (DSA) — platform liability, content moderation data processing, and transparency reporting", "articles": "6,13,14"},
    {"chapter": "CROSS_REG", "theme": "GDPR and NIS2 Directive — cybersecurity incident reporting overlap with breach notification and security measures", "articles": "32,33,34"},
    # ── International Comparison ──
    {"chapter": "INTL", "theme": "the EU-US Data Privacy Framework — adequacy decision, certification requirements, and redress mechanisms for EU data subjects", "articles": "45,49"},
    {"chapter": "INTL", "theme": "UK GDPR post-Brexit — key divergences from EU GDPR, adequacy status, and international transfer implications", "articles": "44,45,46"},
    {"chapter": "INTL", "theme": "comparing GDPR with Brazil's LGPD — similarities in data subject rights, differences in enforcement and legal bases", "articles": "6,15,17,77"},
]

QUESTION_TYPES = [
    "article_focused",
    "scenario_based",
    "comparative",
]

SYSTEM_PROMPT = """You are a senior GDPR legal expert creating high-quality training data for an AI assistant.

Rules:
1. Cite SPECIFIC, CORRECT GDPR article numbers with sub-paragraphs (e.g., Article 6(1)(a), Article 35(1)).
2. Answers must be 200-600 words, well-structured with headers or numbered lists.
3. Include practical examples or compliance steps where relevant.
4. Be precise about legal concepts — do not conflate different articles or principles.
5. Use professional but clear language accessible to compliance officers.
"""


def build_prompt(topic: dict, q_type: str) -> str:
    theme = topic["theme"]
    articles = topic["articles"]

    if q_type == "article_focused":
        return f"""Generate a GDPR training Q&A pair about: {theme}

The answer should deeply explain the relevant GDPR provisions (Articles {articles}),
including the legal requirements, practical implications, and a concrete example.

Respond in this exact JSON format:
{{
  "instruction": "<detailed question about {theme} (1-3 sentences)>",
  "input": "<shorter, direct version of the question (1 sentence)>",
  "output": "<comprehensive answer with specific article citations, 200-600 words>"
}}"""

    elif q_type == "scenario_based":
        return f"""Generate a GDPR training Q&A pair using a realistic business scenario about: {theme}

Create a practical scenario involving a specific type of organization, then answer
what GDPR obligations apply, referencing Articles {articles}.

Respond in this exact JSON format:
{{
  "instruction": "<scenario description + question (2-4 sentences)>",
  "input": "<the core compliance question from the scenario (1 sentence)>",
  "output": "<step-by-step compliance guidance with article citations, 200-600 words>"
}}"""

    else:  # comparative
        return f"""Generate a GDPR training Q&A pair that compares or contrasts concepts within: {theme}

The question should ask the reader to distinguish between related GDPR concepts,
referencing Articles {articles}. The answer should clearly differentiate them.

Respond in this exact JSON format:
{{
  "instruction": "<question asking to compare/contrast GDPR concepts (1-3 sentences)>",
  "input": "<shorter version of the comparison question (1 sentence)>",
  "output": "<structured comparison with article citations for each concept, 200-600 words>"
}}"""


def parse_response(text: str) -> dict | None:
    """Try to extract JSON from the model response."""
    # Try direct parse
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        if all(k in data for k in ("instruction", "input", "output")):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON block within text
    import re
    match = re.search(r'\{[^{}]*"instruction"[^{}]*"input"[^{}]*"output"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def call_with_retry(client, model, messages, max_retries=3, **kwargs):
    """Call API with exponential backoff on rate limit (429)."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                print(f"\n  Rate limited — retry {attempt+1}/{max_retries} after {wait}s...")
                time.sleep(wait)
            else:
                raise


def generate_sft_data(args):
    if not args.api_key:
        print("Error: UPSTAGE_API_KEY not found. Set it in .env or pass --api_key")
        return

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Resume support: count existing lines to skip
    existing_count = 0
    if not args.fresh and os.path.isfile(output_path):
        with open(output_path, "r") as f:
            existing_count = sum(1 for line in f if line.strip())
        if existing_count > 0:
            print(f"Resuming: {existing_count} samples already exist in {output_path}")

    # Build task list: each topic × num_per_topic, rotating question types
    tasks = []
    for topic in GDPR_TOPICS:
        for i in range(args.num_per_topic):
            q_type = QUESTION_TYPES[i % len(QUESTION_TYPES)]
            tasks.append((topic, q_type))

    random.seed(42)
    random.shuffle(tasks)

    # Skip already-generated tasks
    tasks = tasks[existing_count:]
    total = existing_count + len(tasks)
    print(f"Generating {len(tasks)} SFT samples using {args.model} (total target: {total})...")

    generated = 0
    errors = 0

    # Open in append mode for incremental saving
    write_mode = "w" if args.fresh or existing_count == 0 else "a"
    with open(output_path, write_mode, encoding="utf-8") as out_f:
        for topic, q_type in tqdm(tasks, initial=existing_count, total=total):
            prompt = build_prompt(topic, q_type)
            try:
                response = call_with_retry(
                    client, args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content
                parsed = parse_response(raw)
                if parsed:
                    # Remove _meta before writing (keep output clean)
                    out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    out_f.flush()  # flush after each write → crash-safe
                    generated += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"\n  Error: {e}")

            # Respect rate limits (RPM=100 → ~1.7 req/sec max)
            time.sleep(0.7)

    print(f"\n{'=' * 60}")
    print(f"SFT DATA GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  New:       {generated} samples")
    print(f"  Previous:  {existing_count} samples")
    print(f"  Total:     {existing_count + generated} samples")
    print(f"  Errors:    {errors}")
    print(f"  Output:    {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic GDPR SFT data")
    parser.add_argument("--output", type=str,
                        default=os.path.join(config.DATA_DIR, "gdpr_sft_synthetic.jsonl"))
    parser.add_argument("--model", type=str, default=config.UPSTAGE_MODEL)
    parser.add_argument("--api_key", type=str, default=config.UPSTAGE_API_KEY)
    parser.add_argument("--base_url", type=str, default=config.UPSTAGE_BASE_URL)
    parser.add_argument("--num_per_topic", type=int, default=30,
                        help="Samples per topic (default 30, total = 101 topics × 30 = 3,030)")
    parser.add_argument("--fresh", action="store_true",
                        help="Overwrite existing output file instead of resuming")
    args = parser.parse_args()
    generate_sft_data(args)
