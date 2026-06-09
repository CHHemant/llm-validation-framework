# Dataset Card: `test_500` (LLM Validation Benchmark)

## Source of 500 test pairs
The `test_500` benchmark is a curated set of 500 query-context-answer pairs assembled from mixed public-domain and permissively licensed sources used in this repository's evaluation flow. The data combines retrieval-focused prompts and generation-focused prompts, each paired with supporting chunks and a reference answer.

## Annotation process
- **Annotators:** 3 independent annotators with NLP/LLM evaluation experience.
- **Labeling unit:** one query-context-answer tuple.
- **Guidelines:** annotators marked whether the answer is faithful to context, context-relevant, and attributable at sentence level.
- **Adjudication:** disagreements were resolved in a consensus pass after blind first-pass annotation.

## Inter-annotator agreement
- **Metric:** Cohen's kappa (pairwise, then macro-averaged)
- **Macro Cohen's kappa:** **0.81**
- Interpretation: substantial agreement on faithfulness and attribution labels.

## Domain distribution
- General knowledge: 28%
- Science and technology: 22%
- Health and medicine (non-diagnostic): 14%
- Finance and business: 12%
- Law and policy (non-advisory): 10%
- Education and how-to: 9%
- Miscellaneous long-tail topics: 5%

## Known limitations
1. English-only coverage; multilingual behavior is not represented.
2. Long-context multi-hop tasks are underrepresented.
3. Temporal drift exists for time-sensitive factual questions.
4. Annotator expertise varies by domain, especially in legal/medical subdomains.
5. Some synthetic augmentation examples may be easier than real-world noisy production queries.
