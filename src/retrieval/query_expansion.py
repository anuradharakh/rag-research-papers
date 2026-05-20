from typing import Any, Dict, List

from src.generation.generator import AnswerGenerator


class HyDEGenerator:
    """GENERATE HYPOTHETICAL DOCUMENT ANSWER FOR HyDE RETRIEVAL. **"""

    def __init__(self, llm_config: Dict[str, Any], expansion_config: Dict[str, Any]):
        self.generator = AnswerGenerator(
            llm_config=llm_config,
            generation_config={
                "fallback_message": "",
            },
        )
        self.max_tokens = expansion_config.get("max_tokens", 250)

    def generate(self, query: str) -> str:
        """GENERATE A SHORT HYPOTHETICAL ANSWER FOR RETRIEVAL. **"""
        prompt = f"""
You are helping improve retrieval for scientific papers.

Given the user question, write a short hypothetical answer paragraph that might appear in a relevant research paper.
Do not say you are guessing. Do not add citations. Keep it technical and concise.

Question:
{query}

Hypothetical answer:
""".strip()

        return self.generator.generate_from_prompt(prompt)


class MultiQueryGenerator:
    """GENERATE MULTIPLE QUERY REFORMULATIONS. **"""

    def __init__(self, llm_config: Dict[str, Any], expansion_config: Dict[str, Any]):
        self.generator = AnswerGenerator(
            llm_config=llm_config,
            generation_config={
                "fallback_message": "",
            },
        )
        self.num_queries = expansion_config.get("num_queries", 3)

    def generate(self, query: str) -> List[str]:
        """GENERATE QUERY REFORMULATIONS. **"""
        prompt = f"""
Rewrite the question into {self.num_queries} diverse search queries for retrieving relevant scientific paper passages.

Rules:
- Keep each query concise.
- Preserve the original meaning.
- Use technical keywords when useful.
- Return one query per line.
- Do not number the queries.

Original question:
{query}

Rewritten queries:
""".strip()

        response = self.generator.generate_from_prompt(prompt)

        queries = [
            line.strip("- ").strip()
            for line in response.splitlines()
            if line.strip()
        ]

        return queries[: self.num_queries]