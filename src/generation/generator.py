import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

from src.generation.prompt_builder import build_grounded_prompt


class AnswerGenerator:
    """GENERATE GROUNDED ANSWERS FROM RETRIEVED CONTEXT. **"""

    def __init__(self, llm_config: Dict[str, Any], generation_config: Dict[str, Any]):
        """INITIALIZE LLM CLIENT. **"""
        load_dotenv()

        self.provider = llm_config["provider"]
        self.model_name = llm_config["name"]
        self.temperature = llm_config.get("temperature", 0.0)
        self.max_tokens = llm_config.get("max_tokens", 700)
        self.fallback_message = generation_config["fallback_message"]

        if self.provider == "groq":
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        elif self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """GENERATE ANSWER FROM QUESTION AND RETRIEVED CHUNKS. **"""
        prompt = build_grounded_prompt(
            question=question,
            chunks=chunks,
            fallback_message=self.fallback_message,
        )

        return self.generate_from_prompt(prompt)
    

    def generate_from_prompt(self, prompt: str) -> str:
        """GENERATE TEXT DIRECTLY FROM A PROMPT. **"""

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        raise ValueError(f"Unsupported LLM provider: {self.provider}")