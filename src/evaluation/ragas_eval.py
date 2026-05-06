import json
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


def run_ragas_evaluation(
    generated_answers_path: str,
    answers_path: str,
    output_path: str,
    sample_size: int = 5,
) -> Dict[str, Any]:
    """RUN RAGAS EVALUATION ON GENERATED ANSWERS. **"""
    load_dotenv()

    with Path(generated_answers_path).open("r", encoding="utf-8") as file:
        generated_answers = json.load(file)

    with Path(answers_path).open("r", encoding="utf-8") as file:
        ground_truth_answers = json.load(file)

    if sample_size:
        generated_answers = generated_answers[:sample_size]

    rows: List[Dict[str, Any]] = []

    for item in generated_answers:
        query_id = item["query_id"]

        if query_id not in ground_truth_answers:
            continue

        rows.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "contexts": [
                    chunk["chunk_text"]
                    for chunk in item["retrieved_chunks"]
                ],
                "ground_truth": ground_truth_answers[query_id],
            }
        )

    dataset = Dataset.from_list(rows)

    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    result_df = result.to_pandas()
    summary = result_df.mean(numeric_only=True).to_dict()

    output = {
        "sample_size": len(rows),
        "metrics": summary,
        "per_query": result_df.to_dict(orient="records"),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    return output