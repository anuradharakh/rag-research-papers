from typing import Any, Dict, List

from src.indexing.embedding import EmbeddingModel
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_expansion import HyDEGenerator, MultiQueryGenerator
from src.retrieval.rrf import reciprocal_rank_fusion


class ExpandedHybridRetriever:
    """HYDE + MULTI-QUERY EXPANDED HYBRID RETRIEVER. **"""

    def __init__(
        self,
        experiment_name: str,
        index_dir: str,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        embedding_model: EmbeddingModel,
        llm_config: Dict[str, Any],
        query_expansion_config: Dict[str, Any],
        rrf_k: int = 60,
    ):
        self.experiment_name = experiment_name
        self.index_dir = index_dir
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k

        self.hyde_enabled = query_expansion_config.get("hyde", {}).get("enabled", False)
        self.multi_query_enabled = query_expansion_config.get("multi_query", {}).get("enabled", False)

        self.hyde_generator = None
        self.multi_query_generator = None

        if self.hyde_enabled:
            self.hyde_generator = HyDEGenerator(
                llm_config=llm_config,
                expansion_config=query_expansion_config["hyde"],
            )

        if self.multi_query_enabled:
            self.multi_query_generator = MultiQueryGenerator(
                llm_config=llm_config,
                expansion_config=query_expansion_config["multi_query"],
            )

    def retrieve(
        self,
        query: str,
        fetch_k: int = 20,
        top_k: int = 3,
        multi_query_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """RETRIEVE USING ORIGINAL QUERY + HYDE + MULTI-QUERY EXPANSION. **"""

        ranked_lists = []

        original_results = self.hybrid_retriever.retrieve(
            query=query,
            fetch_k=fetch_k,
            top_k=fetch_k,
        )
        ranked_lists.append(original_results)

        if self.hyde_generator is not None:
            hyde_text = self.hyde_generator.generate(query)

            hyde_results = self.dense_retriever.retrieve(
                query=hyde_text,
                top_k=fetch_k,
            )

            for item in hyde_results:
                item["retrieval_strategy"] = "hyde_dense"
                item["hyde_text"] = hyde_text

            ranked_lists.append(hyde_results)

        if self.multi_query_generator is not None:
            expanded_queries = self.multi_query_generator.generate(query)

            for expanded_query in expanded_queries:
                expanded_results = self.hybrid_retriever.retrieve(
                    query=expanded_query,
                    fetch_k=multi_query_k,
                    top_k=multi_query_k,
                )

                for item in expanded_results:
                    item["retrieval_strategy"] = "multi_query_hybrid"
                    item["expanded_query"] = expanded_query

                ranked_lists.append(expanded_results)

        fused_results = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            rrf_k=self.rrf_k,
            top_k=top_k,
        )

        for item in fused_results:
            item["retrieval_strategy"] = "expanded_hybrid_rrf"

        return fused_results