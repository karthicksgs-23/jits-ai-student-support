from pathlib import Path
import re

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from crewai.tools import tool

from tools.self_rag_tool import check_rag_evidence


DATA_FILE = Path("data/academic_regulations.txt")

# Local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HybridRAG:
    def __init__(self):
        print("Loading Hybrid RAG...")

        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"{DATA_FILE} not found. Run scraper_tool.py first."
            )

        # 1. Load scraped knowledge
        self.text = DATA_FILE.read_text(encoding="utf-8")

        # 2. Split into chunks
        self.chunks = self.chunk_text(self.text)

        print(f"Created {len(self.chunks)} chunks")

        # 3. Load embedding model
        print("Loading embedding model...")

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # 4. Create embeddings
        print("Creating vector embeddings...")

        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        # Normalize embeddings so Inner Product becomes cosine similarity
        faiss.normalize_L2(embeddings)

        self.embeddings = embeddings

        # 5. Create FAISS index
        dimension = embeddings.shape[1]

        self.faiss_index = faiss.IndexFlatIP(dimension)

        self.faiss_index.add(embeddings)

        # 6. Create BM25 index
        print("Creating BM25 index...")

        tokenized_chunks = [
            self.tokenize(chunk)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

        print("Hybrid RAG ready.")


    def tokenize(self, text):
        return re.findall(
            r"\b\w+\b",
            text.lower()
        )


    def chunk_text(
        self,
        text,
        chunk_size=1200,
        overlap=200
    ):
        """
        Character-based chunking with overlap.
        """

        # Clean excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            # Try not to cut in the middle of a paragraph
            if end < len(text):

                paragraph_end = chunk.rfind("\n\n")

                if paragraph_end > chunk_size // 2:
                    chunk = chunk[:paragraph_end]

            chunk = chunk.strip()

            if chunk:
                chunks.append(chunk)

            next_start = start + len(chunk) - overlap

            if next_start <= start:
                next_start = start + chunk_size - overlap

            start = next_start

        return chunks


    def normalize_scores(self, scores):
        scores = np.array(scores, dtype="float32")

        if len(scores) == 0:
            return scores

        minimum = scores.min()
        maximum = scores.max()

        if maximum - minimum == 0:
            return np.zeros_like(scores)

        return (scores - minimum) / (maximum - minimum)


    def search(
        self,
        query,
        top_k=5,
        vector_weight=0.65,
        bm25_weight=0.35
    ):
        """
        Hybrid Retrieval:
        65% semantic FAISS
        35% BM25 keyword search
        """

        # --------------------------
        # VECTOR SEARCH
        # --------------------------

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        vector_scores, vector_ids = self.faiss_index.search(
            query_embedding,
            len(self.chunks)
        )

        vector_score_map = {
            int(chunk_id): float(score)
            for chunk_id, score
            in zip(vector_ids[0], vector_scores[0])
        }

        # --------------------------
        # BM25 SEARCH
        # --------------------------

        query_tokens = self.tokenize(query)

        bm25_scores = self.bm25.get_scores(query_tokens)

        # --------------------------
        # NORMALIZE SCORES
        # --------------------------

        ordered_vector_scores = np.array([
            vector_score_map.get(i, 0)
            for i in range(len(self.chunks))
        ])

        normalized_vector = self.normalize_scores(
            ordered_vector_scores
        )

        normalized_bm25 = self.normalize_scores(
            bm25_scores
        )

        # --------------------------
        # HYBRID SCORE
        # --------------------------

        final_scores = (
            vector_weight * normalized_vector
            +
            bm25_weight * normalized_bm25
        )

        best_ids = np.argsort(final_scores)[::-1][:top_k]

        results = []

        for rank, chunk_id in enumerate(best_ids, start=1):

            results.append({
                "rank": rank,
                "score": float(final_scores[chunk_id]),
                "content": self.chunks[chunk_id]
            })

        return results
# -------------------------------------------------
# Keep one RAG instance in memory
# -------------------------------------------------

_rag_instance = None


def get_rag():
    global _rag_instance

    if _rag_instance is None:
        _rag_instance = HybridRAG()

    return _rag_instance


# -------------------------------------------------
# CrewAI Tool
# -------------------------------------------------

@tool("JITS Hybrid RAG Search")
def hybrid_rag_search(question: str) -> str:
    """
    Search the JITS J-25 Academic Regulations using
    FAISS + BM25 and validate the retrieved evidence
    using a Self-RAG evidence grader.
    """

    # -----------------------------------------------------
    # 1. Hybrid retrieval
    # -----------------------------------------------------

    rag = get_rag()

    results = rag.search(
        query=question,
        top_k=5
    )

    if not results:
        return "SELF_RAG_INSUFFICIENT"


    # -----------------------------------------------------
    # 2. Build retrieved evidence
    # -----------------------------------------------------

    evidence_parts = []

    for result in results:

        evidence_parts.append(
            f"""
DOCUMENT CHUNK {result['rank']}
Hybrid relevance score: {result['score']:.4f}

{result['content']}
"""
        )


    evidence = "\n\n".join(
        evidence_parts
    )


    # -----------------------------------------------------
    # 3. SELF-RAG CHECK
    # -----------------------------------------------------

    print("\n================================")
    print("SELF-RAG EVIDENCE CHECK")
    print("================================")


    is_sufficient = check_rag_evidence(

        question=question,

        context=evidence
    )


    # -----------------------------------------------------
    # 4. Reject misleading evidence
    # -----------------------------------------------------

    if not is_sufficient:

        print(
            "\nSelf-RAG rejected the retrieved evidence."
        )

        return "SELF_RAG_INSUFFICIENT"


    # -----------------------------------------------------
    # 5. Evidence passed Self-RAG
    # -----------------------------------------------------

    print(
        "\nSelf-RAG approved the retrieved evidence."
    )


    return evidence

if __name__ == "__main__":

    rag = HybridRAG()

    question = input(
        "\nEnter your question: "
    )

    results = rag.search(question)

    print("\n==============================")
    print("HYBRID RAG RESULTS")
    print("==============================")

    for result in results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print("------------------------------")

        print(
            result["content"][:1000]
        )

        print("\n==============================")