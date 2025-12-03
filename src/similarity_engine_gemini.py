import google.generativeai as genai
import pandas as pd
import numpy as np

class CSVCosineSimilarity:
    """
    Compute row-to-row cosine similarity using Gemini embeddings (Text-Embedding-004).
    Uses API key only (NO ADC, NO GCP credentials).
    """

    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("Gemini API key missing.")

        # Configure API key (forces non-ADC mode)
        genai.configure(api_key=gemini_api_key)

        # Model name (standard)
        self.embedding_model = "models/text-embedding-004"

    # ---------------------------
    # INTERNAL: Compute embedding
    # ---------------------------
    def get_embedding(self, text: str):
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)

        response = genai.embed_content(
            model=self.embedding_model,
            content=text
        )

        return np.array(response["embedding"])

    # ---------------------------
    # INTERNAL: Cosine similarity
    # ---------------------------
    def cosine_similarity(self, a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ---------------------------
    # MAIN: Compare two CSV files
    # ---------------------------
    def compute_row_similarity(self, left_df, right_df, left_columns, right_columns):
        result_rows = []

        # Merge selected text columns
        left_df["__merged_text__"] = left_df[left_columns].astype(str).agg(" ".join, axis=1)
        right_df["__merged_text__"] = right_df[right_columns].astype(str).agg(" ".join, axis=1)

        # Precompute embeddings
        left_embeddings = [self.get_embedding(t) for t in left_df["__merged_text__"]]
        right_embeddings = [self.get_embedding(t) for t in right_df["__merged_text__"]]

        # Compare each left row to all right rows
        for i, left_vector in enumerate(left_embeddings):
            best_match_idx = None
            best_score = -1

            for j, right_vector in enumerate(right_embeddings):
                score = self.cosine_similarity(left_vector, right_vector)
                if score > best_score:
                    best_score = score
                    best_match_idx = j

            comment = "OK" if best_score > 0.75 else "Forward to OPs"

            result_rows.append({
                "Left Row": i + 1,
                "Left Text": left_df["__merged_text__"].iloc[i],
                "Right Row": best_match_idx + 1,
                "Right Text": right_df["__merged_text__"].iloc[best_match_idx],
                "Score": round(best_score, 4),
                "Comment": comment
            })

        return pd.DataFrame(result_rows)
