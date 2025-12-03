import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class CSVCosineSimilarity:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )

    def embed_text_list(self, text_list):
        return self.embeddings.embed_documents(text_list)

    def compute_row_similarity(self, df_left, df_right,
                               selected_columns_left, selected_columns_right):

        left_texts = (
            df_left[selected_columns_left]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )

        right_texts = (
            df_right[selected_columns_right]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )

        left_vecs = self.embed_text_list(left_texts)
        right_vecs = self.embed_text_list(right_texts)

        results = []

        for i, left_vec in enumerate(left_vecs):
            left_arr = np.array(left_vec).reshape(1, -1)

            best_score = -1
            best_index = None
            best_text = ""

            for j, right_vec in enumerate(right_vecs):
                right_arr = np.array(right_vec).reshape(1, -1)
                score = cosine_similarity(left_arr, right_arr)[0][0]

                if score > best_score:
                    best_score = score
                    best_index = j + 1
                    best_text = right_texts[j]

            similarity_pct = round(best_score * 100, 2)
            comment = "OK" if similarity_pct == 100 else "Forward to OPs"

            results.append({
                "Left Row Index": i + 1,
                "Left Text": left_texts[i],
                "Best Match Row (Right)": best_index,
                "Right Text": best_text,
                "Similarity (%)": similarity_pct,
                "Comments": comment,
            })

        return pd.DataFrame(results)
