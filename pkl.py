# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle

# df = pd.read_csv("datasets/processed/final_data_processed.csv")
# CV = CountVectorizer()
# count_matrix = CV.fit_transform(df["combined_columns"])
# similarity = cosine_similarity(count_matrix)

# # Save both
# with open("models/similarity.pkl", "wb") as f:
#     pickle.dump(similarity, f)

# with open("models/df.pkl", "wb") as f:
#     pickle.dump(df, f)


import pickle, gzip

with open("models/similarity.pkl", "rb") as f_in:
    with gzip.open("models/similarity.pkl.gz", "wb") as f_out:
        f_out.writelines(f_in)
