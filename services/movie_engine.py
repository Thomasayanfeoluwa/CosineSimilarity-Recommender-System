import pandas as pd
import numpy as np
import pickle
import faiss
import requests
import os
import logging
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

class MovieEngine:
    clf = None
    vectorizer = None
    df = None
    svd = None
    faiss_index = None

    @classmethod
    def _get_project_root(cls):
        """Helper method to get project root"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_dir)

    @classmethod
    def get_clf_vectorizer(cls):
        if cls.clf is None or cls.vectorizer is None:
            try:
                project_root = cls._get_project_root()
                
                model_path = os.path.join(project_root, "models", "comment_sentiments.pkl")
                vectorizer_path = os.path.join(project_root, "models", "transformed.pkl")

                with open(model_path, "rb") as f:
                    cls.clf = pickle.load(f)
                with open(vectorizer_path, "rb") as f:
                    cls.vectorizer = pickle.load(f)
                logging.info(f"✅ Models Loaded Successfully from {project_root}/models/")
            except FileNotFoundError as e:
                logging.error(f"❌ Models Loading Failed: {e}")
                raise e
        return cls.clf, cls.vectorizer

    @classmethod
    def get_df_engine(cls):
        if cls.df is None:
            try:
                project_root = cls._get_project_root()
                
                df_path = os.path.join(project_root, "models", "df.pkl")
                svd_path = os.path.join(project_root, "models", "svd.pkl")
                faiss_path = os.path.join(project_root, "models", "faiss_movies.index")
                
                with open(df_path, "rb") as f:
                    cls.df = pickle.load(f)
                cls.df["movie_title_clean"] = cls.df["movie_title"].str.strip().str.lower()
                if not hasattr(cls, "lookup_dict"):
                    cls.lookup_dict = dict(zip(cls.df["movie_title_clean"], cls.df.index))

                if cls.svd is None:
                    with open(svd_path, "rb") as f:
                        cls.svd = pickle.load(f) 
                    
                if cls.faiss_index is None:
                    cls.faiss_index = faiss.read_index(faiss_path)  

                logging.info(f"✅ FAISS Models Loaded Successfully from {project_root}/models/")
            except FileNotFoundError as e:
                logging.error(f"❌ FAISS Models Loading Failed: {e}")
                raise e
        return cls.df, cls.svd, cls.faiss_index

    @classmethod
    def get_vectorizer(cls):
        if cls.vectorizer is None:
            try:
                project_root = cls._get_project_root()
                vectorizer_path = os.path.join(project_root, "models", "transformed.pkl")
                
                with open(vectorizer_path, "rb") as f:
                    cls.vectorizer = pickle.load(f)
                logging.info(f"✅ Vectorizer Loaded Successfully!")
            except FileNotFoundError as e:
                logging.error(f"❌ Vectorizer Loading Failed: {e}")
                raise e
        return cls.vectorizer

    @classmethod
    def recommend_movies(cls, movie_title):
        df, svd, faiss_index = cls.get_df_engine()
        vectorizer = cls.get_vectorizer()

        m_clean = movie_title.strip().lower()
        lookup_dict = dict(zip(df["movie_title_clean"], df.index))
        if m_clean not in lookup_dict:
            return "Sorry! The movie you requested for is not available."
        i = lookup_dict[m_clean]

        movies_text = df.loc[i, "combined_columns"]
        tfidf_vec = vectorizer.transform([movies_text])
        query_vector = svd.transform(tfidf_vec).astype("float32")
        faiss.normalize_L2(query_vector)
        distance, indices = faiss_index.search(query_vector, k=12)
        neighbor_indices = [idx for idx in indices[0] if idx != i]
        recommendations = [df["movie_title"].iloc[idx] for idx in neighbor_indices][:10]
        return recommendations

    
    @classmethod
    def convert_to_list(cls, my_list):
        try:
            if isinstance(my_list, list):
                return my_list
            if not my_list or my_list == "[]":
                return []
            my_list = my_list.split('","')
            if len(my_list) > 0:
                my_list[0] = my_list[0].replace('["','')
                my_list[-1] = my_list[-1].replace('"]', '')
            return my_list
        except Exception as e:
            logging.info(f"Error Converting List: {e}")
            return []

        
    @classmethod
    def get_suggestions(cls):
        df, _, _ = cls.get_df_engine()
        return list(df["movie_title"].str.capitalize())


    @classmethod
    def get_trailer(cls, imdb_id):
        api_key = os.environ.get("TMDB_API_KEY")
        if not api_key:
            logging.error("TMDB_API_KEY not found in environment variables!")
            return None
        try:
            find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
            response = requests.get(find_url)
            data = response.json()
            if not data.get("movie_results"):
                return None
            tmdb_id = data['movie_results'][0]['id']
            video_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos?api_key={api_key}"
            video_response = requests.get(video_url)
            video_data = video_response.json()
            results = video_data.get('results', [])
            youtube_videos = [v for v in results if v['site'] == 'YouTube']
            if not youtube_videos:
                return None
            trailers = [v for v in youtube_videos if v['type'] == 'Trailer']
            if trailers:
                return trailers[0]['key']
            teasers = [v for v in youtube_videos if v['type'] == 'Teaser']
            if teasers:
                return teasers[0]['key']
            return youtube_videos[0]['key']
        except Exception as e:
            logging.info(f"Error fetching trailer: {e}")
            return None


if __name__ == "__main__":
    MovieEngine().get_df_engine()