import requests
import os
import logging

class TMDBService:
    API_KEY = os.environ.get("TMDB_API_KEY")
    BASE_URL = "https://api.themoviedb.org/3"
    
    @classmethod
    def search_movie(cls, query):
        try:
            response = requests.get(
                f"{cls.BASE_URL}/search/movie",
                params={'api_key': cls.API_KEY, 'query': query}
            )
            return response.json()
        except Exception as e:
            logging.error(f"TMDB Search Error: {e}")
            return {'error': str(e)}
    
    @classmethod
    def get_movie_details(cls, movie_id):
        try:
            response = requests.get(
                f"{cls.BASE_URL}/movie/{movie_id}",
                params={'api_key': cls.API_KEY}
            )
            return response.json()
        except Exception as e:
            logging.error(f"TMDB Movie Details Error: {e}")
            return {'error': str(e)}
    
    @classmethod
    def get_movie_credits(cls, movie_id):
        try:
            response = requests.get(
                f"{cls.BASE_URL}/movie/{movie_id}/credits",
                params={'api_key': cls.API_KEY}
            )
            return response.json()
        except Exception as e:
            logging.error(f"TMDB Credits Error: {e}")
            return {'error': str(e)}
    
    @classmethod
    def get_person_details(cls, person_id):
        try:
            response = requests.get(
                f"{cls.BASE_URL}/person/{person_id}",
                params={'api_key': cls.API_KEY}
            )
            return response.json()
        except Exception as e:
            logging.error(f"TMDB Person Error: {e}")
            return {'error': str(e)}