import requests
import os
import time
import logging

class TMDBService:
    API_KEY = os.environ.get("TMDB_API_KEY")
    BASE_URL = "https://api.themoviedb.org/3"

    _cache = {}
    _cache_ttl = 86400  # 24 hours
    
    @classmethod
    def get_movie_with_ratings(cls, title):
        cache_key = f"enriched_{title}"
        
        # Use existing cache system
        cached = cls._get_cached(cache_key)
        if cached:
            return cached
        
        # Fetch fresh data
        result = cls.search_movie(title)
        if result.get('results'):
            movie_id = result['results'][0]['id']
            details = cls.get_movie_details(movie_id)
            enriched_data = {
                'vote_average': details.get('vote_average'),
                'popularity': details.get('popularity'),
                'vote_count': details.get('vote_count')
            }
            # Use existing cache setter
            cls._set_cache(cache_key, enriched_data)
            return enriched_data
        return None

    @classmethod
    def _get_cached(cls, key):
        if key in cls._cache:
            data, timestamp = cls._cache[key]
            # Check if cache is still valid (less than 24 hours old)
            if time.time() - timestamp < cls._cache_ttl:
                logging.info(f"Cache hit for: {key}")
                return data
            else:
                logging.info(f"Cache expired for: {key}")
        return None
    
    @classmethod
    def _set_cache(cls, key, data):
        cls._cache[key] = (data, time.time())
        logging.info(f"Cached data for: {key}")
    
    @classmethod
    def search_movie(cls, query):
        cache_key = f"search_{query}"
        cached = cls._get_cached(cache_key)
        if cached:
            return cached
        try:
            response = requests.get(
                f"{cls.BASE_URL}/search/movie",
                params={'api_key': cls.API_KEY, 'query': query},
                timeout=3
            )
            result = response.json()
            cls._set_cache(cache_key, result)
            return result
        except Exception as e:
            logging.error(f"TMDB Search Error: {e}")
            return {'error': str(e)}
    
    @classmethod
    def get_movie_details(cls, movie_id):
        cache_key = f"details_{movie_id}"
        cached = cls._get_cached(cache_key)
        if cached:
            return cached
        try:
            response = requests.get(
                f"{cls.BASE_URL}/movie/{movie_id}",
                params={'api_key': cls.API_KEY}
            )
            result = response.json()
            cls._set_cache(cache_key, result)
            return result
        except Exception as e:
            logging.error(f"TMDB Movie Details Error: {e}")
            return {'error': str(e)}
    
    @classmethod
    def get_movie_credits(cls, movie_id):
        cache_key = f"credits_{movie_id}"
        cached = cls._get_cached(cache_key)
        if cached:
            return cached
        try:
            response = requests.get(
                f"{cls.BASE_URL}/movie/{movie_id}/credits",
                params={'api_key': cls.API_KEY}
            )
            result = response.json()
            cls._set_cache(cache_key, result)
            return result
        except Exception as e:
            logging.error(f"TMDB Credits Error: {e}")
            return {'error': str(e)}
    
    @classmethod
    def get_person_details(cls, person_id):
        cache_key = f"person_{person_id}"
        cached = cls._get_cached(cache_key)
        if cached:
            return cached
        try:
            response = requests.get(
                f"{cls.BASE_URL}/person/{person_id}",
                params={'api_key': cls.API_KEY}
            )
            result = response.json()
            cls._set_cache(cache_key, result)
            return result
        except Exception as e:
            logging.error(f"TMDB Person Error: {e}")
            return {'error': str(e)}
