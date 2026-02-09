import os
import pandas as pd
import numpy as np
import json
import models
import pickle
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, render_template, redirect, url_for, session, flash
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from models import db, User, Review, SearchHistory, RecommendationHistory
from dotenv import load_dotenv
load_dotenv()


# Import NLP model and Vectorizer
filename = "models/comment.pkl"
clf = pickle.load(open(filename, "rb"))
vectorizer = pickle.load(open("models/transformed.pkl", "rb"))

# Initialize global variables for df and similarity
df = None
similarity = None

def initiate_similarity():
    global df, similarity
    df = pd.read_csv("datasets/processed/final_data_processed.csv")
    # Create CountVectorizer
    CV = CountVectorizer()
    count_matrix = CV.fit_transform(df["combined_columns"])
    # Compute cosine similarity
    similarity = cosine_similarity(count_matrix)
    return df, similarity

def recommend_movies(m):
    global df, similarity
    m = m.lower()
    
    # Initialize if not already done
    if df is None or similarity is None:
        df, similarity = initiate_similarity()
    
    # Check if movie exists (case-insensitive)
    if m not in df["movie_title"].str.lower().values:
        return("Sorry! The movie you requested for is not currently available. Please check your spelling or try again with another movie")
    else:
        # Find the index of the movie (case-insensitive match)
        i = df[df["movie_title"].str.lower() == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1], reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(df["movie_title"][a])
        return l
    
# Converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    try:
        # If it's already a list, return it
        if isinstance(my_list, list):
            return my_list
            
        # If it's empty or None
        if not my_list or my_list == "[]":
            return []
            
        my_list = my_list.split('","')
        if len(my_list) > 0:
            my_list[0] = my_list[0].replace('["','')
            my_list[-1] = my_list[-1].replace('"]','')
        return my_list
    except Exception as e:
        print(f"Error converting list: {e}")
        return []

def get_suggestions():
    df = pd.read_csv("datasets/processed/final_data_processed.csv")
    return list(df["movie_title"].str.capitalize())

def get_trailer(imdb_id):
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return None
        
    try:
        # 1. Get TMDB ID
        find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_key}&external_source=imdb_id"
        response = requests.get(find_url)
        data = response.json()
        
        if not data.get('movie_results'):
            return None
            
        tmdb_id = data['movie_results'][0]['id']
        
        # 2. Get Videos
        video_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos?api_key={api_key}"
        video_response = requests.get(video_url)
        video_data = video_response.json()
        
        results = video_data.get('results', [])
        youtube_videos = [v for v in results if v['site'] == 'YouTube']
        
        if not youtube_videos:
            return None
            
        # 3. Prioritize Trailer > Teaser > Others
        trailers = [v for v in youtube_videos if v['type'] == 'Trailer']
        if trailers:
            return trailers[0]['key']
            
        teasers = [v for v in youtube_videos if v['type'] == 'Teaser']
        if teasers:
            return teasers[0]['key']
            
        # Fallback to whatever is available (Behind the Scenes, etc.)
        return youtube_videos[0]['key']
        
    except Exception as e:
        print(f"Error fetching trailer: {e}")
        return None


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


# Database Configuration
# Using POSTGRESQL_PASSWORD from environment variable
pg_password = os.environ.get("POSTGRESQL_PASSWORD")

if not pg_password:
    raise ValueError("POSTGRESQL_PASSWORD is missing in environment variables")
app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://postgres:{pg_password}@localhost/movie_db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db.init_app(app)


# Authentication Routes
# @app.route('/signup', methods=['POST'])
# def signup():
#     username = request.form['username']
#     email = request.form['email']
#     password = request.form['password']
    
#     user_exists = User.query.filter_by(email=email).first()
#     if user_exists:
#         flash('Email address already exists', 'error')
#         return redirect(url_for('home')) # Or signup page if separate

#     new_user = User(username=username, email=email, password=generate_password_hash(password))
#     db.session.add(new_user)
#     db.session.commit()
    
#     flash('Account created! Please sign in.', 'success')
#     return redirect(url_for('home'))

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    if User.query.filter_by(email=email).first():
        flash('Email already exists', 'error')
        return redirect(url_for('home'))

    if User.query.filter_by(username=username).first():
        flash('Username already exists', 'error')
        return redirect(url_for('home'))

    try:
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )

        db.session.add(new_user)
        db.session.commit()

        flash('Account created! Please sign in.', 'success')
        return redirect(url_for('home'))

    except Exception as e:
        db.session.rollback()
        print("Signup error:", e)
        flash('Signup failed. Try again.', 'error')
        return redirect(url_for('home'))


@app.route('/signin', methods=['POST'])
def signin():
    username = request.form['signinUsername'] # Matching the user provided form field names
    password = request.form['signinPassword']
    
    user = User.query.filter_by(username=username).first()
    
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.', 'error')
        return redirect(url_for('home'))

    session['user_id'] = user.id
    session['username'] = user.username
    print(session)
    return redirect(url_for('home'))


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    tmdb_api_key = os.environ.get("TMDB_API_KEY")
    return render_template("home.html", suggestions=suggestions, TMDB_API_KEY=tmdb_api_key)

@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form["name"]
    print(f"Received request for movie: {movie}")
    
    # Log Search History if user is logged in
    if 'user_id' in session:
        user_id = session['user_id']
        new_search = SearchHistory(user_id=user_id, search_term=movie)
        db.session.add(new_search)
        db.session.commit()

    rec = recommend_movies(movie)
    print(f"Recommendations: {rec}")
    
    # Log Recommendation History if user is logged in and rec is valid
    if 'user_id' in session and isinstance(rec, list):
         user_id = session['user_id']
         rec_str = ",".join(rec)
         new_rec = RecommendationHistory(user_id=user_id, searched_movie=movie, recommended_movies=rec_str)
         db.session.add(new_rec)
         db.session.commit()

    if type(rec) == type("string"):
        print("Returning error message")
        return rec
    else:
        m_str="---".join(rec)
        print(f"Returning recommendations: {m_str}")
        return m_str
    
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # getting data from AJAX request
        title = request.form['title']
        cast_ids = request.form['cast_ids']
        cast_names = request.form['cast_names']
        cast_chars = request.form['cast_chars']
        cast_bdays = request.form['cast_bdays']
        cast_bios = request.form['cast_bios']
        cast_places = request.form['cast_places']
        cast_profiles = request.form['cast_profiles']
        imdb_id = request.form['imdb_id']
        poster = request.form['poster']
        genres = request.form['genres']
        overview = request.form['overview']
        vote_average = request.form['rating']
        vote_count = request.form['vote_count']
        release_date = request.form['release_date']
        runtime = request.form['runtime']
        status = request.form['status']
        rec_movies = request.form['rec_movies']
        rec_posters = request.form['rec_posters']

        print(f"Processing recommendation request for: {title}")

        # get movie suggestions for auto complete
        suggestions = get_suggestions()

        # call the convert_to_list function for every string that needs to be converted to list
        rec_movies = convert_to_list(rec_movies)
        rec_posters = convert_to_list(rec_posters)
        cast_names = convert_to_list(cast_names)
        cast_chars = convert_to_list(cast_chars)
        cast_profiles = convert_to_list(cast_profiles)
        cast_bdays = convert_to_list(cast_bdays)
        cast_bios = convert_to_list(cast_bios)
        cast_places = convert_to_list(cast_places)
        
        # convert string to list (eg. "[1,2,3]" to [1,2,3])
        cast_ids = cast_ids.split(',')
        cast_ids[0] = cast_ids[0].replace("[","")
        cast_ids[-1] = cast_ids[-1].replace("]","")

        # rendering the string to python string
        for i in range(len(cast_bios)):
            cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

        # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
        
        casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

        cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}
        print(f"calling imdb api: {'https://www.imdb.com/title/{}/reviews/?ref_=tt_ov_rt'.format(imdb_id)}")
        
        # 1. Fetch Trailer
        trailer_key = get_trailer(imdb_id)
        
        # 2. Fetch Local DB Reviews
        # We query by movie_title (approximate match) or exact if we had imdb_id stored. 
        # Using title is safer if frontend passes title. But different users might have slightly different titles?
        # Actually input 'title' is from frontend. Let's use that.
        # Ideally we'd use imdb_id but storing that is inconsistent. Let's try to query by title.
        db_reviews = Review.query.filter(Review.movie_title.ilike(title)).all()
        
        reviews_list = [] # list of reviews
        reviews_status = [] # list of comments (good or bad)
        
        # Process DB reviews first
        for rev in db_reviews:
            reviews_list.append(rev.content)
            reviews_status.append(rev.sentiment)

        # 3. Fetch IMDB Reviews
        url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
        try:
            response = requests.get(url, headers=headers)
            print(f"IMDB response status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml')
                soup_result = soup.find_all("div", {"class": "ipc-html-content-inner-div"})
                print(f"Found {len(soup_result)} IMDB reviews")
        
                for reviews in soup_result:
                    if reviews.string:
                        reviews_list.append(reviews.string)
                        # passing the review to our model
                        movie_review_list = np.array([reviews.string])
                        movie_vector = vectorizer.transform(movie_review_list)
                        pred = clf.predict(movie_vector)
                        reviews_status.append('Good' if pred else 'Bad')
            else:
                print("Failed to retrieve reviews from IMDB")
        except Exception as e:
            print(f"IMDB Scraping Error: {e}")

        # combining reviews and comments into a dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

        # passing all the data to the html file
        # We pass user_logged_in flag to show review form
        user_logged_in = 'user_id' in session
        
        return render_template('recommender.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
            vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
            movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details, 
            TMDB_API_KEY=os.environ.get("TMDB_API_KEY"), user_logged_in=user_logged_in, trailer_key=trailer_key)
                
    except Exception as e:
        print(f"ERROR in recommend route: {e}")
        import traceback
        traceback.print_exc()
        return str(e), 500

@app.route("/add_review", methods=["POST"])
def add_review():
    if 'user_id' not in session:
        return redirect(url_for('home')) # Should enable review button only if logged in
        
    user_id = session['user_id']
    movie_title = request.form['movie_title'] # Hidden field in form
    content = request.form['review_content']
    imdb_id = request.form.get('imdb_id') # Optional
    
    # Analyze Sentiment
    movie_review_list = np.array([content])
    movie_vector = vectorizer.transform(movie_review_list)
    pred = clf.predict(movie_vector)
    sentiment = 'Good' if pred else 'Bad'
    
    new_review = Review(user_id=user_id, movie_title=movie_title, imdb_id=imdb_id, content=content, sentiment=sentiment)
    db.session.add(new_review)
    db.session.commit()
    
    # We can't easily redirect back to the movie page because it's a POST request with lots of data.
    # The frontend is AJAX based or similar?
    # Actually recommender.html is rendered via POST to /recommend. 
    # If we redirect to /home, users lose context.
    # Ideally, we should submit this via AJAX too. 
    # But for simplicity, let's redirect to home with a flash message.
    # Or strict rendering?
    flash("Review added successfully!", 'success')
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Create tables if not exist
    app.run(debug=True)
