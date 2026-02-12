from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    
    # Relationships
    reviews = db.relationship('Review', backref='author', lazy=True)
    searches = db.relationship('SearchHistory', backref='user', lazy=True)
    recommendations = db.relationship('RecommendationHistory', backref='user', lazy=True)

class Review(db.Model):
    __tablename__ = 'reviews'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    movie_title = db.Column(db.String(200), nullable=False)
    # Storing imdb_id might be useful for exact matching if available, but title is easier for now
    imdb_id = db.Column(db.String(20), nullable=True) 
    content = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False) # 'Good' or 'Bad'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class SearchHistory(db.Model):
    __tablename__ = 'search_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    search_term = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class RecommendationHistory(db.Model):
    __tablename__ = 'recommendation_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    searched_movie = db.Column(db.String(200), nullable=False)
    # Storing recommended movies as a simple text string (comma separated) for simplicity
    recommended_movies = db.Column(db.Text, nullable=False) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
