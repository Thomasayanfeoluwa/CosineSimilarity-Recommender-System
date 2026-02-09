import os
from flask import Flask
from models import db, User, Review
from dotenv import load_dotenv
from sqlalchemy import text



db.session.execute(text('SELECT 1'))
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

pg_password = os.environ.get("POSTGRESQL_PASSWORD")
if not pg_password:
    print("WARNING: POSTGRESQL_PASSWORD not found in environment variables.")
    pg_password = "password"

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://postgres:{pg_password}@localhost/movie_db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def check_database():
    with app.app_context():
        try:
            print(f"Connecting to: {app.config['SQLALCHEMY_DATABASE_URI'].replace(pg_password, '******')}")
            
            # Check connection
            db.session.execute('SELECT 1')
            print("Connection successful!")

            # Check Users
            user_count = User.query.count()
            print(f"User Count: {user_count}")
            users = User.query.all()
            for u in users:
                print(f" - User: {u.username}, Email: {u.email}")

            # Check Reviews
            review_count = Review.query.count()
            print(f"Review Count: {review_count}")
            reviews = Review.query.all()
            for r in reviews:
                print(f" - Review by User {r.user_id} for {r.movie_title}: {r.content[:30]}...")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_database()
