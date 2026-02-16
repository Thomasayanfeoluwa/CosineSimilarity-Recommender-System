# 🎬 Movie Recommendation System with Trailer and Sentiment Analysis

The primary goal of this project is to design and deploy an intelligent movie recommendation system that enhances user content discovery, improves engagement, and delivers personalized entertainment experiences while maintaining cost-efficient and scalable infrastructure. The project aims to solve the challenge of overwhelming content libraries by leveraging machine learning techniques to recommend relevant movies based on user interests, movie metadata, and similarity analysis. Additionally, the system integrates user interaction features such as reviews and search behavior to support data-driven personalization and continuous improvement of recommendation accuracy. From a business perspective, the solution was built to improve user retention, increase platform engagement time, optimize infrastructure resources through model compression and FAISS indexing, and provide actionable user preference insights that can support strategic decision-making, content acquisition planning, and competitive advantage in digital entertainment platforms.

https://github.com/user-attachments/assets/64f280a4-f5ae-40dd-a605-2b997cfaf889


<p align="center">
  <img width="150" height="150" alt="TrailerMatch - Watch Trailers + Smart Recommendations" src="https://github.com/user-attachments/assets/a8606912-3408-4cbe-9f34-9b872f0d8319" />
</p>

## 🎬 Demo
<p align="center">
  <a href="https://movies-recommender-system-kxlg.onrender.com/home">
    <img src="https://img.shields.io/badge/LIVE-DEMO-red?style=for-the-badge&logo=render">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask">
</p>


*Search any movie, watch trailers, get AI-powered recommendations and sentiment analysis from real reviews!*

A robust, full-stack movie recommendation engine built with **Flask**, **PostgreSQL**, and **Machine Learning**. This system leverages **Cosine Similarity** and **FAISS (Facebook AI Similarity Search)** for high-performance recommendations, alongside real-time **Sentiment Analysis** for user reviews.

---

## 🚀 Key Features

- **Advanced Recommendation Engine**: Utilization of **TF-IDF** vectorization and **Cosine Similarity** to recommend movies based on content metadata.
- **High-Performance Indexing**: Integrated **FAISS** with **TruncatedSVD** dimensionality reduction to handle large datasets efficiently on resource-constrained environments (e.g., Render free tier).
- **Sentiment Analysis**: Real-time classification of user reviews (Good/Bad) using a model trained on 20,000 IMDb reviews with **Naive Bayes**, **Logistic Regression**, and **Linear SVM**.
- **User Authentication**: Secure Signup/Login system with password hashing and session management.
- **Interactive UI**: Dynamic interface with movie trailers, cast details, and responsive design.
- **Data Management**:**PostgreSQL** database for storing users, reviews, search history, and recommendation logs.
- **API Integration**: Real-time data fetching from the **TMDB API** for up-to-date movie metadata, posters, and trailers.

---

## 🏗️ System Architecture

1.  **Data Collection**:
    - Scraped IMDb datasets to build a comprehensive initial movie database.
    - Integrated TMDB API for real-time metadata updates.

2.  **Model Training (Sentiment Analysis)**:
    - Analyzed and preprocessed a dataset of 20,000 reviews.
    - Trained multiple algorithms: **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear SVC**.
    - Selected the best-performing model for production to ensure accurate sentiment classification.

3.  **Recommendation Engine & Optimization**:
    - **Vectorization**: Used TF-IDF to convert text data into numerical vectors.
    - **Dimensionality Reduction**: Applied **TruncatedSVD** to reduce the feature space. 
    - **Indexing**: Implemented **FAISS** index to enable fast similarity searches, solving the challenge of deploying large similarity matrices on limited cloud storage.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Gunicorn
- **Database**: PostgreSQL
- **Machine Learning**: Scikit-Learn, NumPy, Pandas, FAISS, NLTK
- **Frontend**: HTML5, CSS3, JavaScript (AJAX), Bootstrap
- **APIs**: The Movie Database (TMDB) API
- **Deployment**: Render

---

## 💡 Professional Handling: Challenges & Solutions

### 1. Deployment Limitation Due to Large Model File Size
**Problem:** The trained recommendation model (`.pkl`) and similarity matrix exceeded the storage limits of the free-tier Render hosting, causing build failures.
**Professional Handling:**
- Optimized storage by implementing **FAISS** for indexing instead of loading the full similarity matrix.
- Applied **TruncatedSVD** to reduce dataset dimensionality, creating a lighter, production-optimized model.
- Successfully deployed the system while maintaining recommendation accuracy.

### 2. Performance Optimization Using FAISS Indexing
**Problem:** Brute-force cosine similarity on large vectors resulted in slow query responses and high latency.
**Professional Handling:**
- Replaced brute-force search with **FAISS (Facebook AI Similarity Search)**.
- Built a vector index enabling approximate nearest neighbor search.
- Significantly reduced query latency, enabling real-time scalable recommendations.

### 3. Integrating Cosine Similarity with Flask
**Problem:** Aligning data preprocessing, model loading, and real-time query handling created sync issues and potential crashes.
**Professional Handling:**
- Implemented robust validation checks before querying the index.
- Added graceful fallback messaging for missing movies.
- Optimized data serialization to prevent memory spikes during runtime.

### 4. TMDB API Integration & Synchronization
**Problem:** Asynchronous API calls for metadata, posters, and trailers led to incomplete page loads and synchronization issues.
**Professional Handling:**
- Designed structured **AJAX request chains** for sequential data loading.
- Implemented comprehensive error handling and loading indicators (spinners) to improve UX.
- Ensured consistent UI rendering even if partial API data fails.

### 5. Handling Null or Missing Metadata
**Problem:** Critical identifiers (like IMDb IDs) returning null values caused review submission errors.
**Professional Handling:**
- Added hidden form validation and defensive backend logic.
- Improved template data binding to prevent data corruption.
- Ensured data consistency across user sessions.

### 6. Internal Server Errors During Reviews
**Problem:** Server crashes during review submission due to improper form handling.
**Professional Handling:**
- Implemented structured error logging.
- Validated all form parameters server-side before database insertion.
- Strengthened Flask route exception handling for stability.

### 7. API Key Exposure Risk
**Problem:** API keys were initially exposed in client-side JavaScript.
**Professional Handling:**
- Migrated sensitive API calls to backend Flask routes (`/api/tmdb/...`).
- Removed console logging of credentials to protect intellectual property.

### 8. Frontend Input Validation & UX
**Problem:** Empty search queries and duplicate requests degraded the user experience.
**Professional Handling:**
- Implemented dynamic button states (disable on submit).
- Added keyboard event interception and autocomplete suggestions.
- Reduced invalid backend requests and improved responsiveness.

### 9. Database Schema & Data Integrity
**Problem:** Inconsistent review data due to poor field mapping.
**Professional Handling:**
- Designed a structured **PostgreSQL** schema.
- Implemented timestamp logging and strict data validation.
- Enabled accurate analytics and scalable review storage.

### 10. Performance Bottlenecks from API Calls
**Problem:** Sequential external API calls increased page load times.
**Professional Handling:**
- Optimized request flow with efficient data batching.
- Implemented a robust in-memory caching layer within the TMDBService class. This system stores API responses for 24 hours, significantly reducing redundant network requests.
- Reduced redundant calls and implemented caching strategies where possible.
- Improved user retention through faster load times.

### 11. Security & Session Management
**Problem:** Need for secure authentication and session handling.
**Professional Handling:**
- Implemented Flask session authentication securely.
- Protected sensitive routes (e.g., adding reviews) with login checks.
- Enhanced platform trustworthiness and user data protection.

---

### Home Page
## 📸 Screenshots

<img width="1366" height="768" alt="Screenshot (261)" src="https://github.com/user-attachments/assets/94d02454-e71e-4ee8-b729-3303e3d51ec4" />

### Movie Details & Trailer
<img width="1366" height="732" alt="Screenshot (262)" src="https://github.com/user-attachments/assets/86f1ab95-c2cd-4d6c-b7a3-366b5e8359c3" />

### Reviews
<img width="1366" height="733" alt="Screenshot (264)" src="https://github.com/user-attachments/assets/38098191-aeb5-4b1d-af22-b066eadb2603" />

### Recommendations
<img width="1366" height="733" alt="Screenshot (263)" src="https://github.com/user-attachments/assets/95c5b9c8-ca45-4d0d-826d-fb1ef053a832" />


### 🗄️ Database Schema 
## 📸 Screenshots

### 🎬 Movie Recommendations History
<img width="1366" height="768" alt="Screenshot (265)" src="https://github.com/user-attachments/assets/67ea7d51-364a-495b-91fa-282261b0fc7c" />

### ⭐ User Reviews with Sentiment Analysis
<img width="1366" height="768" alt="Screenshot (266)" src="https://github.com/user-attachments/assets/5f4bbadc-5d08-4af4-9b54-cd22d9f9793d" />

### 👤 User Accounts
<img width="1366" height="768" alt="Screenshot (268)" src="https://github.com/user-attachments/assets/096e4c58-bddf-4f4c-80cc-3a21a85a1a82" />

### 🔍 Search History
<img width="1366" height="768" alt="Screenshot (267)" src="https://github.com/user-attachments/assets/9811ff8a-c9a2-4896-808a-2dae8774508b" />

---

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Thomasayanfeoluwa/CosineSimilarity-Recommender-System
    cd CosineSimilarity-Recommender-System
    ```

2.  **Create a Virtual Environment**
    ```bash
    conda create -n rec_sys python=3.11
    conda activate rec_sys
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory:
    ```env
    TMDB_API_KEY=your_tmdb_api_key
    DATABASE_URL=postgresql://user:password@localhost/dbname
    FLASK_SECRET_KEY=your_secret_key
    ```

5.  **Initialize the Database**
    ```bash
    flask db init
    flask db migrate
    flask db upgrade
    ```

6.  **Run the Application**
    ```bash
    flask run
    ```
    Access the app at   <p align="center">
  <a href="https://movies-recommender-system-kxlg.onrender.com/home">
    <img src="https://img.shields.io/badge/LIVE-DEMO-red?style=for-the-badge&logo=render">
  </a>
</p>

---

## 📄 License

This project is licensed under the MIT License.
