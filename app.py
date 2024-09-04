from flask import Flask, render_template, request
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the datasets
ratings = pd.read_csv(
    'ml-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    encoding='latin-1'
)

movies = pd.read_csv(
    'ml-100k/u.item',
    sep='|',
    names=[
        'item_id', 'title', 'release_date', 'video_release_date',
        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ],
    encoding='latin-1'
)

# Merge ratings with movie titles and genres
ratings = pd.merge(ratings, movies[['item_id', 'title']], on='item_id')

# Create a user-movie matrix
user_movie_matrix = ratings.pivot_table(index='user_id', columns='title', values='rating')

# Compute the correlation matrix for similarity-based recommendations
movie_similarity = user_movie_matrix.corr(method='pearson', min_periods=100)

# Function to get genre-based recommendations
def get_genre_recommendations(movie_name, movie_df, num_recommendations=5):
    if movie_name not in movie_df['title'].values:
        print(f"Movie '{movie_name}' not found in the dataset.")
        return pd.Series([])

    # Extract the genre information for the given movie
    genres = movie_df[movie_df['title'] == movie_name].iloc[0, 5:]
    genres = genres[genres == 1].index.tolist()

    if not genres:
        print(f"No genres found for movie '{movie_name}'.")
        return pd.Series([])

    # Find movies that share at least one genre with the given movie
    genre_mask = movie_df[genres] == 1
    genre_match = genre_mask.any(axis=1)

    # Exclude the input movie itself
    genre_recommendations = movie_df[genre_match & (movie_df['title'] != movie_name)]

    # Select the top N recommendations (you can define your own logic here)
    # For simplicity, we'll take the first N matching movies
    genre_recommendations = genre_recommendations['title'].head(num_recommendations)

    return genre_recommendations

# Function to get movie recommendations based on ratings similarity
def get_similarity_recommendations(movie_name, similarity_matrix, num_recommendations=5):
    if movie_name not in similarity_matrix.index:
        print(f"Movie '{movie_name}' not found in similarity matrix.")
        return pd.Series([])

    # Get similar movies sorted by similarity score
    similar_movies = similarity_matrix[movie_name].dropna().sort_values(ascending=False).head(num_recommendations)

    return similar_movies.index

# Function to combine similarity and genre-based recommendations
def get_combined_recommendations(movie_name, similarity_matrix, movie_df, num_recommendations=5):
    similarity_recs = get_similarity_recommendations(movie_name, similarity_matrix, num_recommendations)
    genre_recs = get_genre_recommendations(movie_name, movie_df, num_recommendations)

    # Combine recommendations, giving priority to similarity-based
    combined_recs = list(similarity_recs) + [rec for rec in genre_recs if rec not in similarity_recs]

    # Limit to the desired number of recommendations
    combined_recs = combined_recs[:num_recommendations]

    return combined_recs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie']
    recommendations = get_combined_recommendations(movie_name, movie_similarity, movies, num_recommendations=10)

    if not recommendations:
        message = f"No recommendations found for '{movie_name}'. Please try another movie."
        return render_template('recommendations.html', message=message, movie_name=movie_name)

    return render_template('recommendations.html', recommendations=recommendations, movie_name=movie_name)

if __name__ == '__main__':
    app.run(debug=True)
