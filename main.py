from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import urllib.request
import bs4 as bs
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('Dataset/processed_dataset/old_main_data.csv')
vectorizer = pickle.load(open('tranformVector.pkl', 'rb'))
model = pickle.load(open('svm_model.pkl', 'rb'))


# Create Similarity Matrix using CountVectorizer and Cosine Similarity
def createSimilarity():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['comb'])
    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    return data, cosine_similarity

# function that takes in movie title as input and returns the top 10 recommended movies
def recomendation(movie):
    movie = movie.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = createSimilarity()
    if movie not in data['movie_title'].unique():
        return (
            'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title'] == movie].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

# function to convert string to list
def converToList(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

# function to get movie suggestions for auto complete
def getSuggestions():
    return list(data['movie_title'].str.capitalize())


# Flask app
app = Flask(__name__)

# Home page route
@app.route("/")
@app.route("/home")
def home():
    suggestions = getSuggestions()
    return render_template('home.html', suggestions=suggestions)

# Auto complete route
@app.route('/autocomplete')
def autocomplete():
    search = request.args.get('search')
    # TODO: Implement movie search logic
    results = [movie for movie in suggestions if search.lower() in movie.lower()]
    return jsonify(results)

# Similarity route and function
@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = recomendation(movie)
    if type(rc) == type('string'):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

# Chatbot page route
@app.route("/chatbot")
def chatbot():
    suggestions = getSuggestions()
    return render_template('chatbot.html', suggestions=suggestions)

# Recommendation page route and function
@app.route("/recommend", methods=["POST"])
def recommend():
    # Using Ajax to get the movie metadada from the webpage
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
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Getting movie suggestions
    suggestions = getSuggestions()

    # Convert string to list (eg. "[1,2,3]" to [1,2,3])
    rec_movies = converToList(rec_movies)
    rec_posters = converToList(rec_posters)
    cast_names = converToList(cast_names)
    cast_chars = converToList(cast_chars)
    cast_profiles = converToList(cast_profiles)
    cast_bdays = converToList(cast_bdays)
    cast_bios = converToList(cast_bios)
    cast_places = converToList(cast_places)

    # Splitting the string to list
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # Replacing the escape characters
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # Creating dictionaries
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in
                    range(len(cast_places))}

    # Getting the reviews using WebScraping
    with urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)) as url:
        sauce = url.read()
    #sauce = request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    # Parsing the html
    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})

    reviews_list = []  #List of reviews
    reviews_status = []  #List of reviews status

    # Getting the reviews
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # Predicting the review status
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = model.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # Creating a dictionary with the movie reviews and their status
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    # Rendering the recommendation page
    return render_template('recommend.html', title=title, poster=poster, overview=overview, release_date=release_date, runtime=runtime, status=status,
                           genres=genres,
                           movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details)

# Running the app
if __name__ == '__main__':
    app.run(debug=True)