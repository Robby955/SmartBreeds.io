import os
import random
import logging
from flask import Blueprint, jsonify, request, render_template, redirect
from google.cloud import storage
from urllib.parse import quote
from flask_login import current_user, login_required
from extensions import db
from models import HighScore, User

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

game_bp = Blueprint('game_bp', __name__)

# Setup Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket('breed_images')

def format_breed_name(breed_name):
    """Format the breed name for consistent bucket access."""
    return ' '.join(word.capitalize() for word in breed_name.split())

def select_random_breeds(num_breeds):
    """Select a random set of breeds from the bucket."""
    blobs = list(bucket.list_blobs(prefix="test_data/"))
    breed_folders = {blob.name.split('/')[1] for blob in blobs if '/' in blob.name and not blob.name.endswith('/')}
    if len(breed_folders) < num_breeds:
        logging.error("Not enough breeds available to select the requested number.")
        return []
    selected_breeds = random.sample(list(breed_folders), num_breeds)
    return selected_breeds

def get_images_for_breed(breed_name, data_type='test'):
    """Retrieve image URLs for a given breed from GCS."""
    formatted_breed_name = format_breed_name(breed_name)
    prefix = f"test_data/{formatted_breed_name}/" if data_type == 'test' else f"{formatted_breed_name}/"
    blobs = bucket.list_blobs(prefix=prefix)
    image_urls = [f"https://storage.googleapis.com/breed_images/{quote(blob.name, safe=':/')}" for blob in blobs if blob.name.endswith('.jpg')]
    if not image_urls:
        logging.warning(f"No images found for breed: {formatted_breed_name}")
    return image_urls

@game_bp.route('/')
def game():
    """Render the game page."""
    return render_template('game.html')

@game_bp.route('/game_data')
def game_data():
    """Provide data necessary for the game."""
    num_breeds = 5
    breeds = select_random_breeds(num_breeds)
    images = {breed: get_images_for_breed(breed) for breed in breeds}
    breed_of_round = random.choice(breeds) if breeds else None

    response = {
        'breeds': breeds,
        'images': images,
        'breedOfRound': breed_of_round
    }
    return jsonify(response)

@game_bp.route('/submit_guess', methods=['POST'])
def submit_guess():
    """Evaluate the user's guess against the correct breed."""
    user_guess = request.form.get('user_guess')
    correct_breed = request.form.get('correct_breed')
    response = {
        'correct': user_guess == correct_breed,
        'message': 'Congrats! You guessed it right!' if user_guess == correct_breed else 'Oops! That is incorrect.'
    }
    return jsonify(response)

@game_bp.route('/submit_score', methods=['POST'])
@login_required
def submit_score():
    data = request.json
    score = data.get('score')
    mode = data.get('mode')
    if score is None or mode not in ['timed', 'endless']:
        return jsonify({'error': 'Invalid score or mode'}), 400

    high_score = HighScore(user_id=current_user.id, score=score, mode=mode)
    db.session.add(high_score)
    db.session.commit()

    return jsonify({'message': 'Score submitted successfully'})

@game_bp.route('/high_scores')
def high_scores():
    mode = request.args.get('mode', 'timed')
    scores = HighScore.query.filter_by(mode=mode).order_by(HighScore.score.desc()).limit(10).all()
    return jsonify([{'username': User.query.get(score.user_id).username, 'score': score.score} for score in scores])
