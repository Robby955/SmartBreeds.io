from flask import Flask, render_template
from flask_login import login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import uuid
from datetime import datetime
from flask_login import UserMixin
from models import User, Player, db, HighScore
from forms import LoginForm, RegistrationForm
from scoreboard import scoreboard_bp
from extensions import db, login_manager
import logging
from flask import redirect, url_for, flash
from flask_login import current_user
from models import ImageResponse, Feedback  # Import the ImageResponse model if not already imported
from flask_login import login_required
import re
import os
from io import StringIO
from users import users_bp
from functools import wraps

from dotenv import load_dotenv
load_dotenv()
# Debug prints for environment variables
print("Loaded environment variables:")
print(f"SECRET_KEY: {os.getenv('SECRET_KEY')}")
print(f"SQLALCHEMY_DATABASE_URI: {os.getenv('SQLALCHEMY_DATABASE_URI')}")
print(f"SQLALCHEMY_BINDS_HIGHSCORES: {os.getenv('SQLALCHEMY_BINDS_HIGHSCORES')}")

from flask_login import current_user
from flask import abort

# Firebase configuration
firebase_config = {
    'apiKey': os.getenv('FIREBASE_API_KEY'),
    'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
    'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    'appId': os.getenv('FIREBASE_APP_ID')
}

print("Firebase configuration loaded:")
print(firebase_config)
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)  # Forbidden access
        return f(*args, **kwargs)
    return decorated_function

from flask_login import LoginManager, current_user
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')

# Database configuration
base_dir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(base_dir, 'mydata.db')

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite:///mydata.db')
app.config['SQLALCHEMY_BINDS'] = {
    'highscores': os.getenv('SQLALCHEMY_BINDS_HIGHSCORES', 'sqlite:///highscores.db')
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

print("App configurations:")
print(app.config['SQLALCHEMY_DATABASE_URI'])
print(app.config['SQLALCHEMY_BINDS'])




# Register the game blueprint with the app
from game import game_bp
app.register_blueprint(game_bp, url_prefix='/game')
app.register_blueprint(scoreboard_bp)
app.register_blueprint(users_bp, url_prefix='/users')
from flask_migrate import Migrate
migrate = Migrate(app, db)



from google.cloud import storage

# Initialize a storage client
storage_client = storage.Client()
bucket_name = 'breed_images'

db.init_app(app)  # Initialize db with the app
login_manager.init_app(app)  # Initialize login_manager with the app
login_manager.login_view = 'login'





import firebase_admin
from firebase_admin import credentials, auth

cred = credentials.Certificate('./credentials.json')
firebase_admin.initialize_app(cred)


from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

admin = Admin(app, name='Smart Dog Breed Classifier', template_mode='bootstrap3')
# Add views for each model
admin.add_view(ModelView(User, db.session))
admin.add_view(ModelView(ImageResponse, db.session))
admin.add_view(ModelView(Player, db.session))
admin.add_view(ModelView(HighScore, db.session))
admin.add_view(ModelView(Feedback, db.session))

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo

@app.route('/leaderboard')
def leaderboard():
    top_scores = HighScore.query.order_by(HighScore.score.desc()).limit(10).all()
    return render_template('leaderboard.html', top_scores=top_scores)


def upload_file_to_gcs(bucket_name, file, predicted_label, user_label):
    print(f"Function called: upload_file_to_gcs with predicted_label: {predicted_label}")

    unique_id = str(uuid.uuid4())
    # Try to match the n[number]-breed format first
    match = re.search(r'n\d+-([a-zA-Z_]+)', predicted_label)
    if not match:
        # If the first pattern doesn't match, try to match any word character sequence
        match = re.search(r'([a-zA-Z_]+)', predicted_label)

    breed_name = match.group(1).replace('_', '-') if match else 'unknown_breed'

    destination_blob_name = f"train/train/{breed_name}/userdefined_{user_label}/{unique_id}-{secure_filename(file.filename)}"
    print(f"Uploading to: {destination_blob_name}")  # Log the destination path

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file)

    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"





class ImageResponse(db.Model):
    __tablename__ = 'image_response'  # Specify the table name explicitly
    __table_args__ = {'extend_existing': True}  # Extend existing table if it exists

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    predicted_breed = db.Column(db.String(50))  # Add column for predicted breed
    user_answer = db.Column(db.String(50))  # Add column for user's answer
    uploaded_at = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())


@app.route('/upload_image_form')
def upload_image_form():
    print("Submit image route accessed")  # Add this line
    return render_template('submit_image_form.html')


@app.route('/submit_image', methods=['POST'])
def submit_image():
    if 'image' not in request.files:
        flash('No image part')
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    try:
        # Create a unique "blob name" for the file in GCS
        destination_blob_name = f"breed_images/{uuid.uuid4()}-{file.filename}"

        # Upload the file to GCS
        predicted_label = request.form.get('predicted_breed')
        user_label = request.form.get('user_answer')

        # Update the function call with the new arguments
        file_url = upload_file_to_gcs('breed_images', file, predicted_label, user_label)
        predicted_breed = request.form.get('predicted_breed')
        user_answer = request.form.get('user_answer')

        # Save the GCS URL of the file instead of the local path
        response = ImageResponse(user_id=current_user.id, file_path=file_url, predicted_breed=predicted_breed, user_answer=user_answer)
        db.session.add(response)
        db.session.commit()

        flash('Image and response saved successfully')
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error in submit_image: {str(e)}")
        flash(f'An error occurred: {e}', 'error')
        return redirect(request.url)


# Set up basic logging
logging.basicConfig(level=logging.INFO)
class ClassificationSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_time = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/submit_classification', methods=['POST'])
def submit_classification():
    if current_user.is_authenticated:
        current_user.classification_count += 1
        db.session.commit()
    try:
        logging.info("Accessed submit_classification route")
        submission = ClassificationSubmission()
        db.session.add(submission)
        db.session.commit()
        flash('Classification submitted successfully!', 'success')
        logging.info("Classification submitted successfully")
    except Exception as e:
        logging.error(f"Failed to submit classification: {e}")
        flash('Failed to submit classification.', 'error')
    return redirect(url_for('index'))



# Define User model
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    classification_count = db.Column(db.Integer, default=0)

class UserResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.String(64), nullable=False)  # Assuming each image has a unique identifier
    actual_breed = db.Column(db.String(120), nullable=False)
    user_confirmed = db.Column(db.Boolean, nullable=False)  # True if user confirmed, False if corrected

# Create database tables
with app.app_context():
    db.create_all()


@app.route('/confirm_breed', methods=['POST'])
def confirm_breed():
    if not current_user.is_authenticated:
        return jsonify({'error': 'User not authenticated'}), 401

    data = request.json
    image_id = data.get('image_id')
    actual_breed = data.get('actual_breed')
    user_confirmed = data.get('user_confirmed', True)  # Default to True if not provided

    user_response = UserResponse(image_id=image_id, actual_breed=actual_breed, user_confirmed=user_confirmed)
    db.session.add(user_response)
    db.session.commit()

    return jsonify({'message': 'Response saved successfully'}), 200


@login_manager.user_loader
def load_user(user_id):
    # Use the session directly from SQLAlchemy to comply with the new standards
    return db.session.get(User, int(user_id))



@app.route('/view_responses')
@login_required
def view_responses():
    try:
        responses = ImageResponse.query.filter_by(user_id=current_user.id).all()
        return render_template('view_responses.html', responses=responses)
    except Exception as e:
        app.logger.error(f"Error fetching responses: {e}")
        flash('Error fetching responses.', 'error')
        return redirect(url_for('index'))




breed_aliases = {
    'Cardigan': 'Corgi',
    'Pembroke': 'Corgi',
    'Chow': 'Chow Chow',
    'Basset': 'Basset Hound',
    'Boston Bull': 'Boston Terrier',
    'Cairn':'Cairn Terrier',
    'Blenheim':'Cavalier King Charles Spaniel'
}




file_path = 'breed_names.txt'

# Read the breed names from the file
with open(file_path, 'r') as file:
    breed_names = [line.strip() for line in file.readlines()]


breed_names = [breed.capitalize() for breed in breed_names]

folder_names=['n02085620-Chihuahua', 'n02085782-Japanese_spaniel', 'n02085936-Maltese_dog', 'n02086079-Pekinese', 'n02086240-Shih-Tzu', 'n02086646-Blenheim_spaniel', 'n02086910-papillon', 'n02087046-toy_terrier', 'n02087394-Rhodesian_ridgeback', 'n02088094-Afghan_hound', 'n02088238-basset', 'n02088364-beagle', 'n02088466-bloodhound', 'n02088632-bluetick', 'n02089078-black-and-tan_coonhound', 'n02089867-Walker_hound', 'n02089973-English_foxhound', 'n02090379-redbone', 'n02090622-borzoi', 'n02090721-Irish_wolfhound', 'n02091032-Italian_greyhound', 'n02091134-whippet', 'n02091244-Ibizan_hound', 'n02091467-Norwegian_elkhound', 'n02091635-otterhound', 'n02091831-Saluki', 'n02092002-Scottish_deerhound', 'n02092339-Weimaraner', 'n02093256-Staffordshire_bullterrier', 'n02093428-American_Staffordshire_terrier', 'n02093647-Bedlington_terrier', 'n02093754-Border_terrier', 'n02093859-Kerry_blue_terrier', 'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier', 'n02094258-Norwich_terrier', 'n02094433-Yorkshire_terrier', 'n02095314-wire-haired_fox_terrier', 'n02095570-Lakeland_terrier', 'n02095889-Sealyham_terrier', 'n02096051-Airedale', 'n02096177-cairn', 'n02096294-Australian_terrier', 'n02096437-Dandie_Dinmont', 'n02096585-Boston_bull', 'n02097298-Scotch_terrier', 'n02097474-Tibetan_terrier', 'n02097658-silky_terrier', 'n02098105-soft-coated_wheaten_terrier', 'n02098286-West_Highland_white_terrier', 'n02098413-Lhasa', 'n02099267-flat-coated_retriever', 'n02099429-curly-coated_retriever', 'n02099601-golden_retriever', 'n02099712-Labrador_retriever', 'n02099849-Chesapeake_Bay_retriever', 'n02100236-German_short-haired_pointer', 'n02100583-vizsla', 'n02100735-English_setter', 'n02100877-Irish_setter', 'n02101006-Gordon_setter', 'n02101388-Brittany_spaniel', 'n02101556-clumber', 'n02102040-English_springer', 'n02102177-Welsh_springer_spaniel', 'n02102318-cocker_spaniel', 'n02102480-Sussex_spaniel', 'n02102973-Irish_water_spaniel', 'n02104029-kuvasz', 'n02104365-schipperke', 'n02105056-groenendael', 'n02105162-malinois', 'n02105251-briard', 'n02105412-kelpie', 'n02105505-komondor', 'n02105641-Old_English_sheepdog', 'n02105855-Shetland_sheepdog', 'n02106030-collie', 'n02106166-Border_collie', 'n02106382-Bouvier_des_Flandres', 'n02106550-Rottweiler', 'n02106662-German_shepherd', 'n02107142-Doberman', 'n02107312-miniature_pinscher', 'n02107574-Greater_Swiss_Mountain_dog', 'n02107683-Bernese_mountain_dog', 'n02107908-Appenzeller', 'n02108000-EntleBucher', 'n02108089-boxer', 'n02108422-bull_mastiff', 'n02108551-Tibetan_mastiff', 'n02108915-French_bulldog', 'n02109047-Great_Dane', 'n02109525-Saint_Bernard', 'n02109961-Eskimo_dog', 'n02110063-malamute', 'n02110627-affenpinscher', 'n02110806-basenji', 'n02110958-pug', 'n02111129-Leonberg', 'n02111277-Newfoundland', 'n02111500-Great_Pyrenees', 'n02111889-Samoyed', 'n02112018-Pomeranian', 'n02112137-chow', 'n02112350-keeshond', 'n02112706-Brabancon_griffon', 'n02113023-Pembroke', 'n02113186-Cardigan', 'n02113978-Mexican_hairless', 'n02115641-dingo', 'n02115913-dhole', 'n02116738-African_hunting_dog', 'n02rob-poodle', 'n02rob-schnauzer']
def extract_breed(folder_name):
    # This regex matches the pattern 'n' followed by any number of digits and a hyphen, and captures the subsequent text.
    match = re.search(r'n\d+-(.*)', folder_name)
    if match:
        # Replace underscores with hyphens and return the breed name in lowercase.
        return match.group(1).replace('_', '-').lower()
    return None
# Example usage
breed_mapping = {folder: extract_breed(folder) for folder in folder_names}
#print(breed_mapping)


@app.route('/breed_names')
def get_breed_names():
    try:
        # Your logic here...
        return jsonify(breed_names)
    except Exception as e:
        app.logger.error(f"Failed to fetch breed names: {e}")
        return jsonify(error=str(e)), 500


# User authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        user = User(username=form.username.data, email=form.email.data, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html', title='Sign In', form=form)


@app.route('/submit_login', methods=['POST'])
def submit_login():
    username = request.form['username']
    password = request.form['password']

    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password_hash, password):
        login_user(user)
        return redirect(url_for('index'))  # Redirect to the homepage or dashboard after login

    flash('Invalid username or password')
    return redirect(url_for('login'))  # Redirect back to the login page if authentication fails

# Dog breed classification routes
@app.route('/')
def index():
    return render_template('index.html')


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.85)(x)
predictions = Dense(115, activation='softmax')(x)  # Replace num_classes with actual number
model = Model(inputs=base_model.input, outputs=predictions)
# Model and utility functions


weights_path = os.path.join(base_dir, 'models', 'best_model_weights.weights.h5')

# Load the model weights
model.load_weights(weights_path)
from flask import request, jsonify
from image_preprocessing import crop_dog, load_model
import os
import tempfile
from werkzeug.utils import secure_filename

# Load the SSD model

destination_dir = 'downloaded_model'
# Load the model from the destination directory
ssd_model = load_model(destination_dir)
logging.basicConfig(level=logging.DEBUG)


# Assuming classification_model is loaded elsewhere
from flask import jsonify
import base64
from PIL import Image
import io

@app.route('/classify', methods=['POST'])
def classify():
    logging.info("Classify route accessed!")
    try:
        if 'file' not in request.files:
            logging.error('No file part in the request.')
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.error('No selected file.')
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file temporarily
        filepath = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(filepath)
        logging.info(f'File saved temporarily at {filepath}')

        # Load and crop the image using the SSD model
        original_image, cropped_image = crop_dog(filepath, ssd_model)

        if cropped_image is None:
            logging.error('No dog found in the image.')
            return jsonify({'error': 'No dog found in the image'}), 400

        # Convert the cropped image to a format suitable for JSON response
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Resize the cropped image for the classification model
        cropped_image_resized = cropped_image.resize((224, 224))
        image_array = np.array(cropped_image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Run the classification model
        predictions = model.predict(image_array)
        predicted_breed_index = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_breed = breed_names[predicted_breed_index]  # Ensure breed_names is defined

        # Return the prediction, confidence, and cropped image
        return jsonify({
            'prediction': predicted_breed,
            'confidence': float(confidence),
            'croppedImage': f"data:image/jpeg;base64,{img_str}"
        })
    except Exception as e:
        logging.error(f"Error during classification: {str(e)}")
        return jsonify({'error': f'Error occurred during classification: {str(e)}'}), 500


@app.route('/responses')
def responses():
    try:
        responses = Response.query.all()  # Confirm Response model exists and is correctly defined
        return render_template('responses.html', responses=responses)
    except Exception as e:
        app.logger.error(f"Error fetching responses: {e}")
        flash('Error fetching responses.')
        return redirect(url_for('index'))  # Redirect to a safe page
@app.route('/submit_response', methods=['POST'])
def submit_response():
    if 'image' not in request.files:
        flash('No image part')
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    try:
        # Generate a secure filename and create a unique blob name for the file in GCS
        filename_secure = secure_filename(file.filename)
        destination_blob_name = f"breed_images/{uuid.uuid4()}-{filename_secure}"

        predicted_label = request.form.get('predicted_breed')
        user_label = request.form.get('user_answer')

        # Check if predicted_label and user_label are not None
        if not predicted_label or not user_label:
            flash('Missing predicted label or user answer')
            return redirect(request.url)

        # Upload the file to GCS and get the file URL
        file_url = upload_file_to_gcs('breed_images', file, predicted_label, user_label)

        # Save the file URL and other response details in the database
        response = ImageResponse(user_id=current_user.id, file_path=file_url, predicted_breed=predicted_label, user_answer=user_label)
        db.session.add(response)
        db.session.commit()

        flash('Image and response saved successfully')
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Error in submit_response: {str(e)}")
        flash(f'An error occurred: {e}', 'error')
        return redirect(request.url)
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    return img


@app.route('/save_response', methods=['POST'])
def save_response():
    user_id = request.form.get('user_id')
    breed_correct = request.form.get('breed_correct')
    breed_chosen = request.form.get('breed_chosen')

    response = GameResponse(user_id=user_id, breed_correct=breed_correct, breed_chosen=breed_chosen)
    db.session.add(response)
    db.session.commit()

    return jsonify({'status': 'success'})


@app.route('/submit_registration', methods=['POST'])
def submit_registration():
    form = RegistrationForm(request.form)
    if form.validate_on_submit():
        existing_user = User.query.filter(
            (User.username == form.username.data) | (User.email == form.email.data)).first()
        if existing_user:
            flash('A user with that username or email already exists.', 'error')
            return render_template('register.html', form=form)

        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        user = User(username=form.username.data, email=form.email.data, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in the {getattr(form, field).label.text} field - {error}", 'error')

    return render_template('register.html', form=form)


@app.route('/score_board')
def scoreboard():
    # Logic to fetch top players' scores from the database
    top_players = Player.query.order_by(Player.score.desc()).limit(10).all()  # Example query, adjust as needed
    return render_template('scoreboard.html', top_players=top_players)


@app.route('/dashboard')
@login_required
def dashboard():
    total_uploads = ImageResponse.query.filter_by(user_id=current_user.id).count()
    total_classifications = len(
        set([response.predicted_breed for response in ImageResponse.query.filter_by(user_id=current_user.id).all()]))
    return render_template('dashboard.html', total_uploads=total_uploads, total_classifications=total_classifications, user=current_user)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    # Your logic to update the user profile
    # For example, you might retrieve form data and update the database
    return redirect(url_for('dashboard'))  # Redirect back to the dashboard after updating



@app.route('/delete_account', methods=['POST'])
@login_required  # Make sure only logged-in users can delete their account
def delete_account():
    # Logic to delete the user's account
    # For example, you might mark the user as inactive or remove their data from the database
    return redirect(url_for('index'))  # Redirect to the home page or a confirmation page after deletion

@app.route('/privacy_policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/about')
def about():
    return render_template('about.html')


def download_confusion_matrix(bucket_name, source_blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_string().decode('utf-8')

def download_blob_to_dataframe(bucket_name, source_blob_name):
    """Downloads a blob from the bucket and loads it into a Pandas DataFrame."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_string()
    df = pd.read_csv(StringIO(data.decode('utf-8')))
    return df

@app.route('/api/confusion_matrix')
def get_confusion_matrix():
    bucket_name = 'predict-breed-models'
    source_blob_name = 'confusion_matrix_20240403-204919.csv'
    try:
        logging.debug("Downloading confusion matrix from GCS.")
        df = download_blob_to_dataframe(bucket_name, source_blob_name)
        logging.debug("Confusion matrix loaded successfully.")
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        logging.error("Error loading confusion matrix: %s", e)
        return jsonify({'error': str(e)}), 500

def find_most_misclassified(confusion_matrix_csv):
    # Read the confusion matrix into a DataFrame
    df = pd.read_csv(pd.compat.StringIO(confusion_matrix_csv), index_col=0)

    # Assuming the rows represent the actual labels and the columns represent the predicted labels
    # Subtract the diagonal (correct predictions) from the sum of rows to get total misclassifications
    total_misclassified = df.sum(axis=1) - df.values.diagonal()
    most_misclassified_breed = total_misclassified.idxmax()
    misclassification_count = total_misclassified.max()

    # Find the breed which this breed is most often confused with
    most_confused_with = df.loc[most_misclassified_breed].idxmax()

    return most_misclassified_breed, most_confused_with, misclassification_count


# Download the confusion matrix
confusion_matrix_csv = download_confusion_matrix('predict-breed-models', 'confusion_matrix_20240403-204919.csv')


@app.route('/users')
@admin_required
def users():
    # Add any required logic to check for user authentication or authorization
    return render_template('users.html')



@app.route('/api/users')
@admin_required
def get_users():
    # Implement authentication and authorization checks as needed
    users = User.query.all()  # Assuming you're using SQLAlchemy
    users_list = [{'id': user.id, 'username': user.username, 'email': user.email} for user in users]
    return jsonify(users_list)


logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

@app.route('/api/results')
def api_results():
    file_path = 'result_summary_20240403-204919.txt'

    try:
        # Load your data
        results_df = pd.read_csv(file_path, sep=r'\s{2,}', engine='python')
        logging.debug("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return jsonify({'error': 'Failed to load data'}), 500

    try:
        # Convert DataFrame to a list of dictionaries for JSON conversion
        results_data = results_df.to_dict(orient='records')
        logging.debug("Data converted to JSON successfully.")
    except Exception as e:
        logging.error(f"Error converting data to JSON: {e}")
        return jsonify({'error': 'Failed to convert data to JSON'}), 500

    # Logging the data for debugging (Consider removing or truncating for large datasets)
    logging.debug(f"Results data: {results_data[:5]}")  # Log the first 5 records to check

    return jsonify(results_data)


@app.route('/breed-results')
def breed_results():
    return render_template('breed_results.html')

breed_names=['Affenpinscher', 'Afghan Hound', 'African Hunting Dog', 'Airedale', 'American Staffordshire Terrier', 'Appenzeller', 'Australian Terrier', 'Basenji', 'Basset', 'Beagle', 'Bedlington Terrier', 'Bernese Mountain Dog', 'Black-and-tan Coonhound', 'Blenheim Spaniel', 'Bloodhound', 'Bluetick', 'Border Collie', 'Border Terrier', 'Borzoi', 'Boston Bull', 'Bouvier Des Flandres', 'Boxer', 'Brabancon Griffon', 'Briard', 'Brittany Spaniel', 'Bull Mastiff', 'Cairn', 'Cardigan', 'Chesapeake Bay Retriever', 'Chihuahua', 'Chow', 'Clumber', 'Cocker Spaniel', 'Collie', 'Curly-coated Retriever', 'Dandie Dinmont', 'Dhole', 'Dingo', 'Doberman', 'English Foxhound', 'English Setter', 'English Springer', 'Entlebucher', 'Eskimo Dog', 'Flat-coated Retriever', 'French Bulldog', 'German Shepherd', 'German Short-haired Pointer', 'Golden Retriever', 'Gordon Setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain Dog', 'Groenendael', 'Ibizan Hound', 'Irish Setter', 'Irish Terrier', 'Irish Water Spaniel', 'Irish Wolfhound', 'Italian Greyhound', 'Japanese Spaniel', 'Keeshond', 'Kelpie', 'Kerry Blue Terrier', 'Komondor', 'Kuvasz', 'Labrador Retriever', 'Lakeland Terrier', 'Leonberg', 'Lhasa', 'Malamute', 'Malinois', 'Maltese Dog', 'Mexican Hairless', 'Miniature Pinscher', 'Newfoundland', 'Norfolk Terrier', 'Norwegian Elkhound', 'Norwich Terrier', 'Old English Sheepdog', 'Otterhound', 'Papillon', 'Pekinese', 'Pembroke', 'Pomeranian', 'Poodle', 'Pug', 'Redbone', 'Rhodesian Ridgeback', 'Rottweiler', 'Saint Bernard', 'Saluki', 'Samoyed', 'Schipperke', 'Schnauzer', 'Scotch Terrier', 'Scottish Deerhound', 'Sealyham Terrier', 'Shetland Sheepdog', 'Shih-tzu', 'Silky Terrier', 'Soft-coated Wheaten Terrier', 'Staffordshire Bullterrier', 'Sussex Spaniel', 'Tibetan Mastiff', 'Tibetan Terrier', 'Toy Terrier', 'Vizsla', 'Walker Hound', 'Weimaraner', 'Welsh Springer Spaniel', 'West Highland White Terrier', 'Whippet', 'Wire-haired Fox Terrier', 'Yorkshire Terrier']
class_labels=breed_names
from flask import Flask, render_template
import pandas as pd
import urllib.parse

@app.route('/api/most_common_misclassifications')
def most_common_misclassifications():
    csv_path = 'most_common_misclassifications.csv'
    try:
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Function to generate signed URL
def generate_signed_url(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(expiration=datetime.utcnow() + timedelta(hours=1), method='GET')
    return url
import requests


def get_breed_image_urls(breed_name):
    # Assuming breed_name is the folder name in the GCS bucket
    # Replace 'your_bucket_name' with your actual GCS bucket name
    base_url = f"https://storage.googleapis.com/breed_images/{breed_name}/"
    # Example URL: https://storage.googleapis.com/breed_images/Dingo/n02115641_10261_cropped.jpg

    # Fetch the list of image URLs for the breed.
    # You need to implement this part based on how you organize your GCS bucket.
    # The following line is a placeholder for the logic to retrieve image URLs.
    response = requests.get(f"https://storage.googleapis.com/storage/v1/b/breed_images/o?prefix={breed_name}/")
    if response.status_code == 200:
        items = response.json().get('items', [])
        image_urls = [f"{base_url}{item['name'].split('/')[-1]}" for item in items if item['name'].endswith('.jpg')]
        return image_urls
    return []












# Function to fetch image URLs directly without signing them
from urllib.parse import quote
from google.cloud import storage
import random
import logging

import logging

from urllib.parse import quote_plus


from urllib.parse import quote

def format_breed_name(breed_name):
    """Format the breed name to match the GCS folder structure."""
    parts = breed_name.split()
    if len(parts) > 1:
        # Capitalize the first word and lower the others if there are multiple words
        return parts[0].capitalize() + ' ' + ' '.join(word.lower() for word in parts[1:])
    else:
        # Capitalize the whole word if it's a single-word breed
        return breed_name.capitalize()


def format_test_breed_name(breed_name):
    """Format the breed name to match the GCS folder structure for test data."""
    # Capitalize each part of the breed name
    return ' '.join(word.capitalize() for word in breed_name.split())


def get_images_for_breed(breed_name, data_type='all'):
    """Retrieve image URLgets for a given breed from GCS, either all, training, or test images."""
    logging.debug(f"Fetching images for breed: {breed_name}, Data type: {data_type}")

    print(f"Fetching images for breed: {breed_name}, Data type: {data_type}")


    bucket_name = 'breed_images'


    if data_type=='train' or data_type=='all':
         formatted_breed_name = format_breed_name(breed_name)
    else:
        formatted_breed_name = format_test_breed_name(breed_name)

    if data_type == 'test':
        prefix = f"test_data/{formatted_breed_name}"
    else:
        prefix = f"{formatted_breed_name}"
        bucket_name = 'breed_images'

    logging.debug(f"Looking in bucket '{bucket_name}' with prefix '{prefix}'")
    print(f"Looking in bucket '{bucket_name}' with prefix '{prefix}'")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logging.warning(f"No blobs found for {formatted_breed_name} with data type {data_type}.")
        print(f"No blobs found for {formatted_breed_name} with data type {data_type}.")
        return []

    image_urls = [f"https://storage.googleapis.com/{bucket_name}/{quote(blob.name, safe=':/')}" for blob in blobs if blob.name.endswith('.jpg')]

    logging.debug(f"Found image URLs: {image_urls}")
    print(f"Found image URLs: {image_urls}")

    return image_urls
import json
import random
import logging
from flask import render_template, abort
from google.cloud import storage

# Initialize the storage client globally
storage_client = storage.Client()

import json
import random
import logging
from flask import render_template, abort
from google.cloud import storage

# Initialize the storage client globally
storage_client = storage.Client()

import json
import random
import logging
from flask import render_template, abort
from google.cloud import storage

# Initialize the storage client globally
storage_client = storage.Client()

import json
import random
import logging
from flask import render_template, abort

import json
import random
import logging
from flask import render_template, abort
from google.cloud import storage
from urllib.parse import unquote

# Initialize the storage client globally
storage_client = storage.Client()

@app.route('/breed/<path:breed_name>')
def breed_detail(breed_name):
    logging.debug(f"Fetching details for breed: {breed_name}")

    # Load the upgraded results dictionary from GCS
    try:
        bucket = storage_client.get_bucket('predict-breed-models')
        predictions_blob = bucket.blob('upgraded_results.json')
        all_predictions = json.loads(predictions_blob.download_as_string())['results']
        # Modify keys to be URL decoded to match images displayed on the webpage
        predictions_for_breed = {
            unquote(p['image_url']).replace('gs://breed_images/', 'https://storage.googleapis.com/breed_images/'): p
            for p in all_predictions if p['true_name'] == breed_name
        }
        logging.debug(f"Predictions loaded for {breed_name}: {predictions_for_breed}")
    except Exception as e:
        logging.error(f"Failed to load predictions: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")  # This will give the full stack trace
        abort(500, description="Internal server error")

    # Fetch training images
    train_images = get_images_for_breed(breed_name)
    selected_train_images = random.sample(train_images, min(len(train_images), 4)) if train_images else []

    # Fetch test images and associate them with predictions
    test_images = get_images_for_breed(breed_name, data_type='test')
    selected_test_images = random.sample(test_images, min(len(test_images), 4)) if test_images else []

    test_images_with_predictions = []
    for img in selected_test_images:
        img_key = unquote(img)  # Decode URL to match with predictions dictionary
        prediction = predictions_for_breed.get(img_key, {'pred_name': 'No prediction'})
        test_images_with_predictions.append({
            'url': img,
            'prediction': prediction['pred_name']
        })

    if not selected_train_images and not test_images_with_predictions:
        logging.warning(f"No images found at all for {breed_name}.")
        return "No images found", 404

    also_known_as = breed_aliases.get(breed_name, '')

    return render_template('breed_detail.html',
                           breed_name=breed_name,
                           also_known_as=also_known_as,
                           image_urls=selected_train_images,
                           test_images_with_predictions=test_images_with_predictions)









# Utility function to fetch image names from GCS
def get_images_from_gcs(bucket_name, folder_name):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_name, delimiter='/')
    return [os.path.basename(blob.name) for blob in blobs if not blob.name.endswith('/')]






'''
    # Function to list files in a GCS bucket folder
    def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
        return [blob.name for blob in blobs]

    # List images for the selected breed and the misclassification
    breed_images = list_blobs_with_prefix('breed_images', f'{breed_name}/')
    misclassified_images = list_blobs_with_prefix('breed_images', f'{misclassification}/')

    # Randomly select a few images
    breed_images = random.sample(breed_images, min(4, len(breed_images)))
    misclassified_images = random.sample(misclassified_images, min(4, len(misclassified_images)))

    # Generate public URLs for the images
    breed_image_urls = [f"https://storage.googleapis.com/{bucket.name}/{image}" for image in breed_images]
    misclassified_image_urls = [f"https://storage.googleapis.com/{bucket.name}/{image}" for image in misclassified_images]

    return render_template('breed_detail.html', breed_name=breed_name, image_urls=breed_image_urls, misclassification=misclassification, misclassified_images=misclassified_image_urls)
'''

#from flask_wtf import CSRFProtect

#csrf = CSRFProtect(app)




from flask import jsonify

@app.route('/api/admin/users')
@admin_required
def admin_get_users():
    try:
        users = User.query.all()
        users_list = []
        for user in users:
            num_classifications = ImageResponse.query.filter_by(user_id=user.id).count()
            users_list.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'num_classifications': num_classifications
            })
        return jsonify(users_list)
    except Exception as e:
        app.logger.error(f"Failed to fetch users: {str(e)}")
        return jsonify({'error': 'Unable to fetch users'}), 500


@app.route('/feedback', methods=['GET'])
def feedback():
    return render_template('feedback.html')


@app.route('/gan_playground', methods=['GET'])
def gan_playground():
    return render_template('gan_playground.html')

@app.route('/terms', methods=['GET'])
def terms():
    return render_template('terms.html')

@app.route('/data_deletion', methods=['GET'])
def data_deletion():
    return render_template('data_deletion.html')


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    name = request.form['name']
    email = request.form['email']
    feedback_text = request.form['feedback']

    feedback = Feedback(name=name, email=email, feedback=feedback_text)
    db.session.add(feedback)
    db.session.commit()

    flash('Thank you for your feedback!', 'success')
    return redirect(url_for('feedback_submitted'))

@app.route('/feedback_submitted')
def feedback_submitted():
    return "Thank you for your feedback!"

@app.route('/view_feedback')
@admin_required
def view_feedback():
    feedback_list = Feedback.query.order_by(Feedback.timestamp.desc()).all()
    return render_template('view_feedback.html', feedback_list=feedback_list)

import os



'''
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8081))
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'  # Ensure that debug mode is off by default
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
'''

base_dir = '/app' if os.path.exists('/app/models') else '.'
model_path = os.path.join(base_dir, 'models', 'best_model_weights.weights.h5')

@app.route('/test-model')
@admin_required
def test_model():
    try:
        # Load model dynamically based on environment
        model = load_model(model_path)
        logging.info("Model loaded successfully.")

        # Perform a dummy prediction (make sure to match the input shape and preprocessing of your actual use case)
        dummy_input = np.random.rand(1, 224, 224, 3)  # Adjust this as per your model's input requirements
        prediction = model.predict(dummy_input)
        return jsonify({'status': 'success', 'prediction': prediction.tolist()})
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/debug/model')
@admin_required
def debug_model():
    # Check if model file exists
    model_exists = os.path.exists(model_path)
    breed_index_path = os.path.join(base_dir, 'models', 'breed_to_index.json')
    index_exists = os.path.exists(breed_index_path)

    return jsonify({
        "model_exists": model_exists,
        "model_path": model_path,
        "index_exists": index_exists,
        "index_path": breed_index_path
    })

@app.route('/user/high_scores')
@login_required
def user_high_scores():
    timed_high_score = HighScore.query.filter_by(user_id=current_user.id, mode='timed').order_by(HighScore.score.desc()).first()
    endless_high_score = HighScore.query.filter_by(user_id=current_user.id, mode='endless').order_by(HighScore.score.desc()).first()

    return jsonify({
        'timed': timed_high_score.score if timed_high_score else 0,
        'endless': endless_high_score.score if endless_high_score else 0
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8081))
    app.run(host='0.0.0.0', port=port, debug=False)