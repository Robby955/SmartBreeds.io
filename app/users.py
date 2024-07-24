from flask import Blueprint, render_template, redirect, url_for, flash, request, session, jsonify
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, ImageResponse  # Import ImageResponse model
from forms import LoginForm, RegistrationForm
import firebase_admin
from firebase_admin import auth as firebase_auth
import time
users_bp = Blueprint('users', __name__)

@users_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            session['user_id'] = user.id
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template('login.html', form=form)

@users_bp.route('/logout')
def logout():
    logout_user()
    session.pop('user_id', None)
    return redirect(url_for('index'))

@users_bp.route('/firebase_register', methods=['POST'])
def firebase_register():
    data = request.get_json()
    id_token = data.get('idToken')

    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        email = decoded_token['email']
        username = email.split('@')[0]  # Just an example, adjust as needed

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({"message": "User already exists"}), 400

        # Create a new user with the email
        user = User(username=username, email=email)
        db.session.add(user)
        db.session.commit()

        login_user(user)  # Log the user in immediately after registration
        return jsonify({"message": "Registration and login successful"}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 400


@users_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
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
        return redirect(url_for('users.login'))
    return render_template('register.html', form=form)

@users_bp.route('/firebase_login', methods=['POST'])
def firebase_login():
    id_token = request.json.get('idToken')
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            decoded_token = firebase_auth.verify_id_token(id_token)
            user_id = decoded_token['uid']
            user = User.query.filter_by(email=decoded_token['email']).first()
            if not user:
                user = User(username=decoded_token['name'], email=decoded_token['email'])
                db.session.add(user)
                db.session.commit()
            login_user(user)
            return jsonify({'message': 'Logged in successfully'}), 200
        except firebase_auth.InvalidIdTokenError as e:
            if 'Token used too early' in str(e):
                time.sleep(1)  # Wait for 1 second before retrying
                retry_count += 1
                continue
            return jsonify({'message': 'Invalid ID token'}), 401
        except exceptions.FirebaseError as e:
            return jsonify({'message': str(e)}), 401
    return jsonify({'message': 'Token used too early after multiple retries'}), 401

@users_bp.route('/submit_login', methods=['POST'])
def submit_login():
    username = request.form['username']
    password = request.form['password']

    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password_hash, password):
        login_user(user)
        return redirect(url_for('index'))  # Redirect to the main page

    flash('Invalid username or password')
    return redirect(url_for('users.login'))  # Redirect back to the login page if authentication fails



@users_bp.route('/dashboard')
@login_required
def dashboard():
    total_uploads = ImageResponse.query.filter_by(user_id=current_user.id).count()
    total_classifications = len(
        set([response.predicted_breed for response in ImageResponse.query.filter_by(user_id=current_user.id).all()]))
    return render_template('dashboard.html', total_uploads=total_uploads, total_classifications=total_classifications, user=current_user)
