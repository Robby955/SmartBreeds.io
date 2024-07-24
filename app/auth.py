# app/auth.py
from flask import request
from flask_restful import Resource
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('path/to/your/firebase_credentials.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

class Register(Resource):
    def post(self):
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')

        hashed_password = generate_password_hash(password, method='sha256')

        user_ref = db.collection('users').document(email)
        if user_ref.get().exists:
            return {'message': 'User already exists'}, 400

        user_ref.set({
            'name': name,
            'email': email,
            'password': hashed_password
        })

        return {'message': 'User registered successfully'}, 201

class Login(Resource):
    def post(self):
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        user_ref = db.collection('users').document(email)
        user = user_ref.get()
        if not user.exists:
            return {'message': 'User not found'}, 404

        user_data = user.to_dict()
        if not check_password_hash(user_data['password'], password):
            return {'message': 'Incorrect password'}, 401

        access_token = create_access_token(identity={'email': email, 'name': user_data['name']})
        return {'access_token': access_token}, 200
