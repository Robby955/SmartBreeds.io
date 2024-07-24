from werkzeug.security import generate_password_hash
from models import db, User
from main import app  # Importing app from main.py

def create_admin_user():
    with app.app_context():
        # Check if an admin user already exists
        admin_user = User.query.filter_by(username='admin').first()
        if admin_user:
            print("Admin user already exists.")
        else:
            # Create a new admin user
            hashed_password = generate_password_hash('yourpassword', method='pbkdf2:sha256')
            admin_user = User(
                username='admin',
                email='admin@example.com',
                password_hash=hashed_password,
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created successfully.")

if __name__ == '__main__':
    create_admin_user()
