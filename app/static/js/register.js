// Import Firebase services
import { auth } from './firebaseInit.js';
import { createUserWithEmailAndPassword } from 'https://www.gstatic.com/firebasejs/9.6.10/firebase-auth.js';

document.addEventListener('DOMContentLoaded', () => {
    console.log("register.js loaded");

    document.getElementById('register-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const email = e.target.email.value;
        const password = e.target.password.value;

        console.log(`Registration attempt with email: ${email}`);
        console.log('Auth object:', auth);

        try {
            const userCredential = await createUserWithEmailAndPassword(auth, email, password);
            console.log('User registered:', userCredential.user);
            // Redirect to login after successful registration
            window.location.href = '/login';
        } catch (error) {
            console.error('Error registering user:', error);
            alert('Error registering user: ' + error.message);
        }
    });
});
