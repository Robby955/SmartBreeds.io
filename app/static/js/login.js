import { auth } from './firebaseInit.js';
import { signInWithEmailAndPassword } from 'https://www.gstatic.com/firebasejs/9.6.10/firebase-auth.js';

document.addEventListener('DOMContentLoaded', () => {
    console.log("login.js loaded");

    document.getElementById('login-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const email = e.target.email.value;
        const password = e.target.password.value;

        console.log(`Login attempt with email: ${email}`);
        console.log('Auth object:', auth);

        try {
            const userCredential = await signInWithEmailAndPassword(auth, email, password);
            console.log('User logged in:', userCredential.user);
            console.log('User ID:', userCredential.user.uid);

            // Set a cookie or local storage item for Flask to recognize the login state
            localStorage.setItem('userToken', userCredential.user.stsTokenManager.accessToken);

            // Redirect to dashboard after successful login
            window.location.href = '/dashboard';
        } catch (error) {
            console.error('Error logging in user:', error);
            alert('Error logging in user: ' + error.message);
        }
    });
});
