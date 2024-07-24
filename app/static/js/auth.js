// static/js/auth.js
import { auth } from './firebaseInit.js';

document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const email = e.target.email.value;
    const password = e.target.password.value;

    try {
        const userCredential = await auth.signInWithEmailAndPassword(email, password);
        console.log('User logged in:', userCredential.user);
        // Redirect to dashboard after successful login
        window.location.href = '/dashboard';
    } catch (error) {
        console.error('Error logging in user:', error);
        alert('Error logging in user: ' + error.message);
    }
});
