import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, getRedirectResult } from "firebase/auth";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyD6hFZy4BKOnpmd50iInbJeqmjB_Rqdik4",
  authDomain: "smartdogbreed-416219.firebaseapp.com",
  databaseURL: "https://smartdogbreed-416219-default-rtdb.firebaseio.com",
  projectId: "smartdogbreed-416219",
  storageBucket: "smartdogbreed-416219.appspot.com",
  messagingSenderId: "668071408107",
  appId: "1:668071408107:web:50fbd5a83e974de37269ae"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
const auth = getAuth(app);

document.addEventListener('DOMContentLoaded', function () {
    const signInButton = document.getElementById('signInButton');
    if (signInButton) {
        signInButton.addEventListener('click', signInWithGoogle);
    } else {
        console.error('signInButton element not found.');
    }
});

function signInWithGoogle() {
    const provider = new GoogleAuthProvider();
    signInWithPopup(auth, provider)
    .then((result) => {
        console.log('User signed in:', result.user);
        window.location = '/dashboard'; // Redirect after successful login
    }).catch((error) => {
        console.error('Error during Google sign in:', error.message);
    });
}

// Handle the redirect result
getRedirectResult(auth).then((result) => {
    if (result && result.user) {
        console.log('User signed in:', result.user);
        // Perform further actions here
    }
}).catch((error) => {
    console.error('Error during Google sign in:', error.message);
});
