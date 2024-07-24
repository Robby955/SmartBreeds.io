// Import the necessary Firebase modules
import { initializeApp } from 'https://www.gstatic.com/firebasejs/9.6.10/firebase-app.js';
import { getAuth, setPersistence, browserLocalPersistence } from 'https://www.gstatic.com/firebasejs/9.6.10/firebase-auth.js';
import { getFirestore } from 'https://www.gstatic.com/firebasejs/9.6.10/firebase-firestore.js';

// Firebase configuration
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

// Firebase services
const auth = getAuth(app);
const db = getFirestore(app);

// Set auth persistence
setPersistence(auth, browserLocalPersistence)
    .then(() => {
        console.log("Auth persistence set to LOCAL");
    })
    .catch((error) => {
        console.error("Error setting auth persistence:", error.message);
    });

// Export Firebase services for use in other scripts
export { auth, db, app };
