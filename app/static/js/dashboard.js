document.addEventListener('DOMContentLoaded', function() {
    // Function to update the count of dog breeds classified
    function updateBreedCount() {
        // Placeholder: Fetch the count from your server or API
        let breedCount = 100; // Example count

        // Update the count on the dashboard
        const breedCountElement = document.getElementById('breed-count');
        if (breedCountElement) {
            breedCountElement.textContent = breedCount;
        }
    }

    // Function to update the count of user interactions
    function updateUserInteractionCount() {
        // Placeholder: Fetch the count from your server or API
        let interactionCount = 150; // Example count

        // Update the count on the dashboard
        const interactionCountElement = document.getElementById('interaction-count');
        if (interactionCountElement) {
            interactionCountElement.textContent = interactionCount;
        }
    }

    // Function to update the user's high scores
    function updateUserHighScores() {
        fetch('/user/high_scores')
            .then(response => response.json())
            .then(data => {
                const timedHighScore = data.timed || 0;
                const endlessHighScore = data.endless || 0;

                const timedHighScoreElement = document.getElementById('timed-high-score');
                const endlessHighScoreElement = document.getElementById('endless-high-score');

                if (timedHighScoreElement) {
                    timedHighScoreElement.textContent = timedHighScore;
                }

                if (endlessHighScoreElement) {
                    endlessHighScoreElement.textContent = endlessHighScore;
                }
            })
            .catch(error => {
                console.error('Error fetching high scores:', error);
                const timedHighScoreElement = document.getElementById('timed-high-score');
                const endlessHighScoreElement = document.getElementById('endless-high-score');

                if (timedHighScoreElement) {
                    timedHighScoreElement.textContent = 'Error';
                }

                if (endlessHighScoreElement) {
                    endlessHighScoreElement.textContent = 'Error';
                }
            });
    }

    // Call the functions to update counts and high scores on page load
    updateBreedCount();
    updateUserInteractionCount();
    updateUserHighScores();

    // Add more interactive functionalities as needed
});
