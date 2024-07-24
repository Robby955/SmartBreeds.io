document.addEventListener('DOMContentLoaded', function () {
    const timedModeBtn = document.getElementById('timed-mode-btn');
    const endlessModeBtn = document.getElementById('endless-mode-btn');
    const breedOfRoundElement = document.getElementById('breed-of-round');
    const playAgainBtn = document.getElementById('play-again');
    const homeButton = document.getElementById('home-button');
    const questionElement = document.getElementById('question');
    const timerElement = document.getElementById('timer');
    const currentRoundElement = document.getElementById('current-round');
    const currentScoreElement = document.getElementById('current-score');
    const feedbackElement = document.getElementById('feedback');
    const imageContainer = document.getElementById('image-container');
    const initialInstructions = document.getElementById('initial-instructions');
    const loadingElement = document.getElementById('loading');
    const livesElement = document.getElementById('lives');
    const highScoreElement = document.getElementById('high-score');

    let currentRound = 0;
    let score = 0;
    let lives = 3;
    let timerInterval;
    let startTime;
    const TIMED_ROUND_DURATION = 5; // seconds
    const ENDLESS_ROUND_DURATION = 45; // seconds
    let gameMode = '';

    timedModeBtn.addEventListener('click', () => startGameHandler('timed'));
    endlessModeBtn.addEventListener('click', () => startGameHandler('endless'));
    playAgainBtn.addEventListener('click', () => window.location.reload());

    function startGameHandler(mode) {
        gameMode = mode;
        timedModeBtn.style.display = 'none';
        endlessModeBtn.style.display = 'none';
        initialInstructions.classList.add('hidden');
        playAgainBtn.style.display = 'none';
        homeButton.style.display = 'inline-block';
        highScoreElement.classList.add('hidden'); // Hide high score at the start
        startGame();
    }

    function startGame() {
        currentRound = 0;
        score = 0;
        lives = 3; // Reset lives
        updateScoreAndRound();
        updateLives();
        loadGameData();
    }

    async function loadGameData() {
        loadingElement.classList.remove('hidden'); // Show loading text
        try {
            const response = await fetch('/game/game_data');
            const data = await response.json();
            displayGameData(data);
            questionElement.textContent = `Which one is a ${formatBreedName(data.breedOfRound)}?`;
            questionElement.classList.remove('hidden');
            if (gameMode === 'timed') {
                startTimer(TIMED_ROUND_DURATION);
            } else {
                startTimer(ENDLESS_ROUND_DURATION);
            }
        } catch (error) {
            console.error('Error loading game data:', error);
            feedbackElement.textContent = "Failed to load game data. Please try again.";
        } finally {
            loadingElement.classList.add('hidden'); // Hide loading text
        }
    }

    function displayGameData(data) {
        imageContainer.innerHTML = '';
        const breeds = Object.keys(data.images);
        const selectedBreeds = getRandomBreeds(breeds, 5); // Ensure five images are shown

        selectedBreeds.forEach(breed => {
            const breedImages = data.images[breed];
            const randomImage = breedImages[Math.floor(Math.random() * breedImages.length)];
            const imgElement = document.createElement('img');
            imgElement.src = randomImage;
            imgElement.alt = `Image of ${formatBreedName(breed)}`;
            imgElement.classList.add('game-image');
            imgElement.addEventListener('click', () => submitGuess(data.breedOfRound, breed));
            imageContainer.appendChild(imgElement);
        });
    }

    function getRandomBreeds(breeds, numBreeds) {
        const shuffled = [...breeds].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, numBreeds);
    }

    function formatBreedName(breedName) {
        return breedName.replace(/_/g, ' ').split(/\s+/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    function startTimer(duration) {
        let secondsLeft = duration;
        startTime = Date.now();
        if (gameMode === 'timed') {
            timerElement.textContent = `Time left: ${secondsLeft} seconds`;
            timerElement.classList.remove('hidden');
        } else {
            timerElement.classList.add('hidden'); // Hide timer in endless mode
        }
        timerInterval = setInterval(() => {
            if (secondsLeft > 0) {
                secondsLeft--;
                if (gameMode === 'timed') {
                    timerElement.textContent = `Time left: ${secondsLeft} seconds`;
                }
            } else {
                clearInterval(timerInterval);
                if (gameMode === 'timed') {
                    handleTimeout();
                } else {
                    lives--;
                    updateLives();
                    if (lives === 0) {
                        showFinalScore();
                    } else {
                        handleTimeout();
                    }
                }
            }
        }, 1000);
    }

    function submitGuess(correctBreed, userGuess) {
        clearInterval(timerInterval);
        const endTime = Date.now();
        const timeTaken = (endTime - startTime) / 1000;

        if (userGuess === correctBreed) {
            if (gameMode === 'timed') {
                const points = Math.max(10 - timeTaken, 0);
                score += points;
            } else {
                score += 1;
            }
            feedbackElement.textContent = 'Correct!';
            feedbackElement.style.color = 'green';
        } else {
            feedbackElement.textContent = 'Oops!';
            feedbackElement.style.color = 'red';
            highlightCorrectImage(correctBreed);
            if (gameMode === 'endless') {
                lives--;
                updateLives();
                if (lives === 0) {
                    showFinalScore();
                    return;
                }
            }
        }
        setTimeout(() => {
            feedbackElement.textContent = '';
            currentRound++;
            if (gameMode === 'timed' && currentRound < 5) {
                updateScoreAndRound();
                loadGameData();
            } else if (gameMode === 'endless') {
                updateScoreAndRound();
                loadGameData();
            } else {
                showFinalScore();
            }
        }, 2000);
    }

    function highlightCorrectImage(correctBreed) {
        const images = document.querySelectorAll('.game-image');
        images.forEach(img => {
            if (img.alt.includes(formatBreedName(correctBreed))) {
                img.classList.add('correct');
            } else {
                img.classList.add('incorrect');
            }
        });
    }

    function handleTimeout() {
        feedbackElement.textContent = 'Time\'s up!';
        highlightCorrectImage(document.querySelector('.game-image').alt.split(' ').slice(2).join(' '));
        setTimeout(() => {
            feedbackElement.textContent = '';
            currentRound++;
            if (gameMode === 'timed' && currentRound < 5) {
                updateScoreAndRound();
                loadGameData();
            } else if (gameMode === 'endless') {
                updateScoreAndRound();
                loadGameData();
            } else {
                showFinalScore();
            }
        }, 2000);
    }

    async function submitScore() {
        const response = await fetch('/game/submit_score', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ score: Math.round(score), mode: gameMode })
        });
        const result = await response.json();
        console.log(result.message);
    }

    async function fetchHighScore() {
        const response = await fetch(`/game/high_scores?mode=${gameMode}`);
        const highScores = await response.json();
        const highestScore = highScores.length > 0 ? highScores[0].score : 'N/A';
        highScoreElement.textContent = `High Score: ${highestScore}`;
        highScoreElement.classList.remove('hidden');
    }

    function showFinalScore() {
        feedbackElement.textContent = `Game over! Your final score is: ${Math.round(score)}`;
        submitScore();
        fetchHighScore();
        playAgainBtn.style.display = 'block';
    }

    function updateScoreAndRound() {
        currentRoundElement.textContent = `Round: ${currentRound + 1}`;
        currentScoreElement.textContent = `Score: ${Math.round(score)}`;
        currentRoundElement.classList.remove('hidden');
        currentScoreElement.classList.remove('hidden');
    }

    function updateLives() {
        if (gameMode === 'endless') {
            livesElement.textContent = `Lives: ${lives}`;
            livesElement.classList.remove('hidden');
        } else {
            livesElement.classList.add('hidden');
        }
    }
});
