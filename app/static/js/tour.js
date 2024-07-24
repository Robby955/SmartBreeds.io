document.addEventListener('DOMContentLoaded', function () {
    // Initialize the Shepherd tour
    const tour = new Shepherd.Tour({
        defaultStepOptions: {
            cancelIcon: {
                enabled: true
            },
            classes: 'shadow-md bg-purple-dark',
            scrollTo: { behavior: 'smooth', block: 'center' }
        }
    });

    // Define the steps of the tour
    tour.addStep({
        id: 'welcome',
        text: 'Welcome to the Smart Dog Breed Classifier! Let me show you how to use this app.',
        attachTo: {
            element: '#file-label',
            on: 'top'
        },
        buttons: [
            {
                text: 'Next',
                action: tour.next
            },
            {
                text: 'Quit Tour',
                action: tour.cancel
            }
        ]
    });

    tour.addStep({
        id: 'upload',
        text: 'First, click "Choose an image" to select a dog image from your device. Here is an example image.',
        attachTo: {
            element: '#file-label',
            on: 'right'
        },
        buttons: [
            {
                text: 'Back',
                action: tour.back
            },
            {
                text: 'Next',
                action: tour.next
            },
            {
                text: 'Quit Tour',
                action: tour.cancel
            }
        ],
        beforeShowPromise: function () {
            return new Promise((resolve) => {
                const fileInput = document.getElementById('file-input');
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(new File([""], "IMG_1134.jpeg"));
                fileInput.files = dataTransfer.files;

                const imageUrl = "/static/images/IMG_1134.jpeg";
                const imagePreview = document.getElementById('image-preview');
                imagePreview.src = imageUrl;
                imagePreview.classList.remove('hidden');

                resolve();
            });
        }
    });

    tour.addStep({
        id: 'classify',
        text: 'Once the image is displayed, click "Classify" to predict the breed.',
        attachTo: {
            element: '#classify-btn',
            on: 'right'
        },
        buttons: [
            {
                text: 'Back',
                action: tour.back
            },
            {
                text: 'Next',
                action: () => {
                    document.getElementById('classify-btn').click();
                    setTimeout(() => {
                        document.getElementById('result-section').classList.remove('hidden');
                        document.getElementById('result-text').textContent = "Classification complete!";
                        document.getElementById('predicted-breed').textContent = "Pug";
                        document.getElementById('confidence').textContent = "Confidence: 95%";
                        document.getElementById('confirmation-container').classList.remove('hidden');
                        tour.next();
                    }, 2000); // Simulate processing time
                }
            },
            {
                text: 'Quit Tour',
                action: tour.cancel
            }
        ]
    });

    tour.addStep({
        id: 'confirmation',
        text: 'You can confirm or deny the prediction to help improve our model. If you know the breed, select it from the dropdown and submit.',
        attachTo: {
            element: '#confirmation-container',
            on: 'top'
        },
        buttons: [
            {
                text: 'Back',
                action: tour.back
            },
            {
                text: 'Finish Tour',
                action: () => {
                    tour.complete();
                    resetPageAfterTour();
                }
            }
        ]
    });

    document.getElementById('start-tour').addEventListener('click', function () {
        tour.start();
    });

    function resetPageAfterTour() {
        // Clear the example image and reset the form
        document.getElementById('file-input').value = '';
        document.getElementById('image-preview').classList.add('hidden');
        document.getElementById('image-preview').src = '';
        document.getElementById('result-section').classList.add('hidden');
        document.getElementById('confirmation-container').classList.add('hidden');
        document.getElementById('result-text').textContent = '';
        document.getElementById('predicted-breed').textContent = '';
        document.getElementById('confidence').textContent = '';
        document.getElementById('confirmation-container').style.display = 'none';
        document.getElementById('confirmation-message').textContent = "Do you know the real breed of this dog?";
        // Hide Yes and No buttons
        document.getElementById('yes-know-btn').style.display = 'inline-block';
        document.getElementById('no-know-btn').style.display = 'inline-block';
        document.getElementById('breed-dropdown').style.display = 'none';

        // Optionally, reset any other elements to their initial state
        document.getElementById('file-label').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});