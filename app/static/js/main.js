document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const classifyBtn = document.getElementById('classify-btn');

    classifyBtn.disabled = true;

    // Event listener for file input change
    fileInput.addEventListener('change', function () {
        // Enable the classify button only if a file is selected
        classifyBtn.disabled = !fileInput.files.length;
    });








    const resultSection = document.getElementById('result-section');


    let predictedBreed; // Allow reassignment
    const predictedBreedElement = document.getElementById('predicted-breed');

    let breedNames = []; // Initialize an empty array for breed names



    const confidence = document.getElementById('confidence');
    const confirmationContainer = document.getElementById('confirmation-container');
    const yesKnowBtn = document.getElementById('yes-know-btn');
    const noKnowBtn = document.getElementById('no-know-btn');
    const breedDropdown = document.getElementById('breed-dropdown');
    const breedSelect = document.getElementById('breed-select');
    const submitBreedBtn = document.getElementById('submit-correct-breed');
    const searchInput = document.getElementById('search-input');
    const uploadForm = document.getElementById('upload-form');
    const userAnswer = document.getElementById('breed-select').value;
    const breedSearchInput = document.getElementById('breed-search-input');
    function filterBreedSuggestions(input) {
        const filteredBreeds = breedNames.filter(breed => breed.toLowerCase().includes(input.toLowerCase()));
        return filteredBreeds;
    }

    // Function to update breed select options based on user input
    function updateBreedOptions(input) {
        const filteredBreeds = filterBreedSuggestions(input);
        breedSelect.innerHTML = ''; // Clear existing options
        filteredBreeds.forEach(breed => {
            const option = document.createElement('option');
            option.value = breed;
            option.textContent = breed.replace(/_/g, ' '); // Replace underscores with spaces
            breedSelect.appendChild(option);
        });
    }
    document.getElementById('file-input').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file.type === "image/jpeg") {
        convertToJPG(file);
    }
});

function convertToJPG(file) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const imgElement = document.createElement("img");
        imgElement.src = event.target.result;
        imgElement.onload = function(e) {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = e.target.width;
            canvas.height = e.target.height;
            ctx.drawImage(e.target, 0, 0, e.target.width, e.target.height);
            canvas.toBlob(function(blob) {
                const newFile = new File([blob], "converted_image.jpg", { type: "image/jpeg" });
                // Now you have a JPG file, you can handle it as needed
            }, "image/jpeg");
        }
    };
    reader.readAsDataURL(file);
}





    // Event listener for breed search input
    breedSearchInput.addEventListener('input', function () {
        const input = breedSearchInput.value;
        updateBreedOptions(input);
    });



    fileInput.addEventListener('change', function () {
        resultSection.classList.add('hidden');
        confirmationContainer.style.display = 'none'; // Hide the confirmation container
        imagePreview.classList.add('hidden'); // Optionally hide the image preview
        predictedBreedElement.textContent = 'Predicted Breed: '; // Reset predicted breed text
        confidence.textContent = 'Confidence: '; // Reset confidence text
    });


    document.getElementById('submit-correct-breed').addEventListener('click', function() {
    const formData = new FormData();
    const fileInput = document.getElementById('file-input');
    const predictedBreed = document.getElementById('predicted-breed').textContent.split(': ')[1];
    const userAnswer = document.getElementById('breed-select').value;

    formData.append('image', fileInput.files[0]);
    formData.append('predicted_breed', predictedBreed);
    formData.append('user_answer', userAnswer);

    fetch('/submit_image', {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        // Handle success, update UI accordingly
    })
    .catch(error => {
        console.error('Error:', error);
        // Handle error, update UI accordingly
    });
});








    fileInput.addEventListener('change', function (event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                // Reset the interface when a new image is loaded
                resetInterface();
            };
            reader.readAsDataURL(file);
        }
    });


classifyBtn.addEventListener('click', function () {
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Display a loading message or spinner
    const resultTextElement = document.getElementById('result-text');
    resultTextElement.textContent = 'Classifying...';
    resultSection.classList.remove('hidden');  // Ensure the result section is visible to show the loading message

    // Clear predicted breed and confidence text
    predictedBreedElement.textContent = '';
    confidence.textContent = '';

    fetch('/classify', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log("Received data:", data); // Debugging log
        const prediction = data.prediction || "Not available";
        const confidenceValue = data.confidence;
        const croppedImageData = data.croppedImage || '';

        // Set the image preview to the cropped image
        if (croppedImageData) {
            imagePreview.src = croppedImageData;
        }

        // Set the predicted breed and confidence text
        predictedBreedElement.textContent = `Predicted Breed: ${prediction}`;
        if (typeof confidenceValue === 'number') {
            confidence.textContent = `Confidence: ${Math.round(confidenceValue * 100)}%`;
        } else {
            console.log("Received unexpected confidence value:", confidenceValue); // Debugging log
            confidence.textContent = 'Confidence: N/A';
        }

        // Clear the "Classifying..." message
        resultTextElement.textContent = '';

        confirmationContainer.style.display = 'block';
    })
    .catch(error => {
        console.error('Error during classification:', error);
        // Clear the "Classifying..." message and show error messages
        resultTextElement.textContent = '';
        predictedBreedElement.textContent = 'Error occurred during classification';
        confidence.textContent = 'Confidence: N/A';
    });
});



    // Handler for 'Yes, I know the breed' button
    yesKnowBtn.addEventListener('click', function () {
        // Hide Yes and No buttons
        yesKnowBtn.style.display = 'none';
        noKnowBtn.style.display = 'none';
        // Show the breed selection dropdown
        breedDropdown.style.display = 'block';
        // Optionally, change the confirmation message or provide additional instructions
        breedSearchInput.value = ''; // Clear previous input
        updateBreedOptions(''); // Update options with all breeds initially
        document.getElementById('confirmation-message').textContent = "Please select the correct breed:";
    });

    // Handler for 'No, I don't know the breed' button
    noKnowBtn.addEventListener('click', function () {
        submitResponse(null); // Submit with null as the user's breed response
    });

    // Handler for submitting the correct breed
    submitBreedBtn.addEventListener('click', function () {
        const userBreed = breedSelect.value;
        submitResponse(userBreed); // Submit with the user's selected breed
    });

    // Function to submit the response
function submitResponse(userBreed) {
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    formData.append('predicted_breed', predictedBreedElement.textContent.split(': ')[1]);
    formData.append('user_answer', userBreed);

    fetch('/submit_response', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log('Submission successful:', data);
        // Show a thank you message for providing feedback
        document.getElementById('confirmation-message').textContent = "Thank you for your feedback!";
        // Wait for a short duration before hiding the confirmation box
        setTimeout(() => {
            // Hide the confirmation container without resetting the entire interface
            document.getElementById('confirmation-container').style.display = 'none';
        }, 2000); // Wait for 2 seconds
    })
    .catch(error => {
        console.error('Submission error:', error);
        // Handle error, update UI accordingly
    });
}












    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const fileInput = document.getElementById('file-input');
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            fetch('/submit_image', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                console.log('Image uploaded successfully:', data);
                // Handle success response, update UI accordingly
            })
            .catch(error => {
                console.error('Error uploading image:', error);
                // Handle error, update UI accordingly
            });
        });
    }






    // Ensure buttons have type="button" attribute
    if (classifyBtn) {
    classifyBtn.type = 'button';
}

    if (yesKnowBtn) {
    yesKnowBtn.type = 'button';
}
    if (noKnowBtn) {
    noKnowBtn.type = 'button';
}


if (confirmationContainer) {

    confirmationContainer.style.display = 'none';
 }

 if(breedDropdown) {
 breedDropdown.style.display='none';
 }

if(fileInput){

    fileInput.addEventListener('change', function (event) {
        const file = event.target.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');

                // Reset and hide the confirmation box
                confirmationContainer.classList.remove('hidden');
                // Reset the confirmation message if it was changed
                document.getElementById('confirmation-message').textContent = "Do you know the real breed of this dog?";

                // Reset the interface for a new classification
                resetInterface();
            };

            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '';
            imagePreview.classList.add('hidden');

            // Reset the interface as no file is selected
            resetInterface();
        }
    });
}


if (classifyBtn) {
    classifyBtn.addEventListener('click', function () {
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Display a loading message or spinner
    const resultTextElement = document.getElementById('result-text');
    resultTextElement.textContent = 'Classifying...';
    resultSection.classList.remove('hidden');  // Ensure the result section is visible to show the loading message

    // Clear predicted breed and confidence text
    predictedBreedElement.textContent = '';
    confidence.textContent = '';

    fetch('/classify', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log("Received data:", data); // Debugging log
        const prediction = data.prediction || "Not available";
        const confidenceValue = data.confidence;
        const croppedImageData = data.croppedImage || '';

        // Set the image preview to the cropped image
        if (croppedImageData) {
            imagePreview.src = croppedImageData;
        }

        // Set the predicted breed and confidence text
        predictedBreedElement.textContent = `Predicted Breed: ${prediction}`;
        if (typeof confidenceValue === 'number') {
            confidence.textContent = `Confidence: ${Math.round(confidenceValue * 100)}%`;
        } else {
            console.log("Received unexpected confidence value:", confidenceValue); // Debugging log
            confidence.textContent = 'Confidence: N/A';
        }

        // Clear the "Classifying..." message
        resultTextElement.textContent = '';

        confirmationContainer.style.display = 'block';
    })
    .catch(error => {
        console.error('Error during classification:', error);
        // Clear the "Classifying..." message and show error messages
        resultTextElement.textContent = '';
        predictedBreedElement.textContent = 'Error occurred during classification';
        confidence.textContent = 'Confidence: N/A';
    });
});
}




if(yesKnowBtn){
    yesKnowBtn.addEventListener('click', function () {
        // Hide Yes and No buttons
        yesKnowBtn.style.display = 'none';
        noKnowBtn.style.display = 'none';

        // Show the breed selection dropdown
        breedDropdown.style.display = 'block';
        if(searchInput){
        searchInput.style.display = 'inline-block';
        }

        // Optionally, change the confirmation message or provide additional instructions
        document.getElementById('confirmation-message').textContent = "Please select the correct breed:";
    });
}

if(noKnowBtn){
    noKnowBtn.addEventListener('click', function () {
        document.getElementById('confirmation-message').textContent = "Thank you for your feedback!";
    // Hide Yes and No buttons
    yesKnowBtn.style.display = 'none';
    noKnowBtn.style.display = 'none';

    // Optionally, hide the breed selection dropdown if it's visible
    breedDropdown.style.display = 'none';
        // Reset the interface after a delay
    });
}
if(submitBreedBtn){
submitBreedBtn.addEventListener('click', function () {
    const selectedBreed = breedSelect.value;
    document.getElementById('confirmation-message').textContent = "Thank you for your feedback! Please enter another image.";

    // Hide the dropdown and search input after submission
    breedDropdown.style.display = 'none';
    if (searchInput) {
        searchInput.style.display = 'none';
    }

    // Hide the confirmation container after a brief delay


    // Potentially send selected breed back to server here

});
}
    function resetInterface() {
        yesKnowBtn.style.display = 'inline-block'; // Show Yes button
        noKnowBtn.style.display = 'inline-block'; // Show No button
        breedDropdown.style.display = 'none'; // Hide breed dropdown
        document.getElementById('confirmation-message').textContent = "Do you know the real breed of this dog?";
        // Hide result section until next classification
        resultSection.classList.add('hidden');
    }

    // Function to fetch and populate breed names
    function fetchBreedNames() {
        if (!breedSelect) return;
        console.log("Fetching breed names...");
        fetch('/breed_names')
            .then(response => {
                console.log("Received response:", response);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Breed names fetched:", data);
                if (!Array.isArray(data)) {
                    throw new Error('Response format is incorrect');
                }

                breedNames = data.sort();
                breedSelect.innerHTML = ''; // Clear existing options
                breedNames.forEach(breed => {
                    const option = document.createElement('option');
                    option.value = breed;
                    option.textContent = breed;
                    breedSelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error fetching breed names:', error);
            });
    }
    // Fetch and populate breed names
    fetchBreedNames();


});