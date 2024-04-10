document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    let formData = new FormData(this);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display prediction
        let resultElement = document.getElementById('result');
        resultElement.textContent = `Prediction: ${data.prediction}`;
        resultElement.classList.remove('hidden');
        
        // Display uploaded image
        let imageElement = document.getElementById('uploaded-image');
        imageElement.src = data.image_url;
        imageElement.classList.remove('hidden');
    })
    .catch(error => console.error('Error:', error));
});
