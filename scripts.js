document.addEventListener('DOMContentLoaded', function() {
    const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const summarizeButton = document.getElementById('summarizeButton');
    const overlay = document.getElementById('overlay');

    summarizeButton.addEventListener('click', function(e) {
        e.preventDefault();

        overlay.style.display = 'flex';  // Show the overlay

        fetch('/summarize/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({'text': inputText.value})
        })
        .then(response => response.json())
        .then(data => {
            outputText.value = data.summary;
            overlay.style.display = 'none';  // Hide the overlay
        });
    });
});
