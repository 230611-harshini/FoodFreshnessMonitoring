console.log("script.js loaded");

const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const prediction = document.getElementById("prediction");
const shelfLife = document.getElementById("shelfLife");

window.addEventListener("beforeunload", () => {
    console.log("PAGE IS RELOADING");
});
// Image Preview
imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
    }

});

// Predict Button
predictBtn.addEventListener("click", async () => {

    console.log("Button clicked");

    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    prediction.textContent = "Predicting...";
    shelfLife.textContent = "...";

    const formData = new FormData();
    formData.append("file", file);

    try {

        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        console.log("Status:", response.status);

        const data = await response.json();

        console.log("Before update");

prediction.textContent = data.prediction;
console.log("Prediction updated");

shelfLife.textContent = data.shelf_life;
console.log("Shelf life updated");

        console.log("Response:", data);

        // Update webpage
        prediction.textContent = data.prediction;
        shelfLife.textContent = data.shelf_life;

    }
    catch (error) {

        console.error(error);

        prediction.textContent = "Error!";
        shelfLife.textContent = "Backend not running.";

    }

});