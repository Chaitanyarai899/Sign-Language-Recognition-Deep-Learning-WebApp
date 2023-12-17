
const URL = "https://teachablemachine.withgoogle.com/models/2l5fpoYUy/";
let model, labelContainer, maxPredictions;
let videoElement, cropCanvas, cropCanvasContext;
let sentence = [];
const sentenceMaxLength = 10;
const poseChangeTimeout = 1000; // 2 seconds for pose changing
const predictionInterval = 3000; // 5 seconds for making predictions
const cropSize = 400; // Size of the cropped area
async function init() {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";
    
    model = await tmImage.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();
    videoElement = document.createElement('video');
    videoElement.width = 200;
    videoElement.height = 200;
    cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropSize;
    cropCanvas.height = cropSize;
    cropCanvasContext = cropCanvas.getContext('2d');
    navigator.mediaDevices.getDisplayMedia({video: true})
    .then(stream => {
        videoElement.srcObject = stream;
        videoElement.play();
        document.getElementById("webcam-container").appendChild(cropCanvas);
    })
    .catch(err => {
        console.error("Error: " + err);
    });
    labelContainer = document.getElementById("label-container");
    for (let i = 0; i < 2; i++) {
        labelContainer.appendChild(document.createElement("div"));
    }
    makePredictions();
}
async function makePredictions() {
    let recentPredictions = [];
    setTimeout(async () => {
        const interval = setInterval(async () => {
            // Crop the video feed and draw to the canvas
            const cropX = (videoElement.videoWidth - cropSize) / 2;
            const cropY = (videoElement.videoHeight - cropSize) / 2;
            cropCanvasContext.drawImage(videoElement, cropX, cropY, cropSize, cropSize, 0, 0, cropSize, cropSize);
            const prediction = await model.predict(cropCanvas);
            const maxProbabilityIndex = findMaxProbabilityIndex(prediction);
            recentPredictions.push(prediction[maxProbabilityIndex].className);
        }, 100); // make a prediction every 100ms
        setTimeout(() => {
            clearInterval(interval);
            const modeClass = mode(recentPredictions);
            sentence.push(modeClass);
            if (sentence.length > sentenceMaxLength) {
                sentence.shift();
            }
            if(labelContainer.childNodes[0]) {
                labelContainer.childNodes[0].innerHTML = "Current prediction: " + modeClass; 
            }
            if(labelContainer.childNodes[1]) {
                labelContainer.childNodes[1].innerHTML = "Predicted sentence: " + sentence.join(" ");
            }
            makePredictions();
        }, predictionInterval);
    }, poseChangeTimeout);
}
function findMaxProbabilityIndex(prediction) {
    let maxIndex = 0;
    let maxProbability = prediction[0].probability;
    for (let i = 1; i < maxPredictions; i++) {
        if (prediction[i].probability > maxProbability) {
            maxProbability = prediction[i].probability;
            maxIndex = i;
        }
    }
    return maxIndex;
}
function mode(arr){
    return arr.sort((a,b) =>
        arr.filter(v => v===a).length
        - arr.filter(v => v===b).length
    ).pop();
}