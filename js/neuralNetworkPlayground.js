// neuralNetworkPlayground.js

// Define a simple model
const model = tf.sequential();

// Visualization parameters
let layerVisuals = []; // To store layer visualization data

let currentData = []; // Holds the current dataset
let isTraining = false; // Flag to check if training has started

let dataCanvas, networkCanvas; // Separate canvases for data and network
// Global variables to track training progress and loss
let currentEpoch = 0;
let totalEpochs = 100; // This can be adjusted or set dynamically
let lossHistory = [];
// Function to initialize the model
function initializeModel(activation) {
    // Clear existing model
    model.layers = [];
    layerVisuals = [];

    // Create a simple feedforward neural network
    // Input layer
    model.add(tf.layers.dense({ units: 5, inputShape: [2], activation: activation }));
    // Hidden layer
    model.add(tf.layers.dense({ units: 4, activation: activation }));
    // Output layer
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    // Compile the model with loss and optimizer
    model.compile({
        optimizer: tf.train.adam(parseFloat(document.getElementById('learningRate').value)),
        loss: 'binaryCrossentropy'
    });

    // Prepare visualization data
    prepareVisualizationData();
}

// Function to prepare visualization data based on the model
function prepareVisualizationData() {
    layerVisuals = [];
    model.layers.forEach((layer, i) => {
        let layerInfo = {
            nodes: layer.units,
            activations: Array(layer.units).fill(0)
        };
        layerVisuals.push(layerInfo);
    });
}
function selectDataset() {
    let selection = document.getElementById('dataShape').value;
    switch (selection) {
        case 'linear':
            currentData = generateLinearData();
            break;
        case 'concentric':
            currentData = generateConcentricData();
            break;
        case 'twoClusters':
            currentData = generateTwoClustersData();
            break;
        case 'checkerboard':
            currentData = generateCheckerboardData();
            break;
        case 'moons':
            currentData = generateMoonsData();
            break;
        case 'spiral':
            currentData = generateSpiralData();
            break;
        // Add more cases as needed
    }
    isTraining = false; // Reset training flag
    updateDataVisualization(); // Visualize the new dataset
}
// Generates a linearly separable dataset
function generateLinearData() {
    let data = [];
    for (let x = 0; x < p.width; x += resolution) {
        for (let y = 0; y < p.height; y += resolution) {
            let classLabel = x < p.width / 2 ? 0 : 1;
            data.push({x: x, y: y, label: classLabel});
        }
    }
    return data;
}

// Generates concentric circles dataset
function generateConcentricData() {
    let data = [];
    let centerX = p.width / 2;
    let centerY = p.height / 2;
    let maxRadius = Math.min(centerX, centerY) - 10; // Padding of 10

    for (let angle = 0; angle < TWO_PI; angle += 0.1) {
        let innerRadius = maxRadius / 3;
        let outerRadius = 2 * maxRadius / 3;

        // Inner circle (class 0)
        let xInner = centerX + innerRadius * cos(angle);
        let yInner = centerY + innerRadius * sin(angle);
        data.push({x: xInner, y: yInner, label: 0});

        // Outer ring (class 1)
        let xOuter = centerX + outerRadius * cos(angle);
        let yOuter = centerY + outerRadius * sin(angle);
        data.push({x: xOuter, y: yOuter, label: 1});
    }
    return data;
}

function generateTwoClustersData() {
    let data = [];
    let pointsPerCluster = 100;

    // Parameters for the first cluster
    let mean1 = [p.width * 0.3, p.height * 0.3];
    let variance1 = [p.width * 0.05, p.height * 0.05];

    // Parameters for the second cluster
    let mean2 = [p.width * 0.7, p.height * 0.7];
    let variance2 = [p.width * 0.05, p.height * 0.05];

    // Generate points for the first cluster
    for (let i = 0; i < pointsPerCluster; i++) {
        let x = randomGaussian(mean1[0], variance1[0]);
        let y = randomGaussian(mean1[1], variance1[1]);
        data.push({x: x, y: y, label: 0});
    }

    // Generate points for the second cluster
    for (let i = 0; i < pointsPerCluster; i++) {
        let x = randomGaussian(mean2[0], variance2[0]);
        let y = randomGaussian(mean2[1], variance2[1]);
        data.push({x: x, y: y, label: 1});
    }

    return data;
}

function generateCheckerboardData() {
    let data = [];
    let tileSize = p.width / 10; // Number of tiles per row/column

    for (let x = 0; x < p.width; x += tileSize) {
        for (let y = 0; y < p.height; y += tileSize) {
            let classLabel = (floor(x / tileSize) + floor(y / tileSize)) % 2;
            data.push({x: x + tileSize / 2, y: y + tileSize / 2, label: classLabel});
        }
    }

    return data;
}

function generateMoonsData() {
    let data = [];
    let pointsPerMoon = 100;
    let radius = p.width * 0.2;

    // Generate points for the first moon
    for (let i = 0; i < pointsPerMoon; i++) {
        let angle = random(Math.PI); // half circle
        let x = p.width * 0.5 + radius * cos(angle);
        let y = p.height * 0.5 + radius * sin(angle);
        data.push({x: x, y: y, label: 0});
    }

    // Generate points for the second moon
    for (let i = 0; i < pointsPerMoon; i++) {
        let angle = random(Math.PI); // half circle
        let x = p.width * 0.5 + radius * cos(angle + Math.PI);
        let y = p.height * 0.5 + radius * sin(angle + Math.PI);
        data.push({x: x, y: y, label: 1});
    }

    return data;
}

function generateSpiralData() {
    let data = [];
    let turns = 2;
    let pointsPerTurn = 100;

    // Generate points for the first spiral
    for (let i = 0; i < turns * pointsPerTurn; i++) {
        let angle = map(i, 0, turns * pointsPerTurn, 0, turns * TWO_PI);
        let radius = map(i, 0, turns * pointsPerTurn, 0, p.width / 3);
        let x = p.width / 2 + radius * cos(angle);
        let y = p.height / 2 + radius * sin(angle);
        data.push({x: x, y: y, label: i % 2});
    }

    // Generate points for the second spiral
    for (let i = 0; i < turns * pointsPerTurn; i++) {
        let angle = map(i, 0, turns * pointsPerTurn, 0, turns * TWO_PI) + Math.PI;
        let radius = map(i, 0, turns * pointsPerTurn, 0, p.width / 3);
        let x = p.width / 2 + radius * cos(angle);
        let y = p.height / 2 + radius * sin(angle);
        data.push({x: x, y: y, label: (i + 1) % 2});
    }

    return data;
}



// Add this function to draw different shapes for different classes
function visualizeData() {

    let dataCanvas = select('#dataCanvas');
    // Loop through each point in the current dataset
    for (let i = 0; i < currentData.length; i++) {
        const point = currentData[i];
        // Use the appropriate canvas context (dataCanvas)
        dataCanvas.stroke(0);
        // Fix: Ensure that points of both classes are visualized
        if (point.label === 0) {
            dataCanvas.fill('blue'); // Class 0 as circles
            dataCanvas.ellipse(point.x, point.y, resolution, resolution);
        } else {
            dataCanvas.fill('red'); // Class 1 as plus signs
            drawPlusSign(dataCanvas, point.x, point.y, resolution);
        }
    }
}


// Function to update the epoch and loss on the webpage
function updateTrainingStatus(epoch, loss) {
    document.getElementById('epochInfo').innerText = `Epoch: ${epoch}/${totalEpochs}`;
    document.getElementById('lossInfo').innerText = `Loss: ${loss.toFixed(4)}`;
    updateLossGraph(lossHistory); // [Fix]: Call the newly created updateLossGraph function
}
// [New Function]: Implementing graph logic to display a running graph of the loss
function updateLossGraph(lossHistory) {
    // This function will need to be implemented with a graphing library like Chart.js
    // For simplicity, this example will update a div with the loss values
    const lossGraphElement = document.getElementById('lossGraph');
    lossGraphElement.innerHTML = ''; // Clear previous graph contents
    lossHistory.forEach((loss, index) => {
        // Append new loss value to the graph
        // In a real implementation, this would draw on a canvas
        const lossValueElement = document.createElement('div');
        lossValueElement.innerText = `Epoch ${index + 1}: ${loss}`;
        lossGraphElement.appendChild(lossValueElement);
    });
}

// Function to train the model
async function trainModel() {
    // Initialize or re-initialize the model with the selected activation function
    let activation = document.getElementById('activation').value;
    initializeModel(activation);

    // Convert currentData to tensors with proper shapes
    // The shape is [num_examples, num_features_per_example]
    const inputs = tf.tensor2d(currentData.map(p => [p.x / p.width, p.y / p.height]), [currentData.length, 2]);
    const labels = tf.tensor2d(currentData.map(p => [p.label]), [currentData.length, 1]); // Labels must also be a 2D tensor


    // Reset global tracking variables
    currentEpoch = 0;
    lossHistory = [];

    // Disable the dataShape selector during training
    document.getElementById('dataShape').disabled = true;
    // Update the DOM elements for epochs and loss before training starts
    updateTrainingStatus(0, 0);
    // Train the model
    await model.fit(inputs, labels, {
        epochs: totalEpochs,
        callbacks: {
            onEpochEnd: async (epoch, log) => {
                currentEpoch = epoch + 1; // Update current epoch
                // Update the training status on the webpage
                updateTrainingStatus(currentEpoch, log.loss);
                lossHistory.push(log.loss); // Add loss to history
                document.getElementById('epochInfo').innerText = `Epoch ${currentEpoch}/${totalEpochs}`;
                document.getElementById('lossInfo').innerText = `Loss: ${log.loss.toFixed(4)}`;
                // Update loss graph
                updateLossGraph(lossHistory);
                // Update visualization after each epoch
                await updateDataVisualization();
            }
        }
    });

    // Re-enable the dataShape selector after training
    document.getElementById('dataShape').disabled = false;

    isTraining = true; // Set training flag to true
    document.getElementById('output').innerText = 'Training complete!';
}


// Define the resolution of the grid (lower = more points = slower)
const resolution = 20;


// p5.js draw function
function draw() {
    // Fix: Reference the correct canvas using the assigned ID
    let dataCanvas = select('#dataCanvas');
    let networkCanvas = select('#networkCanvas');
    
    if (dataCanvas) {
        dataCanvas.clear();
        if (isTraining) {
            updateDataVisualization();
        }
    }

    if (networkCanvas) {
        networkCanvas.clear();
        drawNetwork(networkCanvas);
    }
}

// Function to visualize the data flow through the network
async function visualizeDataFlow() {
    // Get the selected sample data point
    let sampleDataValue = document.getElementById('sampleData').value.split(',');
    let sampleData = tf.tensor2d([sampleDataValue.map(v => parseFloat(v))]);

    // Predict the output using the model
    let prediction = model.predict(sampleData);

    // Extract the activations and update visualization
    for (let i = 0; i < model.layers.length; i++) {
        let layer = model.layers[i];
        let output = layer.apply(sampleData);
        let activations = await output.data();
        layerVisuals[i].activations = Array.from(activations);
    }

    // Redraw the network with updated activations
    //draw();
}
// Function to predict and draw the decision boundary
async function drawDecisionBoundary() {
    // Use currentData instead of generateGrid
    // Predict classes for each point in the currentData
    let predictions = await model.predict(tf.tensor2d(currentData.map(p => [p.x / p.width, p.y / p.height]), [currentData.length, 2])).data();

    // Draw each point based on the prediction
    for (let i = 0; i < currentData.length; i++) {
        let prediction = predictions[i];
        let point = currentData[i];
        // Change color based on prediction, but maintain the shape
        if (point.label === 0) {
            dataCanvas.fill(prediction < 0.5 ? 'blue' : 'red');
            dataCanvas.ellipse(point.x, point.y, resolution, resolution); // Circle for class 0
        } else {
            dataCanvas.fill(prediction >= 0.5 ? 'red' : 'blue');
            drawPlusSign(dataCanvas, point.x, point.y, resolution); // Plus sign for class 1
        }
    }
}
  // Helper function to draw plus signs
  function drawPlusSign(canvas, x, y, size) {
    const halfSize = size / 2;
    canvas.strokeWeight(2);
    canvas.stroke(0); // Use black stroke for visibility
    canvas.line(x - halfSize, y, x + halfSize, y); // Horizontal line
    canvas.line(x, y - halfSize, x, y + halfSize); // Vertical line
}
  
// Function to update visualization with latest activations
async function updateDataVisualization() {
    // Clear the data canvas and redraw the ground truth of the data
    dataCanvas.clear();
    visualizeData(); // Fix: Call this to redraw the initial state of the data points

    // Only update the visualization if the model is training
    if (isTraining) {
        await drawDecisionBoundary(); // Fix: Call this to recolor points based on the model's predictions
    }
    drawNetwork(networkCanvas);
}

// Function to draw the neural network visualization
function drawNetwork(canvas) {

    let networkCanvas = select('#networkCanvas');
    // Define spacing and sizing outside of the loop for visibility
    let xSpacing = canvas.p.width / (layerVisuals.length + 1);
    let ySpacing = canvas.p.height / (Math.max(...layerVisuals.map(l => l.nodes)) + 1);

    canvas.clear(); // Use clear instead of background to avoid covering the entire canvas
    canvas.strokeWeight(1); // Set stroke weight for visibility

    // Draw each layer
    layerVisuals.forEach((layer, i) => {
        let xPos = xSpacing * (i + 1);
        for (let j = 0; j < layer.nodes; j++) {
            let yPos = ySpacing * (j + 1);
            let nodeSize = 20;
            canvas.fill(255); // Set fill to white to ensure nodes are visible
            canvas.stroke(0); // Set stroke to black for the nodes
            canvas.ellipse(xPos, yPos, nodeSize, nodeSize); // Draw node
        }
    });

    // Draw connections between nodes only if there are any layers
    if (layerVisuals.length > 1) {
        for (let i = 1; i < layerVisuals.length; i++) {
            let prevLayer = layerVisuals[i - 1];
            let currentLayer = layerVisuals[i];
            let prevXPos = xSpacing * i;
            let currentXPos = xSpacing * (i + 1);

            // Get weights of the current layer to draw connections
            let weights = model.layers[i].getWeights()[0].arraySync();

            for (let j = 0; j < prevLayer.nodes; j++) {
                for (let k = 0; k < currentLayer.nodes; k++) {
                    let prevYPos = ySpacing * (j + 1);
                    let currentYPos = ySpacing * (k + 1);

                    // Map the weight value to a stroke opacity between 50 and 255 for visibility
                    let weight = weights[j][k];
                    let opacity = map(Math.abs(weight), 0, 1, 50, 255); // Use absolute weight for opacity

                    // Change color based on the sign of the weight
                    if (weight > 0) {
                        canvas.stroke(0, 255, 0, opacity); // Green for positive weights
                    } else {
                        canvas.stroke(255, 0, 0, opacity); // Red for negative weights
                    }

                    canvas.line(prevXPos, prevYPos, currentXPos, currentYPos); // Draw line
                }
            }
        }
    }
}

// p5.js setup function
function setup() {

    console.log(p5); // This should log the p5 object if the library is loaded correctly
    // Fix: Explicitly create the canvases for data and network visualization
    // and assign them to the correct div elements with specific IDs.
    let dataVizContainer = select('#dataViz');
    let nnVizContainer = select('#nnViz');

    if (dataVizContainer) {
        dataCanvas = createCanvas(300, 300);
        dataCanvas.parent(dataVizContainer);
        dataCanvas.id('dataCanvas'); // Fix: Assign an ID to the data canvas
    }

    if (nnVizContainer) {
        networkCanvas = createCanvas(300, 300);
        networkCanvas.parent(nnVizContainer);
        networkCanvas.id('networkCanvas'); // Fix: Assign an ID to the network canvas
    }

    // Initial visualization before training
    selectDataset(); // This will prepare the initial dataset visualization
    visualizeData();
    drawNetwork(networkCanvas); // This will render the neural network visualization immediately
}
setup();
// Initialize the model when the script loads
initializeModel(document.getElementById('activation').value);
