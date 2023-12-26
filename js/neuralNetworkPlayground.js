// neuralNetworkPlayground.js

// Define a simple model
const model = tf.sequential();

// Visualization parameters
let layerVisuals = []; // To store layer visualization data

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
            currentData = generateTwoClusters();
            break;
        case 'checkerboard':
            currentData = generateCheckerboard();
            break;
        case 'moons':
            currentData = generateMoons();
            break;
        case 'spiral':
            currentData = generateSpiral();
            break;
        // Add more cases as needed
    }
    isTraining = false; // Reset training flag
    updateDataVisualization(); // Visualize the new dataset
}
// Generates a linearly separable dataset
function generateLinearData() {
    let data = [];
    for (let x = 0; x < width; x += resolution) {
        for (let y = 0; y < height; y += resolution) {
            let classLabel = x < width / 2 ? 0 : 1;
            data.push({x: x, y: y, label: classLabel});
        }
    }
    return data;
}

// Generates concentric circles dataset
function generateConcentricData() {
    let data = [];
    let centerX = width / 2;
    let centerY = height / 2;
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

function generateTwoClusters() {
    let data = [];
    let pointsPerCluster = 100;

    // Parameters for the first cluster
    let mean1 = [width * 0.3, height * 0.3];
    let variance1 = [width * 0.05, height * 0.05];

    // Parameters for the second cluster
    let mean2 = [width * 0.7, height * 0.7];
    let variance2 = [width * 0.05, height * 0.05];

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

function generateCheckerboard() {
    let data = [];
    let tileSize = width / 10; // Number of tiles per row/column

    for (let x = 0; x < width; x += tileSize) {
        for (let y = 0; y < height; y += tileSize) {
            let classLabel = (floor(x / tileSize) + floor(y / tileSize)) % 2;
            data.push({x: x + tileSize / 2, y: y + tileSize / 2, label: classLabel});
        }
    }

    return data;
}

function generateMoons() {
    let data = [];
    let pointsPerMoon = 100;
    let radius = width * 0.2;

    // Generate points for the first moon
    for (let i = 0; i < pointsPerMoon; i++) {
        let angle = random(Math.PI); // half circle
        let x = width * 0.5 + radius * cos(angle);
        let y = height * 0.5 + radius * sin(angle);
        data.push({x: x, y: y, label: 0});
    }

    // Generate points for the second moon
    for (let i = 0; i < pointsPerMoon; i++) {
        let angle = random(Math.PI); // half circle
        let x = width * 0.5 + radius * cos(angle + Math.PI);
        let y = height * 0.5 + radius * sin(angle + Math.PI);
        data.push({x: x, y: y, label: 1});
    }

    return data;
}

function generateSpiral() {
    let data = [];
    let turns = 2;
    let pointsPerTurn = 100;

    // Generate points for the first spiral
    for (let i = 0; i < turns * pointsPerTurn; i++) {
        let angle = map(i, 0, turns * pointsPerTurn, 0, turns * TWO_PI);
        let radius = map(i, 0, turns * pointsPerTurn, 0, width / 3);
        let x = width / 2 + radius * cos(angle);
        let y = height / 2 + radius * sin(angle);
        data.push({x: x, y: y, label: i % 2});
    }

    // Generate points for the second spiral
    for (let i = 0; i < turns * pointsPerTurn; i++) {
        let angle = map(i, 0, turns * pointsPerTurn, 0, turns * TWO_PI) + Math.PI;
        let radius = map(i, 0, turns * pointsPerTurn, 0, width / 3);
        let x = width / 2 + radius * cos(angle);
        let y = height / 2 + radius * sin(angle);
        data.push({x: x, y: y, label: (i + 1) % 2});
    }

    return data;
}


let currentData = []; // Holds the current dataset
let isTraining = false; // Flag to check if training has started

// Add this function to draw different shapes for different classes
function visualizeData() {
    dataCanvas.clear(); // Clear the data canvas to prevent drawing over the network canvas
    // Loop through each point in the current dataset
    for (let i = 0; i < currentData.length; i++) {
        const point = currentData[i];
        // Use the appropriate canvas context (dataCanvas)
        dataCanvas.stroke(0);
        if (point.label === 0) {
            dataCanvas.fill('blue'); // Class 0
            dataCanvas.ellipse(point.x, point.y, resolution, resolution); // Draw circle
        } else {
            dataCanvas.fill('red'); // Class 1
            drawPlusSign(dataCanvas, point.x, point.y, resolution); // Draw plus sign
        }
    }
}
// Function to train the model
async function trainModel() {
    // Initialize or re-initialize the model with the selected activation function
    let activation = document.getElementById('activation').value;
    initializeModel(activation);

    // Convert currentData to tensors with proper shapes
    // The shape is [num_examples, num_features_per_example]
    const inputs = tf.tensor2d(currentData.map(p => [p.x / width, p.y / height]), [currentData.length, 2]);
    const labels = tf.tensor2d(currentData.map(p => [p.label]), [currentData.length, 1]); // Labels must also be a 2D tensor


    // Train the model
    await model.fit(inputs, labels, {
        epochs: 100,
        callbacks: {
            onEpochEnd: async (epoch, log) => {
                console.log(`Epoch ${epoch}: loss = ${log.loss}`);
                // Update visualization after each epoch
                await updateDataVisualization();
            }
        }
    });

    isTraining = true; // Set training flag to true
    // Update the UI with training results
    document.getElementById('output').innerText = 'Training complete!';
}

let dataCanvas, networkCanvas; // Separate canvases for data and network

// Define the resolution of the grid (lower = more points = slower)
const resolution = 20;

// Generate a grid of points across the canvas
function generateGrid() {
  let points = [];
  for (let x = 0; x < width; x += resolution) {
    for (let y = 0; y < height; y += resolution) {
      points.push([x, y]);
    }
  }
  return points;
}

// p5.js draw function
function draw() {
    // Clear the backgrounds
    networkCanvas.background(200);
    dataCanvas.background(255);
    
    // Update the data visualization only if training has started
    if (isTraining) {
        updateDataVisualization();
    }
    // Draw the neural network visualization
    drawNetwork(networkCanvas);
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
    draw();
}
// Function to predict and draw the decision boundary
async function drawDecisionBoundary() {
    // Use currentData instead of generateGrid
    // Predict classes for each point in the currentData
    let predictions = await model.predict(tf.tensor2d(currentData.map(p => [p.x / width, p.y / height]))).data();

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
    // Only update during training
    if (!isTraining) return;

    // Clear the canvas
    dataCanvas.background(255);
    // Draw the ground truth of the data
    visualizeData();

    // Draw decision boundary based on current model predictions
    if (isTraining) {
        drawDecisionBoundary();
    }
}

// Function to draw the neural network visualization
function drawNetwork(canvas) {
    canvas.clear(); // Use clear instead of background to avoid covering the entire canvas
    canvas.strokeWeight(1); // Set stroke weight for visibility

    // Draw each layer
    layerVisuals.forEach((layer, i) => {
        let xSpacing = canvas.width / (layerVisuals.length + 1);
        let ySpacing = canvas.height / (Math.max(...layerVisuals.map(l => l.nodes)) + 1);
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
    // Adjust to create separate canvases for data and network visualization
    let dataVizContainer = select('#dataViz');
    let nnVizContainer = select('#nnViz');

    // Check if the containers exist before creating canvases
    if (dataVizContainer) {
        dataCanvas = createCanvas(300, 400);
        dataCanvas.parent('dataViz');
    }

    if (nnVizContainer) {
        networkCanvas = createCanvas(300, 400);
        networkCanvas.parent('nnViz');
    }

    // Initial visualization before training
    visualizeData();
    drawNetwork(networkCanvas); // Pass the correct canvas
}

// Function to update visualization with latest activations
function updateVisualization() {
    // Ideally, here you would extract the activations from the model and update 'layerVisuals'
    // This is a placeholder to show how you might update the visualization
    layerVisuals.forEach(layer => {
        layer.activations = layer.activations.map(() => Math.random()); // Random activations for demonstration
    });
    // Redraw the network
    draw();
}

// Initialize the model when the script loads
initializeModel(document.getElementById('activation').value);
