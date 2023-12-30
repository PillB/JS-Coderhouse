// neuralNetworkPlayground.js

// TensorFlow.js model and visualization parameters
let model;
let canvas;
let layerVisuals = [];
let currentData = [];
let isTraining = false;
let currentEpoch = 0;
let totalEpochs = 100;
let lossHistory = [];
let resolution = 20;
let dataAreaWidth, networkAreaStartX;
let offscreenGraphics;

// p5.js structure
new p5(p => {
    let canvasWidth = 600; // Total width for both visualizations
    let canvasHeight = 300;

    // p5.js setup function
    p.setup = () => {
        let vizContainer = p.select('#Viz');
        canvas = p.createCanvas(canvasWidth, canvasHeight).parent(vizContainer).id('combinedCanvas');
        dataAreaWidth = canvasWidth / 2;
        networkAreaStartX = canvasWidth / 2;
        p.noLoop(); // Prevent draw loop unless we are training

        // Initialize TensorFlow.js model
        initializeModel(p.select('#activation').value());
        selectDataset();  // Prepare initial visualization
        offscreenGraphics = p.createGraphics(dataAreaWidth, canvasHeight);
        // Draw the points after the rest of the setup logic
        //drawMiddlePoint();
        //drawLowerLeftPoint();
        //drawMiddleDataPoint();
        //drawMiddleNetworkPoint();
        //runAllDataCanvasTests();
        //runVisualizationTests();
    };
    function runVisualizationTests() {
        visualizeData();  // This will include the visualization test
        drawDecisionBoundary();  // This will include the decision boundary test
      }
    // Inside the visualizeData function, add a test for correct scaling and positioning
function visualizeDataTest() {
    p.push();
    p.stroke('yellow');
    p.fill('yellow');
    currentData.forEach(point => {
      // Draw a yellow border around each data point to ensure it's within bounds
      p.rect(point.x * p.width, point.y * p.height, resolution, resolution);
    });
    p.pop();
  }
  // Inside the drawDecisionBoundary function, add a test for correct drawing
async function drawDecisionBoundaryTest() {
    let testRectsDrawn = 0;
    p.push();
    p.stroke('magenta');
    p.fill('magenta');
    // ...existing drawDecisionBoundary code...
    for (let i = 0; i < dataAreaWidth; i += resolution) {
      for (let j = 0; j < canvasHeight; j += resolution) {
        // ...existing drawing code...
        testRectsDrawn++;
      }
    }
    p.pop();
    console.log(`Test rects drawn: ${testRectsDrawn}`);
  }
  
    // Test to ensure data points are within the canvas bounds
function testDataBounds(data) {
    const outOfBounds = data.filter(point => 
        point.x < 0 || point.x > p.width || point.y < 0 || point.y > p.height
    );
    if (outOfBounds.length > 0) {
        console.error(`Out of bounds data points: ${outOfBounds.length}`);
        return false;
    }
    return true;
}

// Test to ensure that the data visualization is correctly scaled
function testVisualizationScaling() {
    const testPoint = { x: p.width / 2, y: p.height / 2, label: 0 };
    p.push();
    p.stroke(255, 0, 0);
    p.fill(255, 0, 0);
    p.ellipse(testPoint.x, testPoint.y, resolution, resolution);
    p.pop();
    console.log('Draw a red test point in the middle of the canvas to check scaling.');
}

// Test to ensure that the decision boundary drawing is synchronous
async function testDecisionBoundarySynchronization() {
    console.log('Starting decision boundary test...');
    await drawDecisionBoundary();
    console.log('Decision boundary test complete.');
}

// Test for memory leaks in TensorFlow.js tensors
function testTensorMemoryLeak() {
    const numTensorsBefore = tf.memory().numTensors;
    drawDecisionBoundary(); // Assume it's modified to be synchronous for testing
    const numTensorsAfter = tf.memory().numTensors;
    if (numTensorsBefore !== numTensorsAfter) {
        console.error(`Memory leak detected: ${numTensorsBefore} -> ${numTensorsAfter} tensors.`);
    }
}

// Run all tests
function runAllDataCanvasTests() {
    if (!testDataBounds(currentData)) {
        console.error('Data bounds test failed.');
    }
    testVisualizationScaling();
    testDecisionBoundarySynchronization();
    testTensorMemoryLeak();
}

    // Function to draw a point in the middle of the canvas
function drawMiddlePoint() {
    let midX = canvasWidth / 2;
    let midY = canvasHeight / 2;
    p.fill('black');
    p.ellipse(midX, midY, 10, 10); // Drawing a small ellipse to represent the point
}

// Function to draw a point in the lower left corner of the canvas
function drawLowerLeftPoint() {
    let cornerX = 0;
    let cornerY = canvasHeight;
    p.fill('black');
    p.ellipse(cornerX, cornerY, 10, 10);
}

// Function to draw a point in the middle of the data half of the canvas
function drawMiddleDataPoint() {
    let midDataX = dataAreaWidth / 2;
    let midDataY = canvasHeight / 2;
    p.fill('black');
    p.ellipse(midDataX, midDataY, 10, 10);
}

// Function to draw a point in the middle of the network half of the canvas
function drawMiddleNetworkPoint() {
    let midNetworkX = networkAreaStartX + (canvasWidth - networkAreaStartX) / 2;
    let midNetworkY = canvasHeight / 2;
    p.fill('black');
    p.ellipse(midNetworkX, midNetworkY, 10, 10);
}
  // Function to update the training status and loss graph
  function updateTrainingStatus(epochs, loss) {
    p.select('#epochInfo').html(`Epoch: ${epochs}/${totalEpochs}`);
    p.select('#lossInfo').html(`Loss: ${loss.toFixed(4)}`);
    const lossGraphElement = p.select('#lossGraph');
    lossGraphElement.html(''); // Clear previous graph contents
    lossHistory.forEach((lossVal, index) => {
      const lossValueElement = p.createElement('div', `Epoch ${index + 1}: ${lossVal}`);
      lossGraphElement.child(lossValueElement);
    });
  }

  // Function to prepare visualization data
  function prepareVisualizationData() {
    layerVisuals = [];
    model.layers.forEach(layer => {
      let layerInfo = {
        nodes: layer.units,
        activations: Array(layer.units).fill(0)
      };
      layerVisuals.push(layerInfo);
    });
  }

  // Function to initialize the TensorFlow.js model
  function initializeModel(activation) {
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 5, inputShape: [2], activation: activation }));
    model.add(tf.layers.dense({ units: 4, activation: activation }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(parseFloat(p.select('#learningRate').value())),
      loss: 'binaryCrossentropy'
    });

    prepareVisualizationData();
  }

  // Function to select dataset based on dropdown
  function selectDataset() {
    let selection = p.select('#dataShape').value();
    currentData = generateData(selection);
    isTraining = false;
    updateDataVisualization();
  }

  // ...continuing from the previous code...

  // Function to generate data based on selection
  function generateData(selection) {
    let data = [];
    switch (selection) {
      case 'linear':
        data = generateLinearData();
        break;
      case 'concentric':
        data = generateConcentricData();
        break;
      case 'twoClusters':
        data = generateTwoClustersData();
        break;
      case 'checkerboard':
        data = generateCheckerboardData();
        break;
      case 'moons':
        data = generateMoonsData();
        break;
      case 'spiral':
        data = generateSpiralData();
        break;
      default:
        data = [];
    }
    return data.map(d => ({
      x: d.x,
      y: d.y,
      label: d.label
    }));
  }
  // Function to draw plus signs for the visualization
  function drawPlusSign(x, y, size) {
    const halfSize = size / 2;
    canvas.strokeWeight(2);
    canvas.stroke(0);
    canvas.line(x - halfSize, y, x + halfSize, y);
    canvas.line(x, y - halfSize, x, y + halfSize);
  }
  function visualizeData() {
    let transformedPoints = []; // Array to store transformed points
    console.log(`Data area dims: ${p.width / 2}x${p.height}`);
    // Loop through each point in the current dataset
    for (let i = 0; i < currentData.length; i++) {
        let point = currentData[i];
        console.log(`In Loop: Data area dims: ${p.width / 2}x${p.height}`);
        
        let transformedX = (resolution/2 + point.x * p.width / 2);
        let transformedY = (resolution/2 + point.y * p.height);

        // Store the transformed points
        transformedPoints.push({ x: transformedX, y: transformedY, label: point.label });

        // Use the appropriate canvas context
        
        // Fix: Ensure that points of both classes are visualized
        if (point.label === 0) {
            p.stroke('blue');
            p.fill('blue'); // Class 0 as circles
            p.ellipse(transformedX, transformedY, resolution/4, resolution/4);
        } else {
            p.stroke(0);
            p.fill('red'); // Class 1 as plus signs
            drawPlusSign(transformedX, transformedY, resolution/4);
        }
    }

    // Log the array of transformed points to the console
    console.log("Transformed Points: ", transformedPoints);
}

// Call this function in the appropriate place, such as after data generation or within p.setup().

    // Add a function to check for NaN or infinity in the dataset
function validateData(data) {
    return data.every(point => 
        !isNaN(point.x) && isFinite(point.x) && 
        !isNaN(point.y) && isFinite(point.y) && 
        (point.label === 0 || point.label === 1)
    );
}

// Function to select dataset based on dropdown
function selectDataset() {
    let selection = p.select('#dataShape').value();
    currentData = generateData(selection);
    if (!validateData(currentData)) {
        console.error('Data validation failed: dataset contains NaN, infinity, or invalid labels.');
        return; // Abort if data is not valid
    }
    isTraining = false;
    updateDataVisualization();
}
  // Function to validate the data shape and type for TensorFlow.js compatibility
function validateDataShapeAndType(data) {
    if (!Array.isArray(data)) {
        console.error('Data is not an array.');
        return false;
    }
    if (data.length === 0) {
        console.error('Data array is empty.');
        return false;
    }
    for (const point of data) {
        if (typeof point.x !== 'number' || typeof point.y !== 'number' || typeof point.label !== 'number') {
            console.error(`Data point has incorrect types. Expected numbers, got x: ${typeof point.x}, y: ${typeof point.y}, label: ${typeof point.label}`);
            return false;
        }
    }
    return true;
}

// Function to validate model architecture against expected specifications
// Function to validate model architecture against expected specifications
function validateModelArchitecture(model) {
    try {
        if (!model || !model.layers || model.layers.length === 0) {
            console.error('Model or model layers are not defined.');
            return false;
        }

        // Check for input layer compatibility
        const inputLayer = model.layers[0];
        if (!inputLayer) {
            console.error('Input layer is not defined.');
            return false;
        }
        // TensorFlow.js might use different ways to store input shape. Adjust accordingly.
        const inputShape = inputLayer.inputShape || (inputLayer.batchInputShape && inputLayer.batchInputShape.slice(1));
        if (!inputShape || inputShape.length !== 2) {
            console.error(`Input layer shape is incorrect or not defined. Found: ${inputShape}`);
            return false;
        }

        // Check for output layer compatibility for binary classification
        const outputLayer = model.layers[model.layers.length - 1];
        if (!outputLayer || outputLayer.units !== 1) {
            console.error('Output layer units is incorrect or not defined. Expected 1 unit for binary classification.');
            return false;
        }
        
        // Ensure all layers have appropriate activation functions and other constraints if necessary
        
    } catch (error) {
        console.error('Error validating model architecture:', error.message);
        return false;
    }
    return true;
}


// Function to check for NaN values in tensors
function checkForNaN(tensor, message = 'Tensor') {
    if (tensor.any(tf.isnan()).dataSync()[0]) {
        console.error(`${message} contains NaN values.`);
        return true;
    }
    return false;
}

// Enhanced data validation before training
function preTrainingChecks() {
    if (!validateDataShapeAndType(currentData)) {
        console.error('Pre-training check failed: Data shape or type is incorrect.');
        return false; // Abort if data shape or type is not valid
    }

    if (!validateModelArchitecture(model)) {
        console.error('Pre-training check failed: Model architecture validation failed.');
        return false; // Abort if model architecture is not valid
    }

    // Check if data contains NaN or Inf values
    const inputs = tf.tensor2d(currentData.map(p => [p.x, p.y]));
    const labels = tf.tensor2d(currentData.map(p => [p.label]));
    if (checkForNaN(inputs, 'Input tensor') || checkForNaN(labels, 'Label tensor')) {
        inputs.dispose();
        labels.dispose();
        return false; // Abort if any tensor contains NaN
    }

    // Clean up tensors
    inputs.dispose();
    labels.dispose();
    return true;
  }

  // Function to train the model
  async function trainModel() {
    // Initialize or re-initialize the model with the selected activation function
    let activation = document.getElementById('activation').value;
    initializeModel(activation);
    generateDecisionGrid();
    isTraining = true;
    //if (!preTrainingChecks()) return; 
    console.log(`Training with activation: ${activation}, learning rate: ${parseFloat(p.select('#learningRate').value())}`);
    if (!validateData(currentData)) {
        console.error('Data validation failed before training: dataset contains NaN, infinity, or invalid labels.');
        return; // Abort training if data is not valid
    }
    p.loop(); // Start the draw loop
    // Disable the dataShape selector during training
    document.getElementById('dataShape').disabled = true;
    document.getElementById('generateDataButton').disabled = true;
    document.getElementById('trainModelButton').disabled = true;
    document.getElementById('visualizeDataFlowButton').disabled = true;
    // Reset global tracking variables
    currentEpoch = 0;
    lossHistory = [];
    // Update the DOM elements for epochs and loss before training starts
    updateTrainingStatus(0, 0);
    let inputs = tf.tensor2d(currentData.map(p => [p.x, p.y]));
    let labels = tf.tensor2d(currentData.map(p => [p.label]));
    //const inputs = tf.tensor2d(currentData.map(p => [p.x / p.width, p.y / p.height]), [currentData.length, 2]);
    //const labels = tf.tensor2d(currentData.map(p => [p.label]), [currentData.length, 1]); // Labels must also be a 2D tensor

    await model.fit(inputs, labels, {
      epochs: totalEpochs,
      callbacks: {
        onEpochEnd: async (epoch, log) => {
          currentEpoch = epoch + 1;
          lossHistory.push(log.loss);
          updateTrainingStatus(currentEpoch, log.loss);
          if (isNaN(log.loss)) {
            console.error('NaN loss detected. Check data, learning rate, and model architecture.');
            model.stopTraining = true; // Stop training if NaN loss is detected
          } else { 
            console.log(`Succeded in training epoch ${currentEpoch}. Preparing visualization step.`);
          }
          // Update visualization after each epoch
          await updateDataVisualization();
          p.redraw(); // Redraw only when needed during training
        }
      }
    });
    isTraining = false;
    p.noLoop(); // Stop the draw loop
    // Re-enable the dataShape selector after training
    document.getElementById('dataShape').disabled = false;
    document.getElementById('generateDataButton').disabled = false;
    document.getElementById('trainModelButton').disabled = false;
    document.getElementById('visualizeDataFlowButton').disabled = false;
    document.getElementById('output').innerText = 'Training complete!';
  }

  // Function to draw the neural network visualization
  // Function to draw the neural network visualization
  function drawNetwork() {
    let networkAreaWidth = canvasWidth / 2;
    let xSpacing = networkAreaWidth / (layerVisuals.length + 1);
    let ySpacing = canvasHeight / (Math.max(...layerVisuals.map(l => l.nodes)) + 1);

    // Drawing connections between nodes
    for (let i = 1; i < layerVisuals.length; i++) {
        let prevLayer = layerVisuals[i - 1];
        let currentLayer = layerVisuals[i];
        let prevX = networkAreaStartX + xSpacing * i;
        let currentX = networkAreaStartX + xSpacing * (i + 1);

        for (let j = 0; j < prevLayer.nodes; j++) {
            for (let k = 0; k < currentLayer.nodes; k++) {
                let prevY = ySpacing * (j + 1);
                let currentY = ySpacing * (k + 1);
                let weight = model.layers[i].getWeights()[0].arraySync()[j][k];
                let opacity = p.map(Math.abs(weight), 0, 1, 50, 255); // Use absolute weight for opacity
                let strokeColor = weight > 0 ? p.color(0, 255, 0, opacity) : p.color(255, 0, 0, opacity);
                p.stroke(strokeColor);
                p.line(prevX, prevY, currentX, currentY);
            }
        }
    }

    // Drawing nodes on top of connections
    for (let i = 0; i < layerVisuals.length; i++) {
        let layer = layerVisuals[i];
        let x = networkAreaStartX + xSpacing * (i + 1);
        for (let j = 0; j < layer.nodes; j++) {
            let y = ySpacing * (j + 1);
            let activation = layer.activations[j];
            let nodeColor = p.lerpColor(p.color(255, 255, 255), p.color(0, 0, 255), activation);
            p.fill(nodeColor);
            p.stroke(0);
            p.ellipse(x, y, 20, 20);
        }
    }
  }
  
  // Function to update data visualization
  async function updateDataVisualization() {
    //p.background(255);
    if (isTraining) {
        console.log(`Updating decision boundary`)
        await drawDecisionBoundary();  // Update the data visualization area with decision boundary
    }
    console.log(`Updating network`)
    drawNetwork();  // This will draw on the right side
    console.log(`Updating data visualization`)
    visualizeData();  // This will draw on the left side
  }
  let decisionBoundaryGrid = [];

    function generateDecisionGrid() {
        decisionBoundaryGrid = []; // Clear previous grid
        let gridResolution = resolution / 2; // Adjust grid resolution as needed
        for (let i = 0; i < dataAreaWidth; i += gridResolution) {
            for (let j = 0; j < canvasHeight; j += gridResolution) {
                let normX = i / dataAreaWidth;
                let normY = j / canvasHeight;
                decisionBoundaryGrid.push([normX, normY]);
            }
        }
    }

  // Function to draw the decision boundary
  async function drawDecisionBoundary() {
    offscreenGraphics.clear(); // Clear the off-screen graphics
    let inputTensor = tf.tensor2d(decisionBoundaryGrid);
    let predictions = await model.predict(inputTensor).data();
    let index = 0;
    let gridResolution = resolution / 2;

    for (let i = 0; i < dataAreaWidth; i += gridResolution) {
        for (let j = 0; j < canvasHeight; j += gridResolution) {
            let predictedClass = predictions[index++] > 0.5 ? 1 : 0;
            if (predictedClass === 0) {
                offscreenGraphics.fill('rgba(0, 0, 255, 0.5)');
            } else {
                offscreenGraphics.fill('rgba(255, 0, 0, 0.5)');
            }
            offscreenGraphics.noStroke();
            offscreenGraphics.rect(resolution/4 + i, resolution/4 + j, gridResolution, gridResolution);
        }
    }
    inputTensor.dispose();
    p.clear()
    // Now draw the off-screen buffer to the main canvas
    p.image(offscreenGraphics, 0, 0);
}

    //drawDecisionBoundaryTest();

  // ... (include all data generation functions here) ...
  // Function to generate a linearly separable dataset
function generateLinearData() {
    let data = [];
    for (let x = 0; x < dataAreaWidth; x += resolution) {
        for (let y = 0; y < canvasHeight; y += resolution) {
            let classLabel = x < dataAreaWidth / 2 ? 0 : 1;
            data.push({x: x / dataAreaWidth, y: y / canvasHeight, label: classLabel});
        }
    }

    console.log("Linear Data Points:", data);
    return data;
}

// Function to generate concentric circles dataset
function generateConcentricData() {
    let data = [];
    let centerX = 0.5;
    let centerY = 0.5;
    let maxRadius = 0.5 - 0.1; // Padding of 0.1 on a normalized scale

    for (let angle = 0; angle < p.TWO_PI; angle += 0.1) {
        let innerRadius = maxRadius / 3;
        let outerRadius = 2 * maxRadius / 3;

        // Inner circle (class 0)
        let xInner = centerX + innerRadius * p.cos(angle);
        let yInner = centerY + innerRadius * p.sin(angle);
        data.push({x: xInner, y: yInner, label: 0});

        // Outer ring (class 1)
        let xOuter = centerX + outerRadius * p.cos(angle);
        let yOuter = centerY + outerRadius * p.sin(angle);
        data.push({x: xOuter, y: yOuter, label: 1});
    }

    console.log("Concentric Data Points:", data);
    return data;
}

// Function to generate two clusters dataset
function generateTwoClustersData() {
    let data = [];
    let pointsPerCluster = 100;
    let mean1 = [0.3, 0.3];
    let mean2 = [0.7, 0.7];

    for (let i = 0; i < pointsPerCluster; i++) {
        let x = p.randomGaussian(mean1[0], 0.05);
        let y = p.randomGaussian(mean1[1], 0.05);
        data.push({x: x, y: y, label: 0});

        x = p.randomGaussian(mean2[0], 0.05);
        y = p.randomGaussian(mean2[1], 0.05);
        data.push({x: x, y: y, label: 1});
    }

    console.log("Cluster Data Points:", data);
    return data;
}

// Function to generate checkerboard dataset
function generateCheckerboardData() {
    let data = [];
    let tileSize = 1 / 10; // normalized tile size

    for (let x = 0; x < 1; x += tileSize) {
        for (let y = 0; y < 1; y += tileSize) {
            let classLabel = (p.floor(x / tileSize) + p.floor(y / tileSize)) % 2;
            data.push({x: x + tileSize / 2, y: y + tileSize / 2, label: classLabel});
        }
    }

    console.log("Checkerboard Data Points:", data);
    return data;
}

// Function to generate moons dataset
function generateMoonsData() {
    let data = [];
    let pointsPerMoon = 100;
    let radius = 0.2;

    for (let i = 0; i < pointsPerMoon; i++) {
        let angle = p.random(Math.PI);
        let x = 0.5 + radius * p.cos(angle);
        let y = 0.5 + radius * p.sin(angle);
        data.push({x: x, y: y, label: 0});

        x = 0.5 + radius * p.cos(angle + Math.PI);
        y = 0.5 + radius * p.sin(angle + Math.PI);
        data.push({x: x, y: y, label: 1});
    }

    console.log("Moons Data Points:", data);
    return data;
}

// Function to generate spiral dataset
function generateSpiralData() {
    let data = [];
    let turns = 2;
    let pointsPerTurn = 100;

    // Generate points for the first spiral
    for (let i = 0; i < turns * pointsPerTurn; i++) {
        let angle = p.map(i, 0, turns * pointsPerTurn, 0, turns * p.TWO_PI);
        let radius = p.map(i, 0, turns * pointsPerTurn, 0, 1 / 3);
        let x = 1 / 2 + radius * p.cos(angle);
        let y = 1 / 2 + radius * p.sin(angle);
        data.push({x: x, y: y, label: 0}); // Assign label 0 to all points of the first spiral
    }

    // Generate points for the second spiral
    for (let i = 0; i < turns * pointsPerTurn; i++) {
        let angle = p.map(i, 0, turns * pointsPerTurn, 0, turns * p.TWO_PI) + Math.PI;
        let radius = p.map(i, 0, turns * pointsPerTurn, 0, 1 / 3);
        let x = 1 / 2 + radius * p.cos(angle);
        let y = 1 / 2 + radius * p.sin(angle);
        data.push({x: x, y: y, label: 1}); // Assign label 1 to all points of the second spiral
    }
    console.log("Spirals Data Points:", data);
    return data;
}


  // Adjusting event listeners for buttons using p5.js methods
  p.select('#generateDataButton').mouseClicked(() => {
    selectDataset()
    updateDataVisualization();
  });

  p.select('#trainModelButton').mouseClicked(() => {
    trainModel();
  });
  // Ensure to call the setup function to initialize everything
  p.setup();

});
