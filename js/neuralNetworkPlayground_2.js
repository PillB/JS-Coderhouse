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
let iterations = 100;
let username;
// p5.js structure
new p5(p => {
    let canvasWidth = 600; // Total width for both visualizations
    let canvasHeight = 300;
    let activation; 
    if (localStorage.getItem("username") === null) {
        username = window.prompt("Please enter your name","Your name here.");
        localStorage.setItem('username',username);
        alert("Hello, "+username+"!");
    }else{
        username = localStorage.getItem('username');
        alert("Welcome back, "+username+"!");
    }
    // p5.js setup function
    p.setup = () => {
        let vizContainer = p.select('#Viz');
        do{
            iterations= parseInt(window.prompt("Please enter the number of epochs to train the model (training takes aprox. 1 sec per epoch)\nA higher epoch count generate a more accurate model.", 100));
        }while(isNaN(iterations)|| iterations < 1);
        
        totalEpochs = iterations;
        p.select('#epochInfo').html(`Epoch: 0/${totalEpochs}`);
        canvas = p.createCanvas(canvasWidth, canvasHeight).parent(vizContainer).id('combinedCanvas');
        dataAreaWidth = canvasWidth / 2;
        networkAreaStartX = canvasWidth / 2;
        p.noLoop(); // Prevent draw loop unless we are training
        let activation = document.getElementById('activation').value;
        // Initialize TensorFlow.js model
        // Initial setup call to populate the architecture table
        initializeModel(p.select('#activation').value());
        createArchitectureTable();
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
  
  // Function to get the layers configuration from the TensorFlow.js model or return the default
function getLayersFromModel() {
    // Check if the model is defined
    if (typeof model === 'undefined' || model.layers.length === 0) {
        // Return the default architecture
        return [
            { units: 5, activation: 'sigmoid' }, // Input layer
            { units: 4, activation: 'sigmoid' }, // Hidden layer
            { units: 1, activation: 'sigmoid' } // Output layer
        ];
    } else {
        // Extract the layers information from the model
        return model.layers.map((layer, index) => {
            // Assuming 'activation' is a property of the layer's configuration
            let units = layer.units;
            let activation = layer.activation ? layer.activation : (index === model.layers.length - 1) ? 'sigmoid' : 'relu';
            return { units, activation };
        });
    }
}

function getLayersFromTable(activation) {
    const tableBody = document.getElementById('architectureTable').querySelector('tbody');
    let layers = [];

    // Skip the first row (Input layer) and last row (Output layer)
    for (let i = 1; i < tableBody.rows.length - 1; i++) {
        let row = tableBody.rows[i];
        let units = parseInt(row.cells[1].innerText, 10);
        //let activation = 'relu'; // Default activation for hidden layers
        layers.push({ units, activation });
    }

    return layers;
}


function initializeModel(activation) {
    // Clear the previous model if it exists
    if (typeof model !== 'undefined') {
        model.dispose();
    }

    // Create the model
    model = tf.sequential();

    // Add input layer (hardcoded as 2 units here)
    model.add(tf.layers.dense({ units: 2, inputShape: [2], activation: activation }));

    const layers = getLayersFromTable(activation); // Ignore the first and last rows in this function
    layers.forEach((layer) => {
        model.add(tf.layers.dense({ units: layer.units, activation: activation }));
    });

    // Add output layer (hardcoded as 1 unit here)
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: tf.train.adam(parseFloat(p.select('#learningRate').value())),
        loss: 'binaryCrossentropy'
    });

    prepareVisualizationData();
}


// Function to update the event listeners for the buttons
function updateEventListeners() {
    const tableBody = document.getElementById('architectureTable').querySelector('tbody');

    // Clear existing event listeners
    ['add-unit', 'remove-unit', 'remove-layer'].forEach(className => {
        const buttons = tableBody.querySelectorAll(`.${className}`);
        buttons.forEach(button => button.onclick = null);
    });

    // Add new event listeners
    const addUnitButtons = tableBody.querySelectorAll('.add-unit');
    const removeUnitButtons = tableBody.querySelectorAll('.remove-unit');
    const removeLayerButtons = tableBody.querySelectorAll('.remove-layer');

    addUnitButtons.forEach((button, index) => {
        // Correcting index by adding 1 to account for input layer
        button.onclick = () => modifyUnits(index + 1, 1);
    });

    removeUnitButtons.forEach((button, index) => {
        // Correcting index by adding 1 to account for input layer
        button.onclick = () => modifyUnits(index + 1, -1);
    });

    removeLayerButtons.forEach((button, index) => {
        // Correcting index by adding 1 to account for input layer
        button.onclick = () => removeLayer(index + 1);
    });
}

// Function to create the initial architecture table based on the current model
function createArchitectureTable() {
    const tableBody = document.getElementById('architectureTable').querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    // Add input layer row
    const inputLayerRow = document.createElement('tr');
    inputLayerRow.innerHTML = `<td>Input Layer</td><td>2</td><td></td>`; // Input layer is fixed and has no actions
    tableBody.appendChild(inputLayerRow);

    // Add hidden layers from the model
    const layers = getLayersFromModel();
    layers.forEach((layer, index) => {
        if (index === 0 || index === layers.length - 1) return; // Skip actual input and output layer

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>Hidden Layer ${index}</td>
            <td>${layer.units}</td>
            <td>
                <button class="add-unit" id="add-unit-${index}">Add Unit</button>
                <button class="remove-unit" id="remove-unit-${index}">Remove Unit</button>
                <button class="remove-layer" id="remove-layer-${index}">Remove Layer</button>
            </td>
        `;
        tableBody.appendChild(row);
    });

    // Add output layer row
    const outputLayerRow = document.createElement('tr');
    outputLayerRow.innerHTML = `<td>Output Layer</td><td>1</td><td></td>`; // Output layer is fixed and has no actions
    tableBody.appendChild(outputLayerRow);

    // Initialize the model with the current architecture and update the visualization
    initializeModel(document.getElementById('activation').value);
    updateDataVisualization();
    updateEventListeners(); // Re-assign event listeners
}

// Function to add a new layer to the model
function addLayer() {
    const tableBody = p.select('#architectureTable tbody');
    const newRow = p.createElement('tr');
    const layerIndex = tableBody.child().length + 1; // Get the next layer index
    newRow.child(p.createElement('td', `Layer ${layerIndex}`));
    newRow.child(p.createElement('td', '1')); // Start with 1 unit for new layer

    const actionsCell = p.createElement('td');
    const addButton = p.createElement('button', 'Add Unit');
    addButton.mouseClicked(() => modifyUnits(layerIndex - 1, 1));
    actionsCell.child(addButton);

    const removeButton = p.createElement('button', 'Remove Unit');
    removeButton.mouseClicked(() => modifyUnits(layerIndex - 1, -1));
    actionsCell.child(removeButton);

    const removeLayerButton = p.createElement('button', 'Remove Layer');
    removeLayerButton.mouseClicked(() => removeLayer(layerIndex - 1));
    actionsCell.child(removeLayerButton);

    newRow.child(actionsCell);
    tableBody.child(newRow);
    initializeModel(document.getElementById('activation').value);
    updateDataVisualization();
}

// Function to modify the number of units in a layer
function modifyUnits(layerIndex, change) {
    const tableBody = document.getElementById('architectureTable').querySelector('tbody');
    const row = tableBody.rows[layerIndex];
    let units = parseInt(row.cells[1].innerText);
    units += change;
    if (units < 1) units = 1; // Minimum of 1 unit
    row.cells[1].innerText = units;

    // Reinitialize the model with updated units and redraw the visualization
    initializeModel(document.getElementById('activation').value);
    updateDataVisualization();
}

// Function to remove a layer
function removeLayer(layerIndex) {
    const tableBody = document.getElementById('architectureTable').querySelector('tbody');
    tableBody.deleteRow(layerIndex); // Remove the layer from the table

    // Reinitialize the model without the removed layer and redraw the visualization
    initializeModel(document.getElementById('activation').value);
    updateDataVisualization();
    updateEventListeners(); // Update event listeners as they may have shifted
}


// Event listener for adding a new layer
p.select('#addLayerButton').mouseClicked(addLayer);



  // Function to select dataset based on dropdown
  function selectDataset() {
    let selection = p.select('#dataShape').value();
    currentData = generateData(selection);
    isTraining = false;
    updateDataVisualization();
  }

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
    activation = document.getElementById('activation').value;
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
  function drawNetwork() {
    // Verify the canvas and model are defined
    if (!canvas || !model) {
        console.error('Canvas or model is undefined.');
        return;
    }
    let networkAreaWidth = canvasWidth / 2;
    let xSpacing = networkAreaWidth / (layerVisuals.length + 1);
    let ySpacing = canvasHeight / (Math.max(2, ...layerVisuals.map(l => l.nodes), 1) + 1); // Including input and output layer nodes

    // Drawing connections
    // Start from the input layer (which has 2 units) and draw to the first hidden layer
    let prevLayerNodes = 2; // Input layer has 2 units
    let prevX = networkAreaStartX + xSpacing; // X position for the input layer

    for (let i = 0; i < layerVisuals.length; i++) {
        let currentLayer = layerVisuals[i];
        let currentX = networkAreaStartX + xSpacing * (i + 1); // Adjust for 0-based index and add 1 for input layer offset
        //canvas.clear(); // Use clear instead of background to avoid covering the entire canvas
        canvas.strokeWeight(1); // Set stroke weight for visibility
        // Add check to verify model layer and weights exist
        if (model.layers[i] && model.layers[i].getWeights().length > 0) {
            let weights = model.layers[i].getWeights()[0].arraySync();
            for (let j = 0; j < prevLayerNodes; j++) {
                for (let k = 0; k < currentLayer.nodes; k++) {
                    let prevY = ySpacing * (j + 1);
                    let currentY = ySpacing * (k + 1);

                    // Access the correct weight matrix from the model
                    //let weightMatrixIndex = i == 0 ? 0 : i - 1;
                    //let weight = model.layers[weightMatrixIndex].getWeights()[0].arraySync()[j][k];
                    let weight = weights[j][k];
                    let opacity = p.map(Math.abs(weight), 0, 1, 20, 255);
                    let thickness = p.map(Math.abs(weight), 0, 1, 2, 6);
                    let strokeColor = weight > 0 ? p.color(0, 255, 0, opacity) : p.color(255, 0, 0, opacity);

                    p.strokeWeight(thickness);
                    p.stroke(strokeColor);
                    p.line(prevX, prevY, currentX, currentY);
                    p.strokeWeight(2);
                }
            }
        } else {
            console.error('Model layer or weights at index ' + i + ' are undefined.');
        }

        // Set up for the next layer
        prevLayerNodes = currentLayer.nodes;
        prevX = currentX;
    }

    // Drawing nodes
    // Include the input layer with 2 units
    let inputLayerX = networkAreaStartX + xSpacing;
    for (let j = 0; j < 2; j++) { // Draw 2 nodes for input layer
        let y = ySpacing * (j + 1);
        p.fill(255); // Assuming input nodes are white
        p.stroke(0);
        p.ellipse(inputLayerX, y, 20, 20);
    }

    // Draw hidden and output layers
    for (let i = 0; i < layerVisuals.length; i++) {
        let layer = layerVisuals[i];
        let x = networkAreaStartX + xSpacing * (i + 1); // Adjust for 0-based index and add 1 for input layer offset

        for (let j = 0; j < layer.nodes; j++) {
            let y = ySpacing * (j + 1);
            let activation = layer.activations[j];
            let nodeColor = p.lerpColor(p.color(255, 255, 255), p.color(0, 0, 255), activation);

            p.fill(nodeColor);
            p.stroke(0);
            p.ellipse(x, y, 20, 20);
        }
    }

    // Draw the output layer with 1 unit
    let outputLayerX = networkAreaStartX + xSpacing * (layerVisuals.length + 1);
    let outputY = ySpacing; // Only one node for output layer
    p.fill(255); // Assuming output nodes are white
    p.stroke(0);
    p.ellipse(outputLayerX, outputY, 20, 20);
}

  
  // Function to update data visualization
  async function updateDataVisualization() {
    //p.background(255);
    if (isTraining) {
        console.log(`Updating decision boundary`)
        await drawDecisionBoundary();  // Update the data visualization area with decision boundary
    } else {
        p.background(255);
    }
    console.log(`Updating network`)
    await drawNetwork();  // This will draw on the right side
    console.log(`Updating data visualization`)
    await visualizeData();  // This will draw on the left side
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
            offscreenGraphics.rect(resolution/8 + i, resolution/8 + j, gridResolution, gridResolution);
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
    let numTilesX = 10;  // Number of tiles in the x direction
    let numTilesY = 10;  // Number of tiles in the y direction
    let tileSizeX = dataAreaWidth / numTilesX;  // Size of each tile in the x direction
    let tileSizeY = canvasHeight / numTilesY;  // Size of each tile in the y direction

    for (let i = 0; i < numTilesX; i++) {
        for (let j = 0; j < numTilesY; j++) {
            let x = i * tileSizeX;
            let y = j * tileSizeY;
            let classLabel = (i + j) % 2;  // Calculate label based on the tile's position
            // Normalize the coordinates to be within [0, 1]
            data.push({
                x: (x + tileSizeX / 2) / dataAreaWidth,
                y: (y + tileSizeY / 2) / canvasHeight,
                label: classLabel
            });
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
    p.clear()
    selectDataset()
    updateDataVisualization();
  });

  p.select('#trainModelButton').mouseClicked(() => {
    trainModel();
  });
  // Ensure to call the setup function to initialize everything
  p.setup();

});
