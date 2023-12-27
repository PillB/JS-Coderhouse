// neuralNetworkPlayground.js

// TensorFlow.js model and visualization parameters
let model;
let layerVisuals = [];
let currentData = [];
let isTraining = false;
let currentEpoch = 0;
let totalEpochs = 100;
let lossHistory = [];
let resolution = 20;

// p5.js structure
new p5(p => {
  // p5.js setup function
  p.setup = () => {
    p.createCanvas(300, 300).parent('dataViz');
    p.createCanvas(300, 300).parent('nnViz');
    p.noLoop(); // Prevent draw loop unless we are training

    // Initialize TensorFlow.js model
    initializeModel(p.select('#activation').value());

    // Prepare initial visualization
    selectDataset();
  };

  // Function to update the training status and loss graph
  function updateTrainingStatus(epoch, loss) {
    p.select('#epochInfo').html(`Epoch: ${epoch}/${totalEpochs}`);
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
      x: d.x / p.width,
      y: d.y / p.height,
      label: d.label
    }));
  }

  // Function to visualize data
  function visualizeData() {
    p.background(255);
    currentData.forEach(point => {
      p.stroke(0);
      if (point.label === 0) {
        p.fill('blue');
        p.ellipse(point.x * p.width, point.y * p.height, resolution, resolution);
      } else {
        p.fill('red');
        drawPlusSign(point.x * p.width, point.y * p.height, resolution);
      }
    });
  }

  // Function to train the model
  async function trainModel() {
    isTraining = true;
    p.loop(); // Start the draw loop

    let inputs = tf.tensor2d(currentData.map(p => [p.x, p.y]));
    let labels = tf.tensor2d(currentData.map(p => [p.label]));

    await model.fit(inputs, labels, {
      epochs: totalEpochs,
      callbacks: {
        onEpochEnd: (epoch, log) => {
          currentEpoch = epoch;
          lossHistory.push(log.loss);
          updateTrainingStatus(epoch, log.loss);
          p.redraw(); // Redraw only when needed during training
        }
      }
    });

    isTraining = false;
    p.noLoop(); // Stop the draw loop
    p.select('#output').html('Training complete!');
    p.select('#dataShape').attribute('disabled', null);
  }

  // Function to draw plus signs for the visualization
  function drawPlusSign(x, y, size) {
    const halfSize = size / 2;
    p.strokeWeight(2);
    p.stroke(0);
    p.line(x - halfSize, y, x + halfSize, y);
    p.line(x, y - halfSize, x, y + halfSize);
  }

  // Function to draw the neural network visualization
  function drawNetwork() {
    p.networkCanvas.clear();
    let xSpacing = p.networkCanvas.p.width / (layerVisuals.length + 1);
    let ySpacing = p.networkCanvas.p.height / (Math.max(...layerVisuals.map(l => l.nodes)) + 1);

    // Draw connections between nodes
    for (let i = 1; i < layerVisuals.length; i++) {
      let prevLayer = layerVisuals[i - 1];
      let currentLayer = layerVisuals[i];
      let prevX = xSpacing * i;
      let currentX = xSpacing * (i + 1);
      for (let j = 0; j < prevLayer.nodes; j++) {
        for (let k = 0; k < currentLayer.nodes; k++) {
          let prevY = ySpacing * (j + 1);
          let currentY = ySpacing * (k + 1);
          let weight = model.layers[i].getWeights()[0].arraySync()[j][k];
          let strokeColor = weight > 0 ? p.color(0, 255, 0) : p.color(255, 0, 0);
          p.networkCanvas.stroke(strokeColor);
          p.networkCanvas.line(prevX, prevY, currentX, currentY);
        }
      }
    }

    // Draw nodes on top of connections
    for (let i = 0; i < layerVisuals.length; i++) {
      let layer = layerVisuals[i];
      let x = xSpacing * (i + 1);
      for (let j = 0; j < layer.nodes; j++) {
        let y = ySpacing * (j + 1);
        let activation = layer.activations[j];
        let nodeColor = p.lerpColor(p.color(255, 255, 255), p.color(0, 0, 255), activation);
        p.networkCanvas.fill(nodeColor);
        p.networkCanvas.stroke(0);
        p.networkCanvas.ellipse(x, y, 20, 20);
      }
    }
  }


  // Function to update data visualization
  function updateDataVisualization() {
    visualizeData();
    if (isTraining) {
      drawDecisionBoundary();
    }
  }

  // Function to draw the decision boundary
  async function drawDecisionBoundary() {
    let gridResolution = 10; // Change to a lower number for a finer grid
    let cols = p.width / gridResolution;
    let rows = p.height / gridResolution;

    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        let x = i * gridResolution;
        let y = j * gridResolution;
        let inputTensor = tf.tensor2d([[x / p.width, y / p.height]]);
        let prediction = model.predict(inputTensor);
        let classIdx = (await prediction.data())[0] > 0.5 ? 1 : 0;
        prediction.dispose();

        if (classIdx === 0) {
          p.dataCanvas.fill('blue');
        } else {
          p.dataCanvas.fill('red');
        }
        p.dataCanvas.noStroke();
        p.dataCanvas.rect(x, y, gridResolution, gridResolution);
      }
    }
  }


  // Event listeners for DOM elements
  //p.select('#trainButton').mousePressed(trainModel);
  //p.select('#dataShape').changed(selectDataset);

  // ... (include all data generation functions here) ...
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
        let x = p.randomGaussian(mean1[0], variance1[0]);
        let y = p.randomGaussian(mean1[1], variance1[1]);
        data.push({x: x, y: y, label: 0});
    }

    // Generate points for the second cluster
    for (let i = 0; i < pointsPerCluster; i++) {
        let x = p.randomGaussian(mean2[0], variance2[0]);
        let y = p.randomGaussian(mean2[1], variance2[1]);
        data.push({x: x, y: y, label: 1});
    }

    return data;
}

function generateCheckerboardData() {
    let data = [];
    let tileSize = p.width / 10; // Number of tiles per row/column

    for (let x = 0; x < p.width; x += tileSize) {
        for (let y = 0; y < p.height; y += tileSize) {
            let classLabel = (p.floor(x / tileSize) + p.floor(y / tileSize)) % 2;
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
        let angle = p.random(Math.PI); // half circle
        let x = p.width * 0.5 + radius * p.cos(angle);
        let y = p.height * 0.5 + radius * p.sin(angle);
        data.push({x: x, y: y, label: 0});
    }

    // Generate points for the second moon
    for (let i = 0; i < pointsPerMoon; i++) {
        let angle = p.random(Math.PI); // half circle
        let x = p.width * 0.5 + radius * p.cos(angle + Math.PI);
        let y = p.height * 0.5 + radius * p.sin(angle + Math.PI);
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
        let angle = p.map(i, 0, turns * pointsPerTurn, 0, turns * p.TWO_PI);
        let radius = p.map(i, 0, turns * pointsPerTurn, 0, p.width / 3);
        let x = p.width / 2 + radius * p.cos(angle);
        let y = p.height / 2 + radius * p.sin(angle);
        data.push({x: x, y: y, label: i % 2});
    }

    // Generate points for the second spiral
    for (let i = 0; i < turns * pointsPerTurn; i++) {
        let angle = p.map(i, 0, turns * pointsPerTurn, 0, turns * p.TWO_PI) + Math.PI;
        let radius = p.map(i, 0, turns * pointsPerTurn, 0, p.width / 3);
        let x = p.width / 2 + radius * p.cos(angle);
        let y = p.height / 2 + radius * p.sin(angle);
        data.push({x: x, y: y, label: (i + 1) % 2});
    }

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
