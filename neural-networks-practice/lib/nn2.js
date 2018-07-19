const matrix = require('./matrix.js');

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) {
    return x * (1 - x);
}

class neuralNetwork {

    constructor(inputNodes, hiddenANodes, hiddenBNodes, outputNodes, learningRate) {

        this.inputNodes = inputNodes;
        this.hiddenANodes = hiddenANodes;
        this.hiddenBNodes = hiddenBNodes;
        this.outputNodes = outputNodes;

        // 权重
        this.weights = {
            'ia': matrix.create(this.hiddenANodes, this.inputNodes),
            'ab': matrix.create(this.hiddenBNodes, this.hiddenANodes),
            'bo': matrix.create(this.outputNodes, this.hiddenBNodes)
        };
        this.weights.ia.randomize();
        this.weights.ab.randomize();
        this.weights.bo.randomize();

        // 偏差值
        this.bias = {
            'a': matrix.create(this.hiddenANodes, 1),
            'b': matrix.create(this.hiddenBNodes, 1),
            'o': matrix.create(this.outputNodes, 1)
        };
        this.bias.a.randomize();
        this.bias.b.randomize();
        this.bias.o.randomize();

        // Learning Rate
        this.learningRate = learningRate;

    }

    feedForward(arr) {

        // 建立 Inputs Nodes
        let inputs = matrix.fromArray(arr);

        // 建立 Hidden A Nodes
        let hiddenA = matrix.multiply(this.weights.ia, inputs);
        hiddenA.add(this.bias.a);
        hiddenA.map(sigmoid);

        // 建立 Hidden B Nodes
        let hiddenB = matrix.multiply(this.weights.ab, hiddenA);
        hiddenB.add(this.bias.b);
        hiddenB.map(sigmoid);

        // 建立 Outputs Nodes
        let outputs = matrix.multiply(this.weights.bo, hiddenB);
        outputs.add(this.bias.o);
        outputs.map(sigmoid);

        // 返回结果
        return outputs.toArray();
    }

    train(inputsArr, targetsArr) {

        // 从feedForward获取outputs
        // 建立 Inputs Nodes
        let inputs = matrix.fromArray(inputsArr);

        // 建立 Hidden A Nodes
        let hiddenA = matrix.multiply(this.weights.ia, inputs);
        hiddenA.add(this.bias.a);
        hiddenA.map(sigmoid);

        // 建立 Hidden B Nodes
        let hiddenB = matrix.multiply(this.weights.ab, hiddenA);
        hiddenB.add(this.bias.b);
        hiddenB.map(sigmoid);

        // 建立 Outputs Nodes
        let outputs = matrix.multiply(this.weights.bo, hiddenB);
        outputs.add(this.bias.o);
        outputs.map(sigmoid);

        // 获取targets矩阵
        let targets = matrix.fromArray(targetsArr);

        // 计算Error: errors = targets - outputs
        let errors = matrix.subtract(targets, outputs);

        // 计算output梯度
        let outputsGradients = matrix.map(outputs, dsigmoid);
        outputsGradients.multiply(errors);
        outputsGradients.multiply(this.learningRate);

        // 计算delta
        let hiddenBT = matrix.transpose(hiddenB);
        let weightsBODeltas = matrix.multiply(outputsGradients, hiddenBT);

        // 修正Outputs Weights
        this.weights.bo.add(weightsBODeltas);
        // 修正Outputs Bias
        this.bias.o.add(outputsGradients);

        // 计算Hidden B Nodes Error
        let weightsBOT = matrix.transpose(this.weights.bo);
        let hiddenBErrors = matrix.multiply(weightsBOT, errors);

        // 计算Hidden B Nodes 梯度
        let hiddenBGradients = matrix.map(hiddenB, dsigmoid);
        hiddenBGradients.multiply(hiddenBErrors);
        hiddenBGradients.multiply(this.learningRate);

        // 计算delta
        let hiddenAT = matrix.transpose(hiddenA);
        let weightsABDeltas = matrix.multiply(hiddenBGradients, hiddenAT);

        // 修正Hidden B Nodes Weights
        this.weights.ab.add(weightsABDeltas);
        // 修正Hidden B Nodes Bias
        this.bias.b.add(hiddenBGradients);

        // 计算Hidden A Nodes Error
        let weightsABT = matrix.transpose(this.weights.ab);
        let hiddenAErrors = matrix.multiply(weightsABT, hiddenBErrors);

        // 计算Hidden A Nodes 梯度
        let hiddenAGradients = matrix.map(hiddenA, dsigmoid);
        hiddenAGradients.multiply(hiddenAErrors);
        hiddenAGradients.multiply(this.learningRate);

        // 计算 Delta
        let inputsT = matrix.transpose(inputs);
        let weightsIADeltas = matrix.multiply(hiddenAGradients, inputsT);

        // 修正Inputs Weights
        this.weights.ia.add(weightsIADeltas);
        // 修正Inputs Bias
        this.bias.a.add(hiddenAGradients);
    }
}

module.exports = {
    'create': function (inputNodes, hiddenANodes, hiddenBNodes, outputNodes, learningRate) {
        let nn = new neuralNetwork(inputNodes, hiddenANodes, hiddenBNodes, outputNodes, learningRate);
        return nn;
    }
}