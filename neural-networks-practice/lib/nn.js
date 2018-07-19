const matrix = require('./matrix.js');

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) {
    return x * (1 - x);
}

class neuralNetwork {

    constructor(inputNodes, hiddenNodes, outputNodes, learningRate) {

        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        // 权重
        this.weights = {
            'ih': matrix.create(this.hiddenNodes, this.inputNodes),
            'ho': matrix.create(this.outputNodes, this.hiddenNodes)
        };
        this.weights.ih.randomize();
        this.weights.ho.randomize();

        // 偏差值
        this.bias = {
            'h': matrix.create(this.hiddenNodes, 1),
            'o': matrix.create(this.outputNodes, 1)
        };
        this.bias.h.randomize();
        this.bias.o.randomize();

        // Learning Rate
        this.learningRate = learningRate;

    }

    feedForward(arr) {

        // 建立 Inputs Nodes
        let inputs = matrix.fromArray(arr);

        // 建立 Hidden Nodes
        let hidden = matrix.multiply(this.weights.ih, inputs);
        hidden.add(this.bias.h);
        hidden.map(sigmoid);

        // 建立 Outputs Nodes
        let outputs = matrix.multiply(this.weights.ho, hidden);
        outputs.add(this.bias.o);
        outputs.map(sigmoid);

        // 返回结果
        return outputs.toArray();
    }

    train(inputsArr, targetsArr) {

        // 从feedForward获取outputs
        // 建立 Inputs Nodes
        let inputs = matrix.fromArray(inputsArr);

        // 建立 Hidden Nodes
        let hidden = matrix.multiply(this.weights.ih, inputs);
        hidden.add(this.bias.h);
        hidden.map(sigmoid);

        // 建立 Outputs Nodes
        let outputs = matrix.multiply(this.weights.ho, hidden);
        outputs.add(this.bias.o);
        outputs.map(sigmoid);

        // 获取targets矩阵
        let targets = matrix.fromArray(targetsArr);

        // 计算Error: errors = targets - outputs
        let errors = matrix.subtract(targets, outputs);

        // 计算梯度
        let gradients = matrix.map(outputs, dsigmoid);
        gradients.multiply(errors);
        gradients.multiply(this.learningRate);

        // 计算delta
        let hiddenT = matrix.transpose(hidden);
        let weightsHODeltas = matrix.multiply(gradients, hiddenT);

        // 修正Outputs Weights
        this.weights.ho.add(weightsHODeltas);
        // 修正Outputs Bias
        this.bias.o.add(gradients);

        // 计算Hidden Nodes Error
        let weightsHOT = matrix.transpose(this.weights.ho);
        let hiddenErrors = matrix.multiply(weightsHOT, errors);

        // 计算Hidden Nodes 梯度
        let hiddenGradient = matrix.map(hidden, dsigmoid);
        hiddenGradient.multiply(hiddenErrors);
        hiddenGradient.multiply(this.learningRate);

        // 计算inputs -> hidden Delta
        let inputsT = matrix.transpose(inputs);
        let weightsIHDeltas = matrix.multiply(hiddenGradient, inputsT);

        this.weights.ih.add(weightsIHDeltas);
        this.bias.h.add(hiddenGradient);
    }
}

module.exports = {
    'create': function (inputNodes, hiddenNodes, outputNodes, learningRate) {
        let nn = new neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
        return nn;
    }
}