const nn = require('./lib/nn2.js');

var test = nn.create(2, 4, 4, 1, 0.1);
// var test = nn.create(2, 8, 1, 0.1);

const trainData = [
    {
        'inputs': [0, 0],
        'targets': [0]
    },
    {
        'inputs': [0, 1],
        'targets': [1]
    },
    {
        'inputs': [1, 0],
        'targets': [1]
    },
    {
        'inputs': [1, 1],
        'targets': [0]
    }
];

for (let i = 0; i < 100000; i++) {
    let rng = Math.floor(Math.random() * 4);
    let data = trainData[rng];
    test.train(data.inputs, data.targets);
}

console.log('此处应为[0]: ' + test.feedForward([1, 1]));
console.log('此处应为[0]: ' + test.feedForward([0, 0]));
console.log('此处应为[1]: ' + test.feedForward([1, 0]));
console.log('此处应为[1]: ' + test.feedForward([0, 1]));