import { readFileSync } from 'fs';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import type { Sequential } from '@tensorflow/tfjs-layers/dist/models';
import type { UniversalSentenceEncoder } from  '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { performance } from 'perf_hooks';
import { LayersModel } from '@tensorflow/tfjs-node';

const EMBED_SHAPE = [1, 512];
const scentenceEncoderModel = global.universalSentenceEncoderModel;

type Corpus = {
    wordTensors: Tensor2D;
    words: string[];
};

async function buildCorpus() {
    console.log('Building corpus');
    const t1 = performance.now();
    const text = readFileSync("data/wiki-horse.txt").toString().toLowerCase();
    const words = _.uniq(text.split(/\s/));
    const wordTensors = await scentenceEncoderModel.embed(words);

    const t2 = performance.now();
    console.log(`Done! (in ${(t2 - t1).toFixed(0)} ms)`);

    return {
        wordTensors,
        words
    };
}

function findClosestWord(tensor: Tensor2D, corpus: Corpus) {
    let closest = -1;
    let closestDist = null;
    for(let i = 0; i < corpus.wordTensors.shape[0]; i++) {
        const dist = tf.norm(tf.squaredDifference(corpus.wordTensors.slice(i, 1), tensor));
        if (closestDist === null || dist < closestDist) {
            closest = i;
            closestDist = dist;
        }
    }

    return corpus.words[closest];
}

async function buildModel() {
    const corpus = await buildCorpus();
    console.log(corpus.words);

    console.log('Should be wikipedia: ', findClosestWord(await scentenceEncoderModel.embed(['Wikipedia']), corpus));

    console.log('Building embeddings');

    scentenceEncoderModel.embed(['test']).then(embeddings => {
        console.log(embeddings.shape);
        // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
        // So in this example `embeddings` has the shape [2, 512].
        embeddings.print(true /* verbose */);
    });

    const wordPredictModel: Sequential = tf.sequential();
    const HIDDEN_SIZE = 1000;

    wordPredictModel.add(
        tf.layers.dense({
            inputShape: [EMBED_SHAPE[1]],
            units: HIDDEN_SIZE,
            activation: "tanh",
        })
    );

    wordPredictModel.add(
        tf.layers.dense({
            units: HIDDEN_SIZE,
            activation: "tanh",
        })
    );

    wordPredictModel.add(
        tf.layers.dense({
            units: EMBED_SHAPE[1],
            activation: "tanh",
        })
    );

    wordPredictModel.summary();

    const ALPHA = 0.1;
    console.log('Compiling word decoding model.');
    wordPredictModel.compile({
        optimizer: tf.train.sgd(ALPHA),
        loss: "meanSquaredError",
    })
    console.log('Done!');

    console.log('Training word decoding model.');

    // First, as an experiment
    // Lets create a training dataset of n-grams and the expected next word
    const buildTrainingData = async () => {
        const text = readFileSync("data/wiki-horse.txt").toString().toLocaleLowerCase();
        // Embed an array of sentences.
        const words = text.split(/[ \n]/).filter(v => v !== '');
        const ngrams = [];
        const expectedOutputs = [];
        const beforeSize = 1;

        for (let i = 0; i < words.length - beforeSize; i++) {
            ngrams.push(words.slice(i, i + beforeSize).join(' '));
            // We want to predict the next word based on previous words.
            expectedOutputs.push(words[i + beforeSize]);
        }

        const wordTensors = await scentenceEncoderModel.embed(ngrams);

        return {
            wordTensors,
            expectedOutputs: await scentenceEncoderModel.embed(expectedOutputs)
        };
    }

    const {
        wordTensors,
        expectedOutputs
    } = await buildTrainingData();

    console.log({
        wordTensors,
        expectedOutputs
    });

    await wordPredictModel.fit(wordTensors, expectedOutputs, {
        epochs: 100,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                if (epoch % 10 === 0) {
                    console.log(`Epoch ${epoch}: error: ${logs.loss}`)
                }
            },
        },
    });

    await wordPredictModel.save('file://wordPredictModel');
    return wordPredictModel;
}

async function getModel() {
    try {
        const wordPredictModel = await tf.loadLayersModel('file://wordPredictModel/model.json');
        return wordPredictModel;
    } catch (e) {
        console.log(e);
        console.log('Model not found/has error. Generating')
    }

    return buildModel();
}

export const main = async () => {
    const corpus = await buildCorpus();
    const wordPredictModel = await getModel() as LayersModel;

    // Test model
    let str = "A 2021 genetic study";
    for (let i = 0; i < 10; i++) {
        const last5 = str.toLowerCase().split(' ').slice(-1);
        const prediction = wordPredictModel.predict(await scentenceEncoderModel.embed(last5.join(' '))) as Tensor2D;
        str += ' ' + findClosestWord(prediction, corpus);
    }

    console.log(str);
};
