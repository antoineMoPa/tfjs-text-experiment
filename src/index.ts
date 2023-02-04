import { readFileSync } from 'fs';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import type { Sequential } from '@tensorflow/tfjs-layers/dist/models';
import type { UniversalSentenceEncoder } from  '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { performance } from 'perf_hooks';

const EMBED_SHAPE = [1, 512];


type Corpus = {
    wordTensors: Tensor2D;
    words: string[];
};

async function buildCorpus() {
    console.log('Building corpus');
    const t1 = performance.now();
    const model = global.universalSentenceEncoderModel;
    const text = readFileSync("data/wiki-horse.txt").toString().toLowerCase();
    const words = _.uniq(text.split(/\s/)).slice(0,1000);
    const wordTensors = await model.embed(words);

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

export const main = async () => {
    const model: UniversalSentenceEncoder = global.universalSentenceEncoderModel;
    const corpus = await buildCorpus();
    console.log(corpus.words);

    console.log('Should be wikipedia: ', findClosestWord(await model.embed(['Wikipedia']), corpus));

    console.log('Building embeddings');

    model.embed(['test']).then(embeddings => {
        console.log(embeddings.shape);
        // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
        // So in this example `embeddings` has the shape [2, 512].
        embeddings.print(true /* verbose */);
    });

    const wordPredictModel: Sequential = tf.sequential();
    const HIDDEN_SIZE = 500;

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

    const ALPHA = 0.001
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
        const words = text.split(/[ \n]/).slice(0,1000).filter(v => v !== '');
        const bigrams = [];
        const expectedOutputs = [];

        for (let i = 0; i < words.length - 2; i++) {
            bigrams.push(words[i] + ' ' + words[i + 1]);
            // Predict the 3rd word based on previous words.
            expectedOutputs.push(words[i+2]);
        }

        const wordTensors = await model.embed(bigrams);

        return {
            wordTensors,
            expectedOutputs: await model.embed(expectedOutputs)
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
        epochs: 200,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                if (epoch % 10 === 0) {
                    console.log(`Epoch ${epoch}: error: ${logs.loss}`)
                }
            },
        },
    });

    const prediction = wordPredictModel.predict(await model.embed("horse")) as Tensor2D;
    console.log({ before: "horse", prediction: findClosestWord(prediction, corpus) });
    console.log('Done!');

};
