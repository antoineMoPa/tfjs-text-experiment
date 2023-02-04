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

    const text = readFileSync("data/wiki-horse.txt").toString();

    // Embed an array of sentences.
    const sentences = text.split('.').slice(0,100);

    console.log('Building embeddings');

    model.embed(['test']).then(embeddings => {
        console.log(embeddings.shape);
        // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
        // So in this example `embeddings` has the shape [2, 512].
        embeddings.print(true /* verbose */);
    });

    const wordDecodeModel: Sequential = tf.sequential();
    const HIDDEN_SIZE = 2;

    wordDecodeModel.add(
        tf.layers.dense({
            inputShape: EMBED_SHAPE,
            units: HIDDEN_SIZE,
            activation: "tanh",
        })
    );

    wordDecodeModel.add(
        tf.layers.dense({
            units: HIDDEN_SIZE,
            activation: "tanh",
        })
    );

    wordDecodeModel.summary();

    const ALPHA = 0.001
    console.log('Compiling word decoding model.');
    wordDecodeModel.compile({
        optimizer: tf.train.sgd(ALPHA),
        loss: "meanSquaredError",
    })
    console.log('Done!');

    console.log('Training word decoding model.');



    // await model.fit(input, expectedOutput, {
    //     epochs: 200,
    //     callbacks: {
    //         onEpochEnd: async (epoch, logs) => {
    //             if (epoch % 10 === 0) {
    //                 console.log(`Epoch ${epoch}: error: ${logs.loss}`)
    //             }
    //         },
    //     },
    // });
    console.log('Done!');

};
