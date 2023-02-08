// Source map support. Does this work?
// eslint-disable-next-line @typescript-eslint/no-var-requires
require('source-map-support').install();

// Test bench deps
import { execSync } from 'child_process';
import { watch } from 'chokidar';

// Slow loading deps
import { load as loadUniversalSentenceEncoder } from '@tensorflow-models/universal-sentence-encoder';

import * as tf from '@tensorflow/tfjs-node-gpu';

tf.ready().then(() => {
    loadUniversalSentenceEncoder().then(model => {
        global.universalSentenceEncoderModel = model;

        const buildAndRun = () => {
            console.log('Building and running');

            execSync('yarn run build');

            const ENTRYPOINT = './tinygpt.js';
            delete require.cache[require.resolve(ENTRYPOINT)];
            try {
                // eslint-disable-next-line @typescript-eslint/no-var-requires
                require(ENTRYPOINT).main();
            } catch (e) {
                console.error(e);
            }
        };

        watch('src/', {}).on('change', buildAndRun).on('ready', function() {
            console.log('ready');
            buildAndRun();
        });
    });
});
