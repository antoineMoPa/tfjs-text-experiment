import {
    TEXT_RATER_INPUT_LENGTH,
    TEXT_RATER_OUTPUT
} from './textRater';
import * as fs from 'fs';
import {
    tokenize,
    CORPUS_PATH
} from './tinygpt';

const buildTextRaterData = () => {
    const textRaterData = {
        [TEXT_RATER_OUTPUT.NEVER]: [],
        [TEXT_RATER_OUTPUT.REPETITIVE]: [],
        [TEXT_RATER_OUTPUT.GOOD]: []
    };
    const sampleFiles = [`${CORPUS_PATH}/wiki-horse.txt`, `${CORPUS_PATH}/wiki-computer.txt`];

    for (const inputFile of sampleFiles) {
        const text = fs.readFileSync(inputFile).toString();
        const lines = text.split('\n');
        const inputLength: number = TEXT_RATER_INPUT_LENGTH;


        lines.forEach(line => {
            const tokens = tokenize(line);

            // Only train on substantial lines
            if (tokens.length < inputLength) {
                return;
            }

            for (let i = 0; i < 2; i++) {
                // Repeat good example
                textRaterData[TEXT_RATER_OUTPUT.GOOD].push(tokens.slice(0, inputLength).join(''));

                // Create random repetitive example
                const randOffset:number = Math.floor(Math.random() * (inputLength - 1));
                const a = tokens[0 + randOffset].trim();
                const b = tokens[1 + randOffset].trim();

                const repetitiveTokens = [];

                while (repetitiveTokens.length < inputLength) {
                    repetitiveTokens.push(a, b);
                }
                textRaterData[TEXT_RATER_OUTPUT.REPETITIVE].push(repetitiveTokens.join(' '));
            }
        });
    }

    console.log('build text rater training data. Sample count:', {
        'GOOD': textRaterData[TEXT_RATER_OUTPUT.GOOD].length,
        'REPETITIVE': textRaterData[TEXT_RATER_OUTPUT.REPETITIVE].length
    });

    return textRaterData;
};

export const textRaterData = buildTextRaterData();

export const flatTextRaterData: string[] = Object.keys(textRaterData).map(key => {
    return textRaterData[key].flat();
}).flat();
