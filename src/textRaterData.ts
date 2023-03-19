import {
    TEXT_RATER_INPUT_LENGTH,
    TEXT_RATER_OUTPUT
} from './textRater';
import * as fs from 'fs';
import {
    tokenize,
    CORPUS_PATH
} from './model';

const getEmptyTextRaterData = () => {
    return {
        [TEXT_RATER_OUTPUT.NEVER]: [],
        [TEXT_RATER_OUTPUT.REPETITIVE]: [],
        [TEXT_RATER_OUTPUT.GOOD]: []
    };
};

const buildTextRaterData = () => {
    const textRaterData = getEmptyTextRaterData();
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

            textRaterData[TEXT_RATER_OUTPUT.GOOD].push(tokens.slice(0, inputLength).join(''));

            // Create some random repetitive example
            for (let i = 0; i < 2; i++) {
                const randOffset:number = Math.floor(Math.random() * (inputLength - 2));
                const a = tokens[0 + randOffset].trim();
                const b = tokens[1 + randOffset].trim();
                const c = tokens[2 + randOffset].trim();

                const repetitiveTokens = [];
                const threeWordRepetition = Math.random() < 0.5;

                while (repetitiveTokens.length < inputLength) {
                    repetitiveTokens.push(a, b);
                    threeWordRepetition &&  repetitiveTokens.push(c);
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

const splitDataSet = (allTextRaterData) => {
    const textRaterData = getEmptyTextRaterData();
    const validationData = getEmptyTextRaterData();

    Object.keys(allTextRaterData).forEach(category => {
        allTextRaterData[category].forEach(line => {
            if (Math.random() < 0.2) {
                validationData[category].push(line)
            } else {
                textRaterData[category].push(line)
            }
        });
    });

    return {
        textRaterData,
        validationData
    };
};

export const { textRaterData, validationData } = splitDataSet(buildTextRaterData());

export const flatTextRaterData: string[] = Object.keys(textRaterData).map(key => {
    return textRaterData[key].flat();
}).flat();
