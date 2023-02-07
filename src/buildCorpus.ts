import { readdirSync, readFileSync, createWriteStream } from 'fs';

const DATA_DIR = 'data';

const fileNames = readdirSync(DATA_DIR)
    .filter(fileName => /wiki-.*.txt/.test(fileName));


const writableStream = createWriteStream('data/data-corpus.txt');

fileNames.forEach(fileName => {
    writableStream.write(readFileSync(DATA_DIR + '/' + fileName));
});

writableStream.close();
