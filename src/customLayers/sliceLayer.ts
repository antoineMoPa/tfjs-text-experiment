import * as tf from '@tensorflow/tfjs-node';

export class SliceLayer extends tf.layers.Layer {
    sliceStart: number[];
    sliceSize: number[];

    constructor(config: { sliceStart: number[]; sliceSize: number[] } & any) {
        super(config);
        this.sliceStart = config.sliceStart
        this.sliceSize = config.sliceSize;
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.sliceSize[1], this.sliceSize[2]]
    }

    call(input) {
        return tf.tidy(() => {
            return tf.slice(
                input[0],
                this.sliceStart,
                this.sliceSize
            );
        });
    }

    getConfig() {
        const config = super.getConfig();

        config.sliceStart = this.sliceStart;
        config.sliceSize = this.sliceSize;

        return config;
    }

    static get className() {
        return 'SliceLayer';
    }
}

tf.serialization.registerClass(SliceLayer);
