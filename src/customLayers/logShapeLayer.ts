import * as tf from '@tensorflow/tfjs-node';

/**
 * A logging layer that just logs the input shape at that point.
 * It passes the input through
 */
export class LogShapeLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
    }

    call(input) {
        console.log(input.shape);
        return input;
    }

    getConfig() {
        return super.getConfig();
    }

    static get className() {
        return 'LogShapeLayer';
    }
}

tf.serialization.registerClass(LogShapeLayer);
