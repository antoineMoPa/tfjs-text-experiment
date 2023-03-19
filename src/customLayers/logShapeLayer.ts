import * as tf from '@tensorflow/tfjs-node';

/**
 * A logging layer that just logs the input shape at that point.
 * It passes the input through
 */
export class LogShapeLayer extends tf.layers.Layer {
    constructor() {
        super();
    }

    call(input) {
        console.log(input.map(i => i.shape));
        return input;
    }

    static get className() {
        return 'LogShapeLayer';
    }
}

tf.serialization.registerClass(LogShapeLayer);
