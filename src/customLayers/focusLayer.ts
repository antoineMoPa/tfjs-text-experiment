import * as tf from '@tensorflow/tfjs-node';

/**
 * FocusLayer
 * The goal is to zero-out non-focused timesteps.
 */
export class FocusLayer extends tf.layers.Layer {
    min = 0;
    max = 0;

    constructor(config: { min: number, max?: number }) {
        super(config as any);
        this.trainable = false;
        this.min = config.min;
        this.max = config.max;
    }

    call(input) {
        return tf.tidy(() => {
            return tf.customGrad((input: tf.Tensor) => {
                const shape = input.shape;
                const timestep = input.slice([0,0,0], [shape[0], 1, shape[2]]);
                const isFocus = tf.logicalAnd(
                    this.max && tf.lessEqual(timestep, this.max) || 1,
                    this.min && tf.greaterEqual(timestep, this.min) || 1
                );
                const value = tf.mul(input, isFocus);

                return {
                    value: value,
                    gradFunc: (x) => tf.zeros(x.shape),
                };
            })(input[0]);
        });
    }

    getConfig() {
        const config = super.getConfig();
        config.min = this.min;
        config.max = this.max;
        return config;
    }

    static get className() {
        return 'FocusLayer';
    }
}

tf.serialization.registerClass(FocusLayer);
