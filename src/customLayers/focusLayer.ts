import * as tf from '@tensorflow/tfjs-node';

/**
 * FocusLayer
 * The goal is to zero-out non-focused timesteps.
 */
export class FocusLayer extends tf.layers.Layer {
    min = 0;
    max = 0;
    maxTimeStep = 0;

    constructor(config: { min: number, max?: number, maxTimeStep: number }) {
        super(config as any);
        this.trainable = false;
        this.min = config.min;
        this.max = config.max;
        this.maxTimeStep = config.maxTimeStep;
    }

    call(input) {
        return tf.tidy(() => {
            return tf.customGrad((input: tf.Tensor, save: any) => {
                const shape = input.shape;
                const timestep = input.slice([0,0,0], [shape[0], 1, shape[2]]);
                const isFocus = tf.logicalAnd(
                    this.max && tf.lessEqual(
                        timestep,
                        this.maxTimeStep - 1 - this.max
                    ) || 1,
                    this.min && tf.greaterEqual(
                        timestep,
                        this.maxTimeStep - 1 - this.min
                    ) || 1
                );
                const value = tf.mul(input, isFocus);
                save([isFocus]);

                return {
                    value: value,
                    gradFunc: () => tf.ones(shape),
                };
            })(input[0]);
        });
    }

    getConfig() {
        const config = super.getConfig();
        config.min = this.min;
        config.max = this.max;
        config.maxTimeStep = this.maxTimeStep;
        return config;
    }

    static get className() {
        return 'FocusLayer';
    }
}

tf.serialization.registerClass(FocusLayer);
