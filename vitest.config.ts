import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
      outputDiffLines: Infinity,
      outputTruncateLength: Infinity
  },
})
