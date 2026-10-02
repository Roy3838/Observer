import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';
import { resolve } from 'path';

const devHost = process.env.TAURI_DEV_HOST;

// onnxruntime-web references its ~26 MB wasm via `new URL(..., import.meta.url)`, so Vite copies it
// into dist/assets, and Cloudflare Pages rejects files over 25 MiB. transformers.js never loads that
// copy: unless wasmPaths is set it fetches the matching ort wasm/mjs from jsDelivr
// (see transformers backends/onnx.js), so we drop the bundled files from the output.
const dropOrtWasm = () => ({
  name: 'drop-ort-wasm',
  generateBundle(_options: unknown, bundle: Record<string, { type: string }>) {
    for (const fileName of Object.keys(bundle)) {
      if (bundle[fileName].type === 'asset' && /(^|\/)ort-wasm[^/]*\.wasm$/.test(fileName)) {
        delete bundle[fileName];
      }
    }
  },
});

export default defineConfig({
  plugins: [
    react(),
    dropOrtWasm(),
    //visualizer({
    //  open: true, // Open the visualization after build
    //  gzipSize: true,
    //  brotliSize: true
    //})
  ],
  worker: {
    format: 'es', // Enable ES module format for workers to support code-splitting
  },
  server: {
    host: devHost || '0.0.0.0',
    port: 3001, // Different from desktop and website
    strictPort: true,
    hmr: devHost
      ? { protocol: 'ws', host: devHost, port: 3001 }
      : undefined,
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@components': resolve(__dirname, './src/components'),
      '@utils': resolve(__dirname, './src/utils'),
      '@web': resolve(__dirname, './src/web'),
      '@desktop': resolve(__dirname, './src/desktop'),
      '@hooks': resolve(__dirname, './src/hooks'),
      '@contexts': resolve(__dirname, './src/contexts')
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 800, // Increase warning threshold
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            // Heavy isolated dependencies get their own chunks
            if (id.includes('tesseract')) {
              return 'vendor-tesseract';
            }
            if (id.includes('jupyterlab')) {
              return 'vendor-jupyter';
            }
            if (id.includes('onnxruntime')) {
              return 'vendor-onnx';
            }
            // Let Vite handle the rest automatically
          }
        }
      }
    }
  },
});
