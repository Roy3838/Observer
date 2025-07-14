import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';
import { resolve } from 'path';

export default defineConfig({
  base: '/',
  plugins: [
    react(),
    //visualizer({
    //  open: true, // Open the visualization after build
    //  gzipSize: true,
    //  brotliSize: true
    //})
  ],
  server: {
    host: '0.0.0.0', 
    port: 3001, // Different from desktop and website
  },
  preview: {
    host: '0.0.0.0',
    port: 8080,
    allowedHosts: 'all', // Allow any host to access the preview server
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@components': resolve(__dirname, './src/components'),
      '@utils': resolve(__dirname, './src/utils'),
      '@web': resolve(__dirname, './src/web'),
      '@desktop': resolve(__dirname, './src/desktop')
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 800, // Increase warning threshold
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Vendor chunks
          if (id.includes('node_modules')) {
            if (id.includes('react') || id.includes('scheduler')) {
              return 'vendor-react';
            }
            if (id.includes('jupyterlab')) {
              return 'vendor-jupyter';
            }
            return 'vendor'; // Other dependencies
          }
          
          // App chunks
          if (id.includes('/src/components/')) {
            return 'app-components';
          }
          if (id.includes('/src/utils/')) {
            return 'app-utils';
          }
        }
      }
    }
  },
});
