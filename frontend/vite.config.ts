/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    host: "127.0.0.1"
  },
  test: {
    // jsdom because the download helpers touch Blob/URL/<a>; the pure modules
    // don't need it but a single environment keeps the config simple.
    environment: "jsdom",
    include: ["src/**/*.test.ts"]
  }
});
