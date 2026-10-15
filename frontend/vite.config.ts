import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 生产构建输出到 dist，由后端 app/api/main.py 用 StaticFiles 挂在根路径；
// 开发模式（npm run dev）通过 server.proxy 把接口代理到本地 8000 后端。
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          antd: ['antd', '@ant-design/icons'],
          markdown: ['react-markdown', 'remark-gfm'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/sse': 'http://127.0.0.1:8000',
      '/charts': 'http://127.0.0.1:8000',
      '/reports': 'http://127.0.0.1:8000',
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
