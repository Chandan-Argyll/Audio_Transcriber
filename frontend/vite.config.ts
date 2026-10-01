import { defineConfig } from 'vitest/config';
import { loadEnv } from 'vite';
import { fileURLToPath } from 'node:url';
import type { ServerResponse } from 'node:http';
import react from '@vitejs/plugin-react';
export default defineConfig(({mode})=>{
  const env=loadEnv(mode,fileURLToPath(new URL('../',import.meta.url)),'API_');
  const target=env.API_PROXY_TARGET||`http://127.0.0.1:${env.API_PORT||'8000'}`;
  return {
    plugins:[react()],
    server:{proxy:{'/api':{
      target,changeOrigin:true,
      configure(proxy){
        proxy.on('error',(_error,_request,res)=>{
          console.error(`[API proxy] Backend unavailable at ${new URL(target).origin}. Start the API or check API_PROXY_TARGET.`);
          const response=res as ServerResponse;
          if(typeof response.writeHead==='function'&&!response.headersSent){
            response.writeHead(503,{'Content-Type':'application/json'});
            response.end(JSON.stringify({detail:'Backend connection failed. Start the API and check API_PROXY_TARGET / API_PORT.'}));
          }
        });
      }
    }}},
    test:{environment:'jsdom',setupFiles:['./src/test-setup.ts'],globals:true}
  };
});
