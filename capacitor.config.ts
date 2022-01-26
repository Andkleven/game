import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.example.app',
  appName: 'game',
  webDir: 'out',
  bundledWebRuntime: false,
  server: {
    url: "http://192.168.11.129:3000",
    cleartext: true
  }
};

export default config;
