import React from 'react';
import ReactDOM from 'react-dom/client';

// Import the shared CSS
import '@/index.css';

// Import the three possible application entry points
import App from './web/App'; // Your existing App.tsx, now the "WebApp"
//import LauncherShell from './desktop/LauncherShell'; // The new "DesktopApp"
import OverlayWindow from './desktop/OverlayWindow'; // The overlay window
import ScreenSelectorWindow from './desktop/ScreenSelectorWindow'; // Screen/window selector

// Import platform detection utilities
import { isDesktop } from './utils/platform';

// Decide which component to render at the root level
function getRootComponent() {
  // Desktop only: overlay route
  if (isDesktop() && window.location.pathname === '/overlay') {
    return OverlayWindow;
  }

  // Desktop only: screen selector route
  if (isDesktop() && window.location.pathname === '/screen-selector') {
    return ScreenSelectorWindow;
  }

  // Desktop Tauri: use LauncherShell with desktop-specific features
  if (isDesktop()) {
    return App;
  }

  // Mobile Tauri or Web: use App
  // This ensures mobile Tauri users get the same UI as web users
  // (without desktop-specific features like keyboard shortcuts, overlay controls, etc.)
  return App;
}

const RootComponent = getRootComponent();

// Render the chosen component
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RootComponent />
  </React.StrictMode>
);
