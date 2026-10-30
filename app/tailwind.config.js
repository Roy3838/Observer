/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      screens: {
        'header-break': '900px', // Example: a custom breakpoint at 900px
        'wide': '1400px',
      },
      fontFamily: {
        // Official Observer font; applied scoped (e.g. the onboarding splash), not globally.
        golos: ['"Golos Text"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        // Grow-and-shrink "click me" cue used by the onboarding tutorial.
        nudge: {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.5)' },
        },
        // Half the growth — for big targets like the Send button.
        'nudge-soft': {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.25)' },
        },
        // The ask_user_info sheet: backdrop fade, card rise, drawn check mark.
        'sheet-fade': { from: { opacity: '0' }, to: { opacity: '1' } },
        'sheet-in': {
          from: { opacity: '0', transform: 'translateY(12px) scale(0.98)' },
          to: { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        'check-draw': { to: { strokeDashoffset: '0' } },
      },
      animation: {
        nudge: 'nudge 1.1s ease-in-out infinite',
        'nudge-soft': 'nudge-soft 1.1s ease-in-out infinite',
        'sheet-fade': 'sheet-fade 200ms ease-out',
        'sheet-in': 'sheet-in 260ms cubic-bezier(0.16, 1, 0.3, 1)',
        'check-draw': 'check-draw 450ms 150ms ease-out forwards',
      },
      colors: {
        // Observer teal-blue (#327e9c @ 500) replaces Tailwind's purple app-wide
        purple: {
          50: '#eef7fa', 100: '#d5eaf1', 200: '#aad4e3', 300: '#78b8cf', 400: '#4c99b6',
          500: '#327e9c', 600: '#286782', 700: '#22536a', 800: '#1f4557', 900: '#1c3a49', 950: '#0f2530',
        },
        // Custom dark mode colors
        dark: {
          bg: '#0f172a',
          surface: '#1e293b',
          border: '#334155',
          text: '#f1f5f9',
          'text-secondary': '#cbd5e1',
        }
      }
    },
  },
  plugins: [],
}
