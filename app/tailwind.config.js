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
      },
      animation: {
        nudge: 'nudge 1.1s ease-in-out infinite',
        'nudge-soft': 'nudge-soft 1.1s ease-in-out infinite',
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
