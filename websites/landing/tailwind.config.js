/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'golos': ['Golos Text', 'sans-serif'],
        'serif': ['Newsreader', 'Georgia', 'serif'],
      },
    },
  },
  plugins: [],
}
