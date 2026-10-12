// Timings for media/minecraft.mp4, in seconds of video. Fill these in when the recording is added:
//   hold:   a stretch with no deaths that loops while the MCP builds its agent
//   deaths: when each death happens (the "You Died!" screen appears)
//   end:    where the run stops
// patterns.spec.ts skips the Minecraft scenario until media/minecraft.mp4 exists.
window.MINECRAFT = { hold: [0, 20], deaths: [], end: 0 };
