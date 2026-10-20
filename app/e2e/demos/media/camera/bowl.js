// Timings for camera_bowl.html, in seconds of video (see camera_assets/camera.js). Fill these in when
// the recordings are added; the suite skips this demo until every file below exists.
//
//   bowl.mp4  overhead on the bowl: 10+ s of a full bowl (the hold), the dog eats, walks away with food left
//             (mark 'walked_away'), comes back, empties it (mark 'finished'), and some empty bowl after.
window.CAMERA = {
  label: 'CAM 3 · Kitchen', night: false, clock: '08:05:40',
  clips: {
    success: {
      file: 'media/camera/bowl.mp4', hold: [0, 10], end: 0,
      marks: [{ at: 0, name: 'walked_away' }, { at: 0, name: 'finished' }],
    },
  },
};
