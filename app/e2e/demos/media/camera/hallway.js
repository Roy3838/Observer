// Timings for camera_hallway.html, in seconds of video (see camera_assets/camera.js). Fill these in when
// the recordings are added; the suite skips this demo until every file below exists.
//
//   hallway.mp4  the hall: 10+ s empty (the hold), then 2-3 people in clearly different clothes come in and
//                leave at different times. One mark per entrance and exit, with what they're wearing.
window.CAMERA = {
  label: 'CAM 4 · Front hall', night: false, clock: '17:20:03',
  clips: {
    success: {
      file: 'media/camera/hallway.mp4', hold: [0, 10], end: 0,
      marks: [
        { at: 0, name: 'enter', data: { person: 'red jacket' } },
        { at: 0, name: 'enter', data: { person: 'grey hoodie' } },
        { at: 0, name: 'leave', data: { person: 'red jacket' } },
        { at: 0, name: 'leave', data: { person: 'grey hoodie' } },
      ],
    },
  },
};
