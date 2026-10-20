// Timings for camera_room.html, in seconds of video (see camera_assets/camera.js). Fill these in when
// the recordings are added; the suite skips this demo until every file below exists.
//
//   room-person.mp4  a static view of a room or doorway: 10+ s of empty room (the hold), then a person
//                    walks in and stays in view. event = when they first appear.
//   room-cat.mp4     the same view: empty room, then only a cat (or a robot vacuum) crosses. event = when it
//                    first appears. No person ever.
window.CAMERA = {
  label: 'CAM 1 · Living room', night: false, clock: '14:02:11',
  clips: {
    success: { file: 'media/camera/room-person.mp4', hold: [0, 10], event: 0, end: 0, marks: [] }, // event: a person walks into the room
    fail: { file: 'media/camera/room-cat.mp4', hold: [0, 10], event: 0, end: 0, marks: [] }, // event: the cat crosses the room (no person ever appears)
  },
};
