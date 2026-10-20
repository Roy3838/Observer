// Timings for camera_yard.html, in seconds of video (see camera_assets/camera.js). Fill these in when
// the recordings are added; the suite skips this demo until every file below exists.
//
//   yard-coyote.mp4  a night/IR backyard view: 10+ s of empty yard, then a coyote walks through (licensed
//                    stock footage is fine). event = when it first appears.
//   yard-dog.mp4     the same kind of view: empty yard, then your dog walks through. event = when it appears.
window.CAMERA = {
  label: 'CAM 2 · Backyard', night: true, clock: '02:13:07',
  clips: {
    success: { file: 'media/camera/yard-coyote.mp4', hold: [0, 10], event: 0, end: 0, marks: [] }, // event: a coyote enters
    fail: { file: 'media/camera/yard-dog.mp4', hold: [0, 10], event: 0, end: 0, marks: [] }, // event: the dog enters
  },
};
