// Timings for camera_printer.html, in seconds of video (see camera_assets/camera.js). Fill these in when
// the recordings are added; the suite skips this demo until every file below exists.
//
//   printer-spaghetti.mp4  a sped-up timelapse: 10+ s of normal printing (the hold), then the part detaches
//                          and it turns into spaghetti (no glue on the bed does it). event = when it detaches.
//   printer-finished.mp4   the same kind of timelapse, ending with a clean finished part and the nozzle
//                          parked. event = when it parks.
window.CAMERA = {
  label: 'PRUSA MK4 · BED CAM', night: false, clock: '23:41:52',
  clips: {
    success: { file: 'media/camera/printer-spaghetti.mp4', hold: [0, 10], event: 0, end: 0, marks: [] }, // event: the print detaches / spaghetti starts
    fail: { file: 'media/camera/printer-finished.mp4', hold: [0, 10], event: 0, end: 0, marks: [] }, // event: the print is done, the nozzle parks
  },
};
