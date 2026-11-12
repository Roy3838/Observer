/**
 * Wrap raw base64 image bytes in a data URL whose MIME type matches the actual bytes.
 *
 * Frames reach the model in different encodings: native (Tauri) screen capture hands back raw
 * JPEG, canvas re-encodes (crops, camera) are PNG. Strict endpoints (Cloudflare behind
 * /v1/systemone) reject a data URL whose declared type doesn't match the bytes, so sniff the
 * magic bytes from the base64 prefix instead of hardcoding image/png.
 */
export function imageDataUrl(b64: string): string {
  if (b64.startsWith('data:')) return b64;
  return `data:${sniffImageMime(b64)};base64,${b64}`;
}

/** MIME type from the first base64 chars (magic bytes). Falls back to image/png. */
export function sniffImageMime(b64: string): string {
  if (b64.startsWith('/9j/')) return 'image/jpeg';   // FF D8 FF
  if (b64.startsWith('iVBOR')) return 'image/png';   // 89 50 4E 47
  if (b64.startsWith('UklGR')) return 'image/webp';  // RIFF (WebP container)
  if (b64.startsWith('R0lGOD')) return 'image/gif';  // GIF8
  return 'image/png';
}
