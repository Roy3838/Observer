import type { CompleteAgent } from './agent_database';

export const PERSON_DETECTOR_ID = 'person_detector_onboarding';

export const PERSON_DETECTOR_AGENT: CompleteAgent = {
  id: PERSON_DETECTOR_ID,
  name: 'Person Detector',
  description: 'Watches your camera and celebrates when it detects a person',
  model_name: 'gemma-4-26B-a4b-it',
  system_prompt: `You are a visual detection agent. Your goal is to determine if a person is visible in the current camera feed.
1. **Describe:** In one sentence, briefly describe the image. If you see a person clearly describe them.
2. **Decide:** On a new line, output your final verdict: \`PERSON_DETECTED\` if a person is visible, otherwise output \`NO_PERSON\`.
$CAMERA`,
  loop_interval_seconds: 15,
};

export const PERSON_DETECTOR_CODE = `if (response.includes("PERSON_DETECTED")) {
  celebrate();
}`;
