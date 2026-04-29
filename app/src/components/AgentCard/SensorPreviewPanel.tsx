// components/AgentCard/SensorPreviewPanel.tsx
import React, { useMemo, useRef, useEffect, ReactNode, useState } from 'react';
import { Mic, Volume2, Crop, RotateCcw, RotateCw, ChevronDown, Save, Images, AlertTriangle, Monitor, Camera, Play, Smartphone } from 'lucide-react';
import { isMobile } from '@utils/platform';
import { isPipSupported } from '@utils/pictureInPicture';
import FixedDropdown from '@components/ui/FixedDropdown';
import { StreamState, AudioStreamType, StreamManager } from '@utils/streamManager';
import { CropConfig, setAgentCrop, getAgentCrop } from '@utils/screenCapture';
import { getAgentMemory, getAgentImageMemory } from '@utils/agent_database';
import { agentHasScreenSensor, agentHasCameraSensor, agentHasSensor } from './agentCapabilities';
import TranscriptionModal from '@components/AgentCard/TranscriptionModal';
import AudioTranscriptionVisualizer from '@components/shared/AudioTranscriptionVisualizer';
import { useTranscriptionState } from '@hooks/useTranscriptionState';

// --- Crop Overlay Component ---

interface CropOverlayProps {
  isActive: boolean;
  onCropSelect: (crop: CropConfig) => void;
  existingCrop?: CropConfig | null;
  videoElement: HTMLVideoElement | null;
}

const CropOverlay: React.FC<CropOverlayProps> = ({ isActive, onCropSelect, existingCrop, videoElement }) => {
  // Two-tap state: null = waiting for first tap, {x,y} = waiting for second tap
  const [firstCorner, setFirstCorner] = useState<{ x: number; y: number } | null>(null);
  const [previewPos, setPreviewPos] = useState<{ x: number; y: number } | null>(null);
  const [currentCrop, setCurrentCrop] = useState<CropConfig | null>(existingCrop || null);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setCurrentCrop(existingCrop || null);
  }, [existingCrop]);

  // Reset state when crop mode is deactivated
  useEffect(() => {
    if (!isActive) {
      setFirstCorner(null);
      setPreviewPos(null);
    }
  }, [isActive]);

  const getVideoScale = () => {
    if (!videoElement || !overlayRef.current) return { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 };

    const video = videoElement;
    const overlay = overlayRef.current;
    const videoRatio = video.videoWidth / video.videoHeight;
    const containerRatio = overlay.offsetWidth / overlay.offsetHeight;

    let scaleX, scaleY, offsetX = 0, offsetY = 0;

    if (videoRatio > containerRatio) {
      // Video is wider - fit to width
      scaleX = video.videoWidth / overlay.offsetWidth;
      scaleY = scaleX;
      offsetY = (overlay.offsetHeight - video.videoHeight / scaleY) / 2;
    } else {
      // Video is taller - fit to height
      scaleY = video.videoHeight / overlay.offsetHeight;
      scaleX = scaleY;
      offsetX = (overlay.offsetWidth - video.videoWidth / scaleX) / 2;
    }

    return { scaleX, scaleY, offsetX, offsetY };
  };

  const getPositionFromEvent = (e: React.MouseEvent | React.TouchEvent) => {
    if (!overlayRef.current) return null;

    const rect = overlayRef.current.getBoundingClientRect();
    const { offsetX, offsetY } = getVideoScale();

    let clientX: number, clientY: number;
    if ('touches' in e) {
      clientX = e.touches[0]?.clientX ?? e.changedTouches[0]?.clientX;
      clientY = e.touches[0]?.clientY ?? e.changedTouches[0]?.clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    return {
      x: clientX - rect.left - offsetX,
      y: clientY - rect.top - offsetY
    };
  };

  const handleTap = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isActive || !overlayRef.current) return;
    e.preventDefault();

    const pos = getPositionFromEvent(e);
    if (!pos) return;

    if (!firstCorner) {
      // First tap - set the first corner
      setFirstCorner(pos);
      setPreviewPos(pos);
      setCurrentCrop(null);
    } else {
      // Second tap - create the crop rectangle
      const { scaleX, scaleY } = getVideoScale();

      const x = Math.min(firstCorner.x, pos.x);
      const y = Math.min(firstCorner.y, pos.y);
      const width = Math.abs(pos.x - firstCorner.x);
      const height = Math.abs(pos.y - firstCorner.y);

      // Convert to video coordinates
      const videoCrop = {
        x: Math.max(0, Math.round(x * scaleX)),
        y: Math.max(0, Math.round(y * scaleY)),
        width: Math.round(width * scaleX),
        height: Math.round(height * scaleY)
      };

      // Only accept if the crop area is large enough
      if (videoCrop.width > 10 && videoCrop.height > 10) {
        setCurrentCrop(videoCrop);
        onCropSelect(videoCrop);
      }

      // Reset for next crop
      setFirstCorner(null);
      setPreviewPos(null);
    }
  };

  const handleMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isActive || !firstCorner || !overlayRef.current) return;

    const pos = getPositionFromEvent(e);
    if (pos) {
      setPreviewPos(pos);
    }
  };

  const handleCancel = () => {
    setFirstCorner(null);
    setPreviewPos(null);
  };

  const renderFirstCornerMarker = () => {
    if (!firstCorner || !overlayRef.current) return null;

    const { offsetX, offsetY } = getVideoScale();

    return (
      <div
        className="absolute w-4 h-4 -ml-2 -mt-2 pointer-events-none"
        style={{
          left: `${firstCorner.x + offsetX}px`,
          top: `${firstCorner.y + offsetY}px`,
        }}
      >
        {/* Pulsing dot */}
        <div className="absolute inset-0 bg-blue-500 rounded-full animate-ping opacity-75" />
        <div className="absolute inset-1 bg-blue-500 rounded-full" />
        {/* Crosshair lines */}
        <div className="absolute left-1/2 -top-2 w-px h-2 bg-blue-500" />
        <div className="absolute left-1/2 -bottom-2 w-px h-2 bg-blue-500" />
        <div className="absolute top-1/2 -left-2 w-2 h-px bg-blue-500" />
        <div className="absolute top-1/2 -right-2 w-2 h-px bg-blue-500" />
      </div>
    );
  };

  const renderPreviewRect = () => {
    if (!firstCorner || !previewPos || !overlayRef.current) return null;

    const { scaleX, scaleY, offsetX, offsetY } = getVideoScale();

    const x = Math.min(firstCorner.x, previewPos.x);
    const y = Math.min(firstCorner.y, previewPos.y);
    const width = Math.abs(previewPos.x - firstCorner.x);
    const height = Math.abs(previewPos.y - firstCorner.y);

    // Calculate video dimensions for display
    const videoWidth = Math.round(width * scaleX);
    const videoHeight = Math.round(height * scaleY);

    const style = {
      left: `${x + offsetX}px`,
      top: `${y + offsetY}px`,
      width: `${width}px`,
      height: `${height}px`,
    };

    return (
      <div
        className="absolute border-2 border-blue-500 border-dashed bg-blue-500 bg-opacity-10 pointer-events-none"
        style={style}
      >
        {videoWidth > 20 && videoHeight > 20 && (
          <div className="absolute -top-6 left-0 bg-blue-500 text-white text-xs px-1 rounded whitespace-nowrap">
            {videoWidth}×{videoHeight}
          </div>
        )}
      </div>
    );
  };

  const renderCropRect = () => {
    if (!currentCrop || !overlayRef.current || firstCorner) return null;

    const { scaleX, scaleY, offsetX, offsetY } = getVideoScale();

    const style = {
      left: `${currentCrop.x / scaleX + offsetX}px`,
      top: `${currentCrop.y / scaleY + offsetY}px`,
      width: `${currentCrop.width / scaleX}px`,
      height: `${currentCrop.height / scaleY}px`,
    };

    return (
      <div
        className="absolute border-2 border-blue-500 bg-blue-500 bg-opacity-20 pointer-events-none"
        style={style}
      >
        <div className="absolute -top-6 left-0 bg-blue-500 text-white text-xs px-1 rounded">
          {currentCrop.width}×{currentCrop.height}
        </div>
      </div>
    );
  };

  // Show overlay if actively cropping OR if there's an existing crop to display
  if (!isActive && !existingCrop) return null;

  return (
    <div
      ref={overlayRef}
      className={`absolute inset-0 z-10 ${isActive ? 'cursor-crosshair' : 'pointer-events-none'}`}
      onClick={isActive ? handleTap : undefined}
      onTouchEnd={isActive ? handleTap : undefined}
      onMouseMove={isActive ? handleMove : undefined}
      onTouchMove={isActive ? handleMove : undefined}
    >
      {renderCropRect()}
      {renderPreviewRect()}
      {renderFirstCornerMarker()}

      {/* Instructions */}
      {isActive && (
        <div className="absolute top-2 left-2 flex items-center gap-2">
          <div className="bg-black bg-opacity-70 text-white text-xs px-2 py-1.5 rounded">
            {firstCorner ? '📍 Tap to set opposite corner' : '👆 Tap to set first corner'}
          </div>
          {firstCorner && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleCancel();
              }}
              className="bg-red-500 bg-opacity-80 hover:bg-opacity-100 text-white text-xs px-2 py-1.5 rounded transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// --- Sensor Placeholder Component ---

interface SensorPlaceholderProps {
  sensorName: string;
  icon: ReactNode;
  onStart: () => void;
  isLoading?: boolean;
  colorClasses?: {
    border: string;
    bg: string;
    text: string;
    button: string;
    buttonHover: string;
  };
}

const SensorPlaceholder: React.FC<SensorPlaceholderProps> = ({
  sensorName,
  icon,
  onStart,
  isLoading = false,
  colorClasses = {
    border: 'border-gray-800',
    bg: 'bg-gray-500/10',
    text: 'text-gray-400',
    button: 'bg-gray-600',
    buttonHover: 'hover:bg-gray-700'
  }
}) => {
  return (
    <div className={`bg-black rounded-lg aspect-video flex-1 min-w-0 flex flex-col items-center justify-center gap-3 border ${colorClasses.border} ${colorClasses.bg}`}>
      <div className={`flex items-center gap-2 ${colorClasses.text}`}>
        {icon}
        <span className="text-sm">{sensorName} not active</span>
      </div>
      <button
        onClick={onStart}
        disabled={isLoading}
        className={`${colorClasses.button} ${colorClasses.buttonHover} text-white px-4 py-2 rounded text-sm font-medium transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        {isLoading ? (
          <>
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Starting...
          </>
        ) : (
          <>
            <Play className="w-4 h-4" />
            Start {sensorName}
          </>
        )}
      </button>
    </div>
  );
};

// --- Audio Sensor Placeholder Component ---

interface AudioPlaceholderProps {
  sensorName: string;
  icon: ReactNode;
  onStart: () => void;
  isLoading?: boolean;
  colorClasses?: {
    border: string;
    bg: string;
    text: string;
    button: string;
    buttonHover: string;
  };
}

const AudioPlaceholder: React.FC<AudioPlaceholderProps> = ({
  sensorName,
  icon,
  onStart,
  isLoading = false,
  colorClasses = {
    border: 'border-gray-700',
    bg: 'bg-gray-500/10',
    text: 'text-gray-400',
    button: 'bg-gray-600',
    buttonHover: 'hover:bg-gray-700'
  }
}) => {
  return (
    <div className={`bg-gray-800 rounded-lg p-3 flex-1 min-w-0 flex flex-col items-center justify-center gap-3 border ${colorClasses.border} ${colorClasses.bg}`}>
      <div className={`flex items-center gap-2 ${colorClasses.text}`}>
        {icon}
        <span className="text-sm">{sensorName} not active</span>
      </div>
      <button
        onClick={onStart}
        disabled={isLoading}
        className={`${colorClasses.button} ${colorClasses.buttonHover} text-white px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        {isLoading ? (
          <>
            <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Starting...
          </>
        ) : (
          <>
            <Play className="w-3 h-3" />
            Start
          </>
        )}
      </button>
    </div>
  );
};

// --- Helper Functions ---

const getActiveAudioStreamsForDisplay = (state: StreamState): { type: AudioStreamType; stream: MediaStream; title: string; icon: ReactNode; }[] => {
  if (state.allAudioStream) {
    return [{ type: 'allAudio', stream: state.allAudioStream, title: 'All Audio (Mixed)', icon: <><Volume2 className="w-3 h-3" /><Mic className="w-3 h-3 -ml-1" /></> }];
  }
  const activeStreams: { type: AudioStreamType; stream: MediaStream; title: string; icon: ReactNode; }[] = [];
  if (state.microphoneStream) {
    activeStreams.push({ type: 'microphone', stream: state.microphoneStream, title: 'Microphone', icon: <Mic className="w-4 h-4" /> });
  }
  if (state.screenAudioStream) {
    activeStreams.push({ type: 'screenAudio', stream: state.screenAudioStream, title: 'System Audio', icon: <Volume2 className="w-4 h-4" /> });
  }
  return activeStreams;
};

// --- Audio Waveform Component (with modal) ---

const AudioWaveformWithModal: React.FC<{
  stream: MediaStream;
  title: string;
  icon: React.ReactNode;
  type: AudioStreamType;
  agentId: string;
}> = ({ stream, title, icon, type, agentId }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const state = useTranscriptionState(type);

  return (
    <>
      <AudioTranscriptionVisualizer
        stream={stream}
        streamType={type}
        agentId={agentId}
        title={title}
        icon={icon}
        onClick={() => setIsModalOpen(true)}
      />

      <TranscriptionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        streamType={type}
        fullTranscript={state.fullTranscript}
        streamTitle={title}
      />
    </>
  );
};

// --- Video Stream Component ---

const VideoStream: React.FC<{
  stream: MediaStream;
  streamType: 'camera' | 'screen';
  agentId: string;
}> = ({ stream, streamType, agentId }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isCropMode, setIsCropMode] = useState(false);
  const [currentCrop, setCurrentCrop] = useState<CropConfig | null>(null);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [isSwitchingCamera, setIsSwitchingCamera] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  // PiP state (mobile only)
  const [isPipActive, setIsPipActive] = useState(false);
  const showPipButton = isMobile() && isPipSupported();

  // Load existing crop config
  useEffect(() => {
    const existingCrop = getAgentCrop(agentId, streamType);
    setCurrentCrop(existingCrop);
  }, [agentId, streamType]);

  // Load available camera devices (only for camera stream type)
  useEffect(() => {
    if (streamType !== 'camera') return;

    const loadCameras = async () => {
      const cameras = await StreamManager.getAvailableCameraDevices();
      setAvailableCameras(cameras);
    };

    loadCameras();

    // Refresh camera list when devices change
    const handleDeviceChange = () => loadCameras();
    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    return () => navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
  }, [streamType]);

  useEffect(() => {
    if (videoRef.current && stream) videoRef.current.srcObject = stream;
  }, [stream]);

  // Web PiP event listeners to track PiP state
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !showPipButton) return;

    const handleEnterPip = () => setIsPipActive(true);
    const handleLeavePip = () => setIsPipActive(false);

    video.addEventListener('enterpictureinpicture', handleEnterPip);
    video.addEventListener('leavepictureinpicture', handleLeavePip);

    return () => {
      video.removeEventListener('enterpictureinpicture', handleEnterPip);
      video.removeEventListener('leavepictureinpicture', handleLeavePip);
    };
  }, [showPipButton]);

  // Handle "Run in Background" button click
  // Uses web PiP API to show the actual video in PiP and keep app alive
  const handleRunInBackground = async () => {
    const video = videoRef.current;
    if (!video) return;

    try {
      if (video.readyState >= 2) {
        await video.requestPictureInPicture();
      } else {
        // Wait for video to be ready
        video.addEventListener('loadeddata', async () => {
          await video.requestPictureInPicture();
        }, { once: true });
      }
    } catch (err) {
      console.error('Failed to enter PiP:', err);
    }
  };

  // Handle stopping background mode
  const handleStopBackground = async () => {
    try {
      if (document.pictureInPictureElement) {
        await document.exitPictureInPicture();
      }
    } catch (err) {
      console.error('Failed to stop PiP:', err);
    }
  };

  const handleCropSelect = (crop: CropConfig) => {
    setAgentCrop(agentId, streamType, crop);
    setCurrentCrop(crop);
    setIsCropMode(false);
  };

  const handleClearCrop = () => {
    setAgentCrop(agentId, streamType, null);
    setCurrentCrop(null);
  };

  const handleSwitchCamera = async (deviceId: string) => {
    setIsSwitchingCamera(true);
    setIsDropdownOpen(false);
    try {
      await StreamManager.switchCameraDevice(deviceId);
    } catch (error) {
      console.error('Failed to switch camera:', error);
    } finally {
      setIsSwitchingCamera(false);
    }
  };

  const handleRotateCamera = async () => {
    if (availableCameras.length !== 2) return;

    const currentDeviceId = stream.getVideoTracks()[0]?.getSettings().deviceId;
    const otherCamera = availableCameras.find(cam => cam.deviceId !== currentDeviceId);

    if (otherCamera) {
      await handleSwitchCamera(otherCamera.deviceId);
    }
  };

  // Get current camera info
  const getCurrentCameraLabel = (): string => {
    const currentDeviceId = stream.getVideoTracks()[0]?.getSettings().deviceId;
    const currentCamera = availableCameras.find(cam => cam.deviceId === currentDeviceId);
    return currentCamera?.label || 'Camera';
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="bg-black rounded-lg overflow-hidden aspect-video flex-1 min-w-0 relative group">
        {/* Video element */}
        <video
          ref={videoRef}
          muted
          autoPlay
          playsInline
          className="w-full h-full object-contain [&::-webkit-media-controls-panel]:!hidden [&::-webkit-media-controls-panel-container]:!hidden [&::-webkit-media-controls-start-playback-button]:!hidden [&::-webkit-media-controls]:!hidden [&::-webkit-media-controls]:opacity-0"
        ></video>

        {/* Crop controls - show on hover, hide when crop mode is active */}
      {!isCropMode && (
        <div className="absolute top-2 left-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center gap-1 z-20">
          <button
            onClick={() => setIsCropMode(true)}
            className="bg-black bg-opacity-70 hover:bg-opacity-90 text-white p-1.5 rounded text-xs flex items-center gap-1 transition-colors"
            title={`Crop ${streamType}`}
          >
            <Crop className="w-3 h-3" />
          </button>

          {currentCrop && (
            <>
              <div className="bg-black bg-opacity-70 text-white text-xs px-1.5 py-1 rounded">
                {currentCrop.width}×{currentCrop.height}
              </div>
              <button
                onClick={handleClearCrop}
                className="bg-red-600 bg-opacity-80 hover:bg-opacity-100 text-white p-1 rounded transition-colors"
                title="Clear crop"
              >
                <RotateCcw className="w-3 h-3" />
              </button>
            </>
          )}
        </div>
      )}

      {/* Camera switcher - show on hover (only for camera streams with multiple devices) */}
      {streamType === 'camera' && availableCameras.length >= 2 && (
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20">
          {availableCameras.length === 2 ? (
            // Mobile mode: Simple rotate button for 2 cameras
            <button
              onClick={handleRotateCamera}
              disabled={isSwitchingCamera}
              className="bg-black bg-opacity-70 hover:bg-opacity-90 text-white p-1.5 rounded transition-colors disabled:opacity-50"
              title="Switch camera"
            >
              <RotateCw className={`w-3 h-3 ${isSwitchingCamera ? 'animate-spin' : ''}`} />
            </button>
          ) : (
            // Desktop mode: Dropdown for 3+ cameras
            <FixedDropdown
              isOpen={isDropdownOpen}
              onOpenChange={setIsDropdownOpen}
              trigger={({ ref, onClick, isOpen }) => (
                <button
                  ref={ref}
                  onClick={onClick}
                  disabled={isSwitchingCamera}
                  className="bg-black bg-opacity-70 hover:bg-opacity-90 text-white px-2 py-1.5 rounded text-xs flex items-center gap-1 transition-colors disabled:opacity-50 min-w-[120px] justify-between"
                  title="Switch camera"
                >
                  <span className="truncate">{getCurrentCameraLabel()}</span>
                  <ChevronDown className={`w-3 h-3 flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                </button>
              )}
            >
              {availableCameras.map((camera) => {
                const isCurrent = stream.getVideoTracks()[0]?.getSettings().deviceId === camera.deviceId;
                return (
                  <button
                    key={camera.deviceId}
                    onClick={() => handleSwitchCamera(camera.deviceId)}
                    disabled={isCurrent}
                    className={`w-full text-left px-3 py-2 text-xs transition-colors ${
                      isCurrent
                        ? 'bg-blue-600 bg-opacity-50 text-white cursor-default'
                        : 'text-gray-300 hover:bg-gray-700'
                    }`}
                  >
                    <div className="truncate">{camera.label || `Camera ${camera.deviceId.slice(0, 8)}`}</div>
                  </button>
                );
              })}
            </FixedDropdown>
          )}
        </div>
      )}

        {/* Crop overlay */}
        <CropOverlay
          isActive={isCropMode}
          onCropSelect={handleCropSelect}
          existingCrop={currentCrop}
          videoElement={videoRef.current}
        />
      </div>

      {/* Run in Background button (mobile only) */}
      {showPipButton && (
        !isPipActive ? (
          <button
            onClick={handleRunInBackground}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors text-sm"
          >
            <Smartphone className="w-4 h-4" />
            <span>Run in Background</span>
          </button>
        ) : (
          <button
            onClick={handleStopBackground}
            className="w-full bg-green-50 border border-green-200 rounded-lg p-2.5 hover:bg-green-100 transition-colors"
          >
            <div className="flex items-center justify-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-sm text-green-800 font-medium">
                Running in background (tap to stop)
              </span>
            </div>
          </button>
        )
      )}
    </div>
  );
};

// --- No Sensors Warning Component ---

const NoSensorsWarning: React.FC = () => (
  <div className="flex items-center justify-center gap-3 p-4 bg-red-50 rounded-lg border border-red-200 text-red-600">
    <AlertTriangle className="w-5 h-5" />
    <span className="text-sm font-medium">No Active Sensors</span>
  </div>
);

// --- Memory Preview Component ---

const MemoryPreview: React.FC<{ agentId: string }> = ({ agentId }) => {
  const [memory, setMemory] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);

  // Load initial memory
  useEffect(() => {
    const loadMemory = async () => {
      try {
        const memoryData = await getAgentMemory(agentId);
        setMemory(memoryData || '');
      } catch (error) {
        console.error('Failed to load memory:', error);
        setMemory('');
      } finally {
        setIsLoading(false);
      }
    };
    loadMemory();
  }, [agentId]);

  // Listen for memory updates via custom events
  useEffect(() => {
    const handleMemoryUpdate = async (event: CustomEvent) => {
      if (event.detail.agentId === agentId) {
        try {
          const memoryData = await getAgentMemory(agentId);
          setMemory(memoryData || '');
        } catch (error) {
          console.error('Failed to reload memory:', error);
        }
      }
    };

    window.addEventListener('agent-memory-update', handleMemoryUpdate as any);
    return () => {
      window.removeEventListener('agent-memory-update', handleMemoryUpdate as any);
    };
  }, [agentId]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 bg-emerald-50 px-3 py-2 rounded-lg border border-emerald-200 flex-1 min-w-0">
        <Save className="w-4 h-4 text-emerald-600 flex-shrink-0" />
        <span className="text-emerald-700 text-sm italic">Loading memory...</span>
      </div>
    );
  }

  if (!memory.trim()) {
    return (
      <div className="flex items-center gap-2 bg-emerald-50 px-3 py-2 rounded-lg border border-emerald-200 flex-1 min-w-0">
        <Save className="w-4 h-4 text-emerald-600 flex-shrink-0" />
        <span className="text-emerald-700 text-sm italic">No memory data yet</span>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2 bg-emerald-50 px-3 py-2 rounded-lg border border-emerald-200 flex-1 min-w-0">
      <Save className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <div className="text-xs font-medium text-emerald-600 mb-1">Memory</div>
        <div className="text-emerald-700 text-sm leading-relaxed break-words line-clamp-3 overflow-hidden">
          {memory}
        </div>
      </div>
    </div>
  );
};

// --- Image Memory Preview Component ---

const ImageMemoryPreview: React.FC<{ agentId: string }> = ({ agentId }) => {
  const [images, setImages] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Load initial image memory
  useEffect(() => {
    const loadImages = async () => {
      try {
        const imageData = await getAgentImageMemory(agentId);
        setImages(imageData || []);
      } catch (error) {
        console.error('Failed to load image memory:', error);
        setImages([]);
      } finally {
        setIsLoading(false);
      }
    };
    loadImages();
  }, [agentId]);

  // Listen for memory updates via custom events
  useEffect(() => {
    const handleMemoryUpdate = async (event: CustomEvent) => {
      if (event.detail.agentId === agentId) {
        try {
          const imageData = await getAgentImageMemory(agentId);
          setImages(imageData || []);
        } catch (error) {
          console.error('Failed to reload image memory:', error);
        }
      }
    };

    window.addEventListener('agent-memory-update', handleMemoryUpdate as any);
    return () => {
      window.removeEventListener('agent-memory-update', handleMemoryUpdate as any);
    };
  }, [agentId]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 bg-purple-50 px-3 py-2 rounded-lg border border-purple-200 flex-1 min-w-0">
        <Images className="w-4 h-4 text-purple-600 flex-shrink-0" />
        <span className="text-purple-700 text-sm italic">Loading image memory...</span>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="flex items-center gap-2 bg-purple-50 px-3 py-2 rounded-lg border border-purple-200 flex-1 min-w-0">
        <Images className="w-4 h-4 text-purple-600 flex-shrink-0" />
        <span className="text-purple-700 text-sm italic">No image memory yet</span>
      </div>
    );
  }

  const imagesToShow = images.slice(-3); // Show last 3 images

  return (
    <div className="bg-purple-50 px-3 py-2 rounded-lg border border-purple-200 flex-1 min-w-0">
      <div className="flex items-center gap-2 mb-2">
        <Images className="w-4 h-4 text-purple-600 flex-shrink-0" />
        <div className="text-xs font-medium text-purple-600">Image Memory</div>
        <span className="text-xs text-purple-500">({images.length} total)</span>
      </div>
      <div className="flex gap-2 flex-wrap">
        {imagesToShow.map((imageBase64, index) => (
          <img
            key={index}
            src={`data:image/png;base64,${imageBase64}`}
            alt={`Memory image ${index + 1}`}
            className="w-16 h-16 rounded border border-purple-200 object-cover flex-shrink-0"
            onError={(e) => {
              // Fallback to JPEG if PNG fails
              (e.target as HTMLImageElement).src = `data:image/jpeg;base64,${imageBase64}`;
            }}
          />
        ))}
      </div>
    </div>
  );
};

// --- Main SensorPreviewPanel Component ---

interface SensorPreviewPanelProps {
  agentId: string;
  streams: StreamState;
  systemPrompt: string;
}

const SensorPreviewPanel: React.FC<SensorPreviewPanelProps> = ({
  agentId,
  streams,
  systemPrompt,
}) => {
  const audioStreamsToDisplay = useMemo(() => getActiveAudioStreamsForDisplay(streams), [streams]);

  // Loading states for start buttons
  const [isStartingScreen, setIsStartingScreen] = useState(false);
  const [isStartingCamera, setIsStartingCamera] = useState(false);
  const [isStartingMicrophone, setIsStartingMicrophone] = useState(false);
  const [isStartingScreenAudio, setIsStartingScreenAudio] = useState(false);
  const [isStartingAllAudio, setIsStartingAllAudio] = useState(false);

  // Detect which sensors this agent actually uses
  const hasScreenSensor = useMemo(() => agentHasScreenSensor(systemPrompt), [systemPrompt]);
  const hasCameraSensor = useMemo(() => agentHasCameraSensor(systemPrompt), [systemPrompt]);
  const hasMemorySensor = useMemo(() => agentHasSensor(systemPrompt, 'MEMORY'), [systemPrompt]);
  const hasImageMemorySensor = useMemo(() => agentHasSensor(systemPrompt, 'IMEMORY') || agentHasSensor(systemPrompt, 'IMEMORY_OCR'), [systemPrompt]);
  const hasClipboardSensor = useMemo(() => agentHasSensor(systemPrompt, 'CLIPBOARD'), [systemPrompt]);
  const hasMicrophoneSensor = useMemo(() => agentHasSensor(systemPrompt, 'MICROPHONE'), [systemPrompt]);
  const hasScreenAudioSensor = useMemo(() => agentHasSensor(systemPrompt, 'SCREEN_AUDIO'), [systemPrompt]);
  const hasAllAudioSensor = useMemo(() => agentHasSensor(systemPrompt, 'ALL_AUDIO'), [systemPrompt]);

  // Check if we have any sensors configured (not necessarily active)
  const hasAnySensorsConfigured = hasScreenSensor || hasCameraSensor || hasMemorySensor || hasImageMemorySensor || hasClipboardSensor || hasMicrophoneSensor || hasScreenAudioSensor || hasAllAudioSensor;

  // Start handler functions
  const handleStartScreen = async () => {
    setIsStartingScreen(true);
    try {
      await StreamManager.requestStreamsForAgent(agentId, ['screenVideo']);
    } catch (error) {
      console.error('Failed to start screen share:', error);
    } finally {
      setIsStartingScreen(false);
    }
  };

  const handleStartCamera = async () => {
    setIsStartingCamera(true);
    try {
      await StreamManager.requestStreamsForAgent(agentId, ['camera']);
    } catch (error) {
      console.error('Failed to start camera:', error);
    } finally {
      setIsStartingCamera(false);
    }
  };

  const handleStartMicrophone = async () => {
    setIsStartingMicrophone(true);
    try {
      await StreamManager.requestStreamsForAgent(agentId, ['microphone']);
    } catch (error) {
      console.error('Failed to start microphone:', error);
    } finally {
      setIsStartingMicrophone(false);
    }
  };

  const handleStartScreenAudio = async () => {
    setIsStartingScreenAudio(true);
    try {
      await StreamManager.requestStreamsForAgent(agentId, ['screenAudio']);
    } catch (error) {
      console.error('Failed to start system audio:', error);
    } finally {
      setIsStartingScreenAudio(false);
    }
  };

  const handleStartAllAudio = async () => {
    setIsStartingAllAudio(true);
    try {
      await StreamManager.requestStreamsForAgent(agentId, ['allAudio']);
    } catch (error) {
      console.error('Failed to start all audio:', error);
    } finally {
      setIsStartingAllAudio(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Video Streams */}
      {hasScreenSensor && (
        streams.screenVideoStream ? (
          // Use PiP stream (with overlay) if available, otherwise use clean stream
          <VideoStream
            stream={streams.screenVideoStreamWithPip || streams.screenVideoStream}
            streamType="screen"
            agentId={agentId}
          />
        ) : (
          <SensorPlaceholder
            sensorName="Screen Share"
            icon={<Monitor className="w-8 h-8" />}
            onStart={handleStartScreen}
            isLoading={isStartingScreen}
            colorClasses={{
              border: 'border-purple-500/50',
              bg: 'bg-purple-500/10',
              text: 'text-purple-400',
              button: 'bg-purple-600',
              buttonHover: 'hover:bg-purple-700'
            }}
          />
        )
      )}
      {hasCameraSensor && (
        streams.cameraStream ? (
          <VideoStream stream={streams.cameraStream} streamType="camera" agentId={agentId} />
        ) : (
          <SensorPlaceholder
            sensorName="Camera"
            icon={<Camera className="w-8 h-8" />}
            onStart={handleStartCamera}
            isLoading={isStartingCamera}
            colorClasses={{
              border: 'border-purple-500/50',
              bg: 'bg-purple-500/10',
              text: 'text-purple-400',
              button: 'bg-purple-600',
              buttonHover: 'hover:bg-purple-700'
            }}
          />
        )
      )}

      {/* Memory Sensor Previews */}
      {(hasMemorySensor || hasImageMemorySensor) && (
        <div className="flex flex-col sm:flex-row gap-4">
          {hasMemorySensor && <MemoryPreview agentId={agentId} />}
          {hasImageMemorySensor && <ImageMemoryPreview agentId={agentId} />}
        </div>
      )}

      {/* Show warning if no sensors are configured */}
      {!hasAnySensorsConfigured && <NoSensorsWarning />}

      {/* Audio Streams */}
      <div className="grid grid-cols-1 gap-2">
        {/* Active audio streams */}
        {audioStreamsToDisplay.map(({ type, stream, title, icon }) => (
          <AudioWaveformWithModal key={type} stream={stream} title={title} icon={icon} type={type} agentId={agentId} />
        ))}

        {/* Placeholders for inactive audio sensors */}
        {hasAllAudioSensor && !streams.allAudioStream && (
          <AudioPlaceholder
            sensorName="All Audio (Mixed)"
            icon={<><Volume2 className="w-4 h-4" /><Mic className="w-4 h-4 -ml-1" /></>}
            onStart={handleStartAllAudio}
            isLoading={isStartingAllAudio}
            colorClasses={{
              border: 'border-orange-500/50',
              bg: 'bg-orange-500/10',
              text: 'text-orange-400',
              button: 'bg-orange-600',
              buttonHover: 'hover:bg-orange-700'
            }}
          />
        )}
        {hasMicrophoneSensor && !streams.microphoneStream && !streams.allAudioStream && (
          <AudioPlaceholder
            sensorName="Microphone"
            icon={<Mic className="w-4 h-4" />}
            onStart={handleStartMicrophone}
            isLoading={isStartingMicrophone}
            colorClasses={{
              border: 'border-amber-500/50',
              bg: 'bg-amber-500/10',
              text: 'text-amber-400',
              button: 'bg-amber-600',
              buttonHover: 'hover:bg-amber-700'
            }}
          />
        )}
        {hasScreenAudioSensor && !streams.screenAudioStream && !streams.allAudioStream && (
          <AudioPlaceholder
            sensorName="System Audio"
            icon={<Volume2 className="w-4 h-4" />}
            onStart={handleStartScreenAudio}
            isLoading={isStartingScreenAudio}
            colorClasses={{
              border: 'border-amber-500/50',
              bg: 'bg-amber-500/10',
              text: 'text-amber-400',
              button: 'bg-amber-600',
              buttonHover: 'hover:bg-amber-700'
            }}
          />
        )}
      </div>
    </div>
  );
};

export default SensorPreviewPanel;
