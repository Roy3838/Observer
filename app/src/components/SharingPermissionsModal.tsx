import React, { useState, useEffect, useRef } from 'react';
import { X, Monitor, Mic, Waves, Blend, Eye, Camera as CameraIcon, Play, Square, AlertTriangle, CheckCircle, Volume2, FileText, RotateCw, ChevronDown } from 'lucide-react';
import FixedDropdown from '@components/ui/FixedDropdown';
import { StreamManager, StreamState } from '@utils/streamManager';
import { Logger } from '@utils/logging';
import { SensorPlaceholder, SENSOR_DESCRIPTIONS, getRequiredStreamsFromSensors } from '@utils/sensorMapping';
import AudioTranscriptionVisualizer from '@components/shared/AudioTranscriptionVisualizer';

// --- Self-Contained UI Sub-Components for the Modal ---


/**
 * A placeholder shown when a stream preview is not yet active.
 */
const StreamPlaceholder: React.FC<{ Icon: React.ElementType; text: string }> = ({ Icon, text }) => (
  <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center aspect-video flex-1 min-w-0 text-gray-400 h-full">
    <Icon className="w-10 h-10 mb-2" />
    <span className="font-medium text-sm">{text}</span>
  </div>
);

// --- InfoTag-Style Sensor Components ---

/**
 * InfoTag-style sensor toggle matching StaticAgentView design patterns
 */
const SensorToggle: React.FC<{
  sensor: SensorPlaceholder;
  isActive: boolean;
  isEnabled: boolean;
  onToggle: (sensor: SensorPlaceholder, active: boolean) => void;
}> = ({ sensor, isActive, isEnabled, onToggle }) => {
  const info = SENSOR_DESCRIPTIONS[sensor];
  const getIconComponent = (iconName: string) => {
    switch (iconName) {
      case 'Monitor': return Monitor;
      case 'FileText': return FileText;
      case 'Camera': return CameraIcon;
      case 'Volume2': return Volume2;
      case 'Mic': return Mic;
      case 'Headphones': return Blend;
      default: return Monitor;
    }
  };
  const IconComponent = getIconComponent(info.icon);

  return (
    <button
      type="button"
      onClick={() => onToggle(sensor, !isActive)}
      disabled={!isEnabled}
      className={`group inline-flex flex-col items-center gap-1 px-3 py-2 rounded-md text-sm font-medium transition-all cursor-pointer ${
        isActive 
          ? 'bg-blue-100 text-blue-800 hover:bg-blue-200' 
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      } disabled:opacity-50 disabled:cursor-not-allowed`}
      title={info.description}
    >
      <div className="flex items-center gap-1.5">
        <IconComponent className="w-4 h-4" />
        <span>{info.name}</span>
        {isActive && <CheckCircle className="w-3.5 h-3.5" />}
      </div>
      <code className={`text-xs font-mono px-1.5 py-0.5 rounded ${
        isActive ? 'bg-blue-200 text-blue-900' : 'bg-gray-200 text-gray-600'
      }`}>
        {sensor}
      </code>
    </button>
  );
};


/**
 * Renders a live video stream with camera selection (if applicable).
 */
const VideoStream: React.FC<{ stream: MediaStream; streamType?: 'camera' | 'screen' }> = ({ stream, streamType = 'screen' }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [isSwitchingCamera, setIsSwitchingCamera] = useState(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    if (videoRef.current && stream) videoRef.current.srcObject = stream;
  }, [stream]);

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
    <div className="bg-black rounded-lg aspect-video flex-1 min-w-0 relative group">
      <video ref={videoRef} muted autoPlay playsInline className="w-full h-full object-contain"></video>

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
    </div>
  );
};

// --- Main Modal Component ---

interface SharingPermissionsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SharingPermissionsModal: React.FC<SharingPermissionsModalProps> = ({ isOpen, onClose }) => {
  // State management
  const [streams, setStreams] = useState<StreamState>(() => StreamManager.getCurrentState());
  const [selectedSensors, setSelectedSensors] = useState<Set<SensorPlaceholder>>(new Set());
  const [activeSensors, setActiveSensors] = useState<Set<SensorPlaceholder>>(new Set());
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const AGENT_ID = 'system-sensor-ui';

  // Available sensors for the modal (per user request)
  const AVAILABLE_SENSORS: SensorPlaceholder[] = ['$SCREEN', '$CAMERA', '$SCREEN_AUDIO', '$MICROPHONE', '$ALL_AUDIO'];

  // Listen to stream state changes
  useEffect(() => {
    if (!isOpen) return;
    const handleStreamUpdate = (newState: StreamState) => setStreams(newState);
    StreamManager.addListener(handleStreamUpdate);
    return () => StreamManager.removeListener(handleStreamUpdate);
  }, [isOpen]);

  // No cleanup when modal closes - streams should persist in background
  // The modal acts as a control panel for a persistent background agent
  // Streams only stop when explicitly stopped by user action

  // Handle sensor toggle in checklist
  const handleSensorToggle = (sensor: SensorPlaceholder, active: boolean) => {
    const updated = new Set(selectedSensors);
    if (active) {
      updated.add(sensor);
    } else {
      updated.delete(sensor);
    }
    setSelectedSensors(updated);
    setError(null); // Clear any previous errors
  };

  // Apply changes - TRANSACTIONAL APPROACH
  const handleApplyChanges = async () => {
    if (selectedSensors.size === 0) {
      // Stop all sensors
      Logger.info("SensorModal", `Stopping all sensors for agent '${AGENT_ID}'`);
      StreamManager.releaseStreamsForAgent(AGENT_ID);
      setActiveSensors(new Set());
      return;
    }

    setIsApplying(true);
    setError(null);

    try {
      const sensorsArray = Array.from(selectedSensors);
      const requiredStreams = getRequiredStreamsFromSensors(sensorsArray);
      
      Logger.info("SensorModal", `Applying sensor changes for agent '${AGENT_ID}': [${sensorsArray.join(', ')}] → [${requiredStreams.join(', ')}]`);
      
      // SINGLE TRANSACTIONAL REQUEST - like main_loop.ts
      await StreamManager.requestStreamsForAgent(AGENT_ID, requiredStreams);
      
      setActiveSensors(new Set(selectedSensors));
      Logger.info("SensorModal", `Successfully applied sensor changes. Active sensors: [${sensorsArray.join(', ')}]`);
      
    } catch (error) {
      Logger.error("SensorModal", `Failed to apply sensor changes for agent '${AGENT_ID}'`, error);
      
      // CRITICAL: ROBUST ERROR CLEANUP - matching main_loop.ts:148 pattern
      Logger.debug("SensorModal", `Cleaning up streams after error for agent '${AGENT_ID}'`);
      StreamManager.releaseStreamsForAgent(AGENT_ID);
      
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(`Failed to acquire sensors: ${errorMessage}`);
      setActiveSensors(new Set()); // Reset active sensors on error
      
    } finally {
      setIsApplying(false);
    }
  };

  // Stop all sensors
  const handleStopAll = () => {
    Logger.info("SensorModal", `User requested stop all sensors for agent '${AGENT_ID}'`);
    StreamManager.releaseStreamsForAgent(AGENT_ID);
    setActiveSensors(new Set());
    setSelectedSensors(new Set());
    setError(null);
  };


  const hasChanges = JSON.stringify([...selectedSensors].sort()) !== JSON.stringify([...activeSensors].sort());

  return (
    <div className={`fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[70] transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-3xl border border-gray-200 m-4 flex flex-col max-h-[90vh] animate-fade-in">
        
        {/* Header */}
        <div className="p-6 border-b border-gray-200 flex justify-between items-center flex-shrink-0">
          <h2 className="text-xl font-bold text-gray-800">Sensor Control Panel</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800 transition-colors">
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="p-6 space-y-4 overflow-y-auto">
          
          {/* Error Display */}
          {error && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-md text-sm flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* SENSOR CONTROL */}
          <div>
            <h4 className="text-xs font-semibold text-gray-500 mb-2">SENSOR CONTROL</h4>
            <div className="flex flex-wrap gap-2 mb-4">
              {AVAILABLE_SENSORS.map(sensor => (
                <SensorToggle
                  key={sensor}
                  sensor={sensor}
                  isActive={selectedSensors.has(sensor)}
                  isEnabled={!isApplying}
                  onToggle={handleSensorToggle}
                />
              ))}
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-3">
              <button
                onClick={handleApplyChanges}
                disabled={!hasChanges || isApplying}
                className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-md font-medium transition-all ${
                  hasChanges && !isApplying
                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                <Play className="w-3.5 h-3.5" />
                {isApplying ? 'Applying...' : 'Apply'}
              </button>
              
              {activeSensors.size > 0 && (
                <button
                  onClick={handleStopAll}
                  disabled={isApplying}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-md bg-red-100 text-red-700 hover:bg-red-200 transition-colors font-medium"
                >
                  <Square className="w-3.5 h-3.5" />
                  Stop All
                </button>
              )}
              
              <div className="text-xs text-gray-500">
                {activeSensors.size === 0 ? 'No sensors active' : `${activeSensors.size} active`}
              </div>
            </div>
          </div>

          {/* Live Previews */}
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {streams.screenVideoStream ? (
                <div className="animate-fade-in"><VideoStream stream={streams.screenVideoStream} streamType="screen" /></div>
              ) : (
                <StreamPlaceholder Icon={Eye} text="Screen Preview" />
              )}
              {streams.cameraStream ? (
                <VideoStream stream={streams.cameraStream} streamType="camera" />
              ) : (
                <StreamPlaceholder Icon={CameraIcon} text="Camera Preview" />
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {streams.screenAudioStream ? (
                <div className="animate-fade-in">
                  <AudioTranscriptionVisualizer
                    stream={streams.screenAudioStream}
                    streamType="screenAudio"
                    agentId={AGENT_ID}
                    title="System Audio"
                    icon={<Volume2 className="w-4 h-4" />}
                  />
                </div>
              ) : (
                <StreamPlaceholder Icon={Waves} text="System Audio" />
              )}
              {streams.microphoneStream ? (
                <div className="animate-fade-in">
                  <AudioTranscriptionVisualizer
                    stream={streams.microphoneStream}
                    streamType="microphone"
                    agentId={AGENT_ID}
                    title="Microphone"
                    icon={<Mic className="w-4 h-4" />}
                  />
                </div>
              ) : (
                <StreamPlaceholder Icon={Mic} text="Microphone" />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SharingPermissionsModal;
