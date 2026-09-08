// src/components/AgentCard/DetailedExecution.tsx
import React from 'react';
import { LogEntry } from '../../utils/logging';
import {
  ImageIcon, Save, Zap, Eye, ScanText, Mic, CheckCircle, XCircle,
  Bell, Mail, Send, MessageSquare, MessageSquarePlus,
  MessageSquareQuote, PlayCircle, StopCircle, Video, VideoOff, Tag, SquarePen, Hourglass, Volume2
} from 'lucide-react';

// Icon components for tools
function PushoverIcon() { 
  return <Send className="w-4 h-4" />; 
}

function SmsIcon() { 
  return <MessageSquarePlus className="w-4 h-4" />; 
}

function DiscordIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" {...props}>
      <path d="M 8.3164062 4.0039062 C 7.4484062 4.0039062 6.1811406 4.3226094 5.2441406 4.5996094 C 4.3771406 4.8556094 3.6552344 5.4479531 3.2402344 6.2519531 C 2.4062344 7.8649531 1.166625 11.064 1.015625 16 C 0.995625 16.67 1.2953594 17.314375 1.8183594 17.734375 C 2.8833594 18.589375 4.5907656 19.659375 7.0097656 19.984375 C 7.0847656 19.995375 7.1603281 20 7.2363281 20 C 7.7603281 20 8.2630781 19.758031 8.5800781 19.332031 L 9 18.65625 C 9 18.65625 10.653 19.019531 12 19.019531 L 12.013672 19.019531 C 13.360672 19.019531 15.013672 18.65625 15.013672 18.65625 L 15.433594 19.330078 C 15.751594 19.757078 16.253344 20 16.777344 20 C 16.853344 20 16.929859 19.994375 17.005859 19.984375 C 19.423859 19.659375 21.130313 18.589375 22.195312 17.734375 C 22.718312 17.314375 23.02 16.671953 23 16.001953 C 22.85 11.065953 21.607437 7.8659531 20.773438 6.2519531 C 20.358438 5.4489531 19.638484 4.8556094 18.771484 4.5996094 C 17.832484 4.3216094 16.565266 4.0039062 15.697266 4.0039062 C 15.443266 4.0039062 15.223641 4.0317031 15.056641 4.0957031 C 14.316641 4.3757031 14.001953 5.1445313 14.001953 5.1445312 C 14.001953 5.1445312 12.686625 4.9882813 12.015625 4.9882812 L 12 4.9882812 C 11.329 4.9882812 10.013672 5.1445312 10.013672 5.1445312 C 10.013672 5.1445312 9.6970313 4.37475 8.9570312 4.09375 C 8.7890312 4.03075 8.5704063 4.0039062 8.3164062 4.0039062 z M 8.3164062 5.5039062 C 8.3804063 5.5039062 8.4242188 5.5067656 8.4492188 5.5097656 C 8.5122188 5.5507656 8.5970469 5.6608906 8.6230469 5.7128906 C 8.8560469 6.2808906 9.4097187 6.6445312 10.011719 6.6445312 C 10.069719 6.6445312 10.1275 6.6417656 10.1875 6.6347656 C 10.6625 6.5787656 11.574672 6.4882812 12.013672 6.4882812 C 12.490672 6.4882812 13.484172 6.5947656 13.826172 6.6347656 C 13.889172 6.6417656 13.951672 6.6445312 14.013672 6.6445312 C 14.609672 6.6445312 15.145953 6.3081406 15.376953 5.7441406 C 15.414953 5.6631406 15.501453 5.5527188 15.564453 5.5117188 C 15.589453 5.5087187 15.633266 5.5058594 15.697266 5.5058594 C 16.231266 5.5058594 17.196703 5.7000625 18.345703 6.0390625 C 18.824703 6.1810625 19.213406 6.5004062 19.441406 6.9414062 C 20.148406 8.3094063 21.356 11.312828 21.5 16.048828 C 21.506 16.246828 21.415812 16.439406 21.257812 16.566406 C 19.933812 17.630406 18.435297 18.280953 16.779297 18.501953 C 16.732297 18.501953 16.68725 18.485984 16.65625 18.458984 L 16.390625 18.029297 C 16.833564 17.838865 17.256029 17.625199 17.640625 17.390625 C 17.994625 17.174625 18.105625 16.713375 17.890625 16.359375 C 17.674625 16.005375 17.212375 15.895375 16.859375 16.109375 C 15.733375 16.796375 13.795906 17.488281 12.003906 17.488281 C 10.216906 17.488281 8.2754844 16.796375 7.1464844 16.109375 C 6.7934844 15.894375 6.3321875 16.005375 6.1171875 16.359375 C 5.9021875 16.712375 6.0131875 17.175625 6.3671875 17.390625 C 6.7539143 17.625908 7.1782112 17.838346 7.6230469 18.029297 L 7.3574219 18.457031 C 7.3274219 18.483031 7.2817969 18.498047 7.2167969 18.498047 L 7.2070312 18.498047 C 5.5780312 18.279047 4.0818125 17.629406 2.7578125 16.566406 C 2.5998125 16.439406 2.509625 16.244875 2.515625 16.046875 C 2.659625 11.311875 3.8672187 8.3104063 4.5742188 6.9414062 C 4.8012188 6.5004062 5.1899219 6.1791094 5.6699219 6.0371094 C 6.8189219 5.6971094 7.7834062 5.5039062 8.3164062 5.5039062 z M 8.5 10 A 1.5 2 0 0 0 8.5 14 A 1.5 2 0 0 0 8.5 10 z M 15.5 10 A 1.5 2 0 0 0 15.5 14 A 1.5 2 0 0 0 15.5 10 z"></path>
    </svg>
  );
}

function WhatsAppIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 50 50" fill="currentColor" {...props}>
      <path d="M 25 2 C 12.309534 2 2 12.309534 2 25 C 2 29.079097 3.1186875 32.88588 4.984375 36.208984 L 2.0371094 46.730469 A 1.0001 1.0001 0 0 0 3.2402344 47.970703 L 14.210938 45.251953 C 17.434629 46.972929 21.092591 48 25 48 C 37.690466 48 48 37.690466 48 25 C 48 12.309534 37.690466 2 25 2 z M 25 4 C 36.609534 4 46 13.390466 46 25 C 46 36.609534 36.609534 46 25 46 C 21.278025 46 17.792121 45.029635 14.761719 43.333984 A 1.0001 1.0001 0 0 0 14.033203 43.236328 L 4.4257812 45.617188 L 7.0019531 36.425781 A 1.0001 1.0001 0 0 0 6.9023438 35.646484 C 5.0606869 32.523592 4 28.890107 4 25 C 4 13.390466 13.390466 4 25 4 z M 16.642578 13 C 16.001539 13 15.086045 13.23849 14.333984 14.048828 C 13.882268 14.535548 12 16.369511 12 19.59375 C 12 22.955271 14.331391 25.855848 14.613281 26.228516 L 14.615234 26.228516 L 14.615234 26.230469 C 14.588494 26.195329 14.973031 26.752191 15.486328 27.419922 C 15.999626 28.087653 16.717405 28.96464 17.619141 29.914062 C 19.422612 31.812909 21.958282 34.007419 25.105469 35.349609 C 26.554789 35.966779 27.698179 36.339417 28.564453 36.611328 C 30.169845 37.115426 31.632073 37.038799 32.730469 36.876953 C 33.55263 36.755876 34.456878 36.361114 35.351562 35.794922 C 36.246248 35.22873 37.12309 34.524722 37.509766 33.455078 C 37.786772 32.688244 37.927591 31.979598 37.978516 31.396484 C 38.003976 31.104927 38.007211 30.847602 37.988281 30.609375 C 37.969311 30.371148 37.989581 30.188664 37.767578 29.824219 C 37.302009 29.059804 36.774753 29.039853 36.224609 28.767578 C 35.918939 28.616297 35.048661 28.191329 34.175781 27.775391 C 33.303883 27.35992 32.54892 26.991953 32.083984 26.826172 C 31.790239 26.720488 31.431556 26.568352 30.914062 26.626953 C 30.396569 26.685553 29.88546 27.058933 29.587891 27.5 C 29.305837 27.918069 28.170387 29.258349 27.824219 29.652344 C 27.819619 29.649544 27.849659 29.663383 27.712891 29.595703 C 27.284761 29.383815 26.761157 29.203652 25.986328 28.794922 C 25.2115 28.386192 24.242255 27.782635 23.181641 26.847656 L 23.181641 26.845703 C 21.603029 25.455949 20.497272 23.711106 20.148438 23.125 C 20.171937 23.09704 20.145643 23.130901 20.195312 23.082031 L 20.197266 23.080078 C 20.553781 22.728924 20.869739 22.309521 21.136719 22.001953 C 21.515257 21.565866 21.68231 21.181437 21.863281 20.822266 C 22.223954 20.10644 22.02313 19.318742 21.814453 18.904297 L 21.814453 18.902344 C 21.828863 18.931014 21.701572 18.650157 21.564453 18.326172 C 21.426943 18.001263 21.251663 17.580039 21.064453 17.130859 C 20.690033 16.232501 20.272027 15.224912 20.023438 14.634766 L 20.023438 14.632812 C 19.730591 13.937684 19.334395 13.436908 18.816406 13.195312 C 18.298417 12.953717 17.840778 13.022402 17.822266 13.021484 L 17.820312 13.021484 C 17.450668 13.004432 17.045038 13 16.642578 13 z M 16.642578 15 C 17.028118 15 17.408214 15.004701 17.726562 15.019531 C 18.054056 15.035851 18.033687 15.037192 17.970703 15.007812 C 17.906713 14.977972 17.993533 14.968282 18.179688 15.410156 C 18.423098 15.98801 18.84317 16.999249 19.21875 17.900391 C 19.40654 18.350961 19.582292 18.773816 19.722656 19.105469 C 19.863021 19.437122 19.939077 19.622295 20.027344 19.798828 L 20.027344 19.800781 L 20.029297 19.802734 C 20.115837 19.973483 20.108185 19.864164 20.078125 19.923828 C 19.867096 20.342656 19.838461 20.445493 19.625 20.691406 C 19.29998 21.065838 18.968453 21.483404 18.792969 21.65625 C 18.639439 21.80707 18.36242 22.042032 18.189453 22.501953 C 18.016221 22.962578 18.097073 23.59457 18.375 24.066406 C 18.745032 24.6946 19.964406 26.679307 21.859375 28.347656 C 23.05276 29.399678 24.164563 30.095933 25.052734 30.564453 C 25.940906 31.032973 26.664301 31.306607 26.826172 31.386719 C 27.210549 31.576953 27.630655 31.72467 28.119141 31.666016 C 28.607627 31.607366 29.02878 31.310979 29.296875 31.007812 L 29.298828 31.005859 C 29.655629 30.601347 30.715848 29.390728 31.224609 28.644531 C 31.246169 28.652131 31.239109 28.646231 31.408203 28.707031 L 31.408203 28.708984 L 31.410156 28.708984 C 31.487356 28.736474 32.454286 29.169267 33.316406 29.580078 C 34.178526 29.990889 35.053561 30.417875 35.337891 30.558594 C 35.748225 30.761674 35.942113 30.893881 35.992188 30.894531 C 35.995572 30.982516 35.998992 31.07786 35.986328 31.222656 C 35.951258 31.624292 35.8439 32.180225 35.628906 32.775391 C 35.523582 33.066746 34.975018 33.667661 34.283203 34.105469 C 33.591388 34.543277 32.749338 34.852514 32.4375 34.898438 C 31.499896 35.036591 30.386672 35.087027 29.164062 34.703125 C 28.316336 34.437036 27.259305 34.092596 25.890625 33.509766 C 23.114812 32.325956 20.755591 30.311513 19.070312 28.537109 C 18.227674 27.649908 17.552562 26.824019 17.072266 26.199219 C 16.592866 25.575584 16.383528 25.251054 16.208984 25.021484 L 16.207031 25.019531 C 15.897202 24.609805 14 21.970851 14 19.59375 C 14 17.077989 15.168497 16.091436 15.800781 15.410156 C 16.132721 15.052495 16.495617 15 16.642578 15 z"></path>
    </svg>
  );
}

interface IterationGroup {
  id: string;
  logs: LogEntry[];
  startTime: string;
  duration?: number;
  hasError: boolean;
}

interface DetailedExecutionProps {
  group: IterationGroup;
}

const DetailedExecution: React.FC<DetailedExecutionProps> = ({ group }) => {
  // Render prompt content with images support (from original AgentLogViewer)
  const renderPromptContent = (content: any) => {
    if (content.modifiedPrompt && Array.isArray(content.images)) {
      return (
        <div className="space-y-4">
          <div className="font-medium text-gray-800 mb-2">Prompt:</div>
          <div className="whitespace-pre-wrap bg-gray-50 p-3 rounded border border-gray-200">
            {content.modifiedPrompt}
          </div>

          {content.images && content.images.length > 0 && (
            <div>
              <div className="font-medium text-gray-800 mb-2">Images:</div>
              <div className="grid grid-cols-1 gap-4">
                {content.images.map((imageData: string, index: number) => (
                  <div key={index} className="border border-gray-200 rounded p-2">
                    <img
                      src={`data:image/png;base64,${imageData}`}
                      alt={`Image ${index + 1}`}
                      className="max-w-full h-auto rounded"
                      onError={(e) => { // Fallback for different image types
                        const imgElement = e.target as HTMLImageElement;
                        const currentSrc = imgElement.src;
                        if (currentSrc.includes('image/png')) {
                          imgElement.src = `data:image/jpeg;base64,${imageData}`;
                        } else if (currentSrc.includes('image/jpeg')) {
                          imgElement.src = `data:image/webp;base64,${imageData}`;
                        } else { // Fallback placeholder
                          imgElement.src = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLWltYWdlLW9mZiI+PHBhdGggZD0iTTIuOTkgMy41SC4wMDEiLz48cGF0aCBkPSJNMTIuOTkgMy41aC0zIi8+PHBhdGggZD0iTTIwLjk5IDMuNWgtMyIvPjxwYXRoIGQ9Ik0xLjk5IDguNWgyIi8+PHBhdGggZD0iTTguOTkgOC41aDEwIi8+PHBhdGggZD0iTTEuOTkgMTMuNWgyIi8+PHBhdGggZD0iTTguOTkgMTMuNWg5Ljk5Ii8+PHBhdGggZD0iTTEuOTkgMTguNWgyIi8+PHBhdGggZD0iTTEzLjk5IDE4LjVoNiIvPjxwYXRoIGQ9Ik03Ljk5IDE4LjVoMSIvPjwvc3ZnPg==";
                          imgElement.style.padding = "20px";
                          imgElement.style.backgroundColor = "#f3f4f6"; // bg-gray-100
                        }
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }
    // Fallback for other object types
    return <pre className="text-sm overflow-auto">{JSON.stringify(content, null, 2)}</pre>;
  };

  // Get tool icon from tool name
  const getToolIcon = (toolName: string) => {
    const iconMap: Record<string, React.ElementType> = {
      sendDiscordBot: DiscordIcon,
      sendWhatsapp: WhatsAppIcon,
      sendSms: SmsIcon,
      sendPushover: PushoverIcon,
      sendEmail: Mail,
      notify: Bell,
      system_notify: Bell,
      getMemory: Save,
      setMemory: SquarePen,
      appendMemory: SquarePen,
      startAgent: PlayCircle,
      stopAgent: StopCircle,
      time: Hourglass,
      startClip: Video,
      stopClip: VideoOff,
      markClip: Tag,
      ask: MessageSquareQuote,
      message: MessageSquare,
      sound: Volume2,
    };
    return iconMap[toolName] || CheckCircle;
  };

  return (
    <div className="mt-4 pt-4 border-t border-gray-200 space-y-6">
      {/* SENSE Section */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <Eye className="w-4 h-4" />
          SENSE
        </h4>
        <div className="space-y-3">
          {group.logs
            .filter(log => log.details?.logType?.startsWith('sensor-'))
            .map((log, index) => {
              const logType = log.details?.logType || '';
              if (logType === 'sensor-screenshot') {
                return (
                  <div key={index} className="bg-blue-50 p-3 rounded border">
                    <div className="flex items-center gap-2 text-sm font-medium text-blue-800 mb-2">
                      <ImageIcon className="w-4 h-4" />
                      Screenshot captured
                    </div>
                    <div className="text-xs text-blue-600">
                      Size: {typeof log.details?.content === 'object' && log.details?.content?.size ? `${Math.round(log.details.content.size / 1024)}KB` : 'Unknown'}
                    </div>
                  </div>
                );
              } else if (logType === 'sensor-ocr') {
                return (
                  <div key={index} className="bg-green-50 p-3 rounded border">
                    <div className="flex items-center gap-2 text-sm font-medium text-green-800 mb-2">
                      <ScanText className="w-4 h-4" />
                      OCR Text Detected
                    </div>
                    <div className="text-xs text-green-700 bg-white p-2 rounded border max-h-24 overflow-y-auto">
                      {typeof log.details?.content === 'string' ? log.details.content : 'No content'}
                    </div>
                  </div>
                );
              } else if (logType === 'sensor-audio') {
                return (
                  <div key={index} className="bg-orange-50 p-3 rounded border">
                    <div className="flex items-center gap-2 text-sm font-medium text-orange-800 mb-2">
                      <Mic className="w-4 h-4" />
                      Audio Transcript
                    </div>
                    <div className="text-xs text-orange-700 bg-white p-2 rounded border max-h-24 overflow-y-auto">
                      {log.details?.content?.transcript || 'No transcript'}
                    </div>
                  </div>
                );
              }
              return null;
            })}
        </div>
      </div>
      
      {/* THINK Section */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <Save className="w-4 h-4" />
          THINK
        </h4>
        {(() => {
          const promptLog = group.logs.find(log => log.details?.logType === 'model-prompt');
          const responseLog = group.logs.find(log => log.details?.logType === 'model-response');
          
          return (
            <div className="space-y-3">
              {promptLog && (
                <div className="bg-blue-50 p-3 rounded border">
                  <div className="text-sm font-medium text-blue-800 mb-2">Prompt Sent</div>
                  <div className="text-xs text-blue-700">
                    {promptLog.details?.content ? renderPromptContent(promptLog.details.content) : 'No prompt content'}
                  </div>
                </div>
              )}
              {responseLog && (
                <div className="bg-green-50 p-3 rounded border">
                  <div className="text-sm font-medium text-green-800 mb-2">Model Response</div>
                  <div className="text-xs text-green-700 bg-white p-2 rounded border max-h-40 overflow-y-auto whitespace-pre-wrap">
                    {typeof responseLog.details?.content === 'string' ? responseLog.details.content : 'No response content'}
                  </div>
                </div>
              )}
            </div>
          );
        })()} 
      </div>
      
      {/* ACT Section */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <Zap className="w-4 h-4" />
          ACT
        </h4>
        <div className="space-y-3">
          {group.logs
            .filter(log => log.details?.logType === 'tool-success' || log.details?.logType === 'tool-error')
            .map((log, index) => {
              const isSuccess = log.details?.logType === 'tool-success';
              const toolName = log.details?.content?.tool || 'unknown';
              const ToolIcon = getToolIcon(toolName);
              
              return (
                <div key={index} className={`p-3 rounded border ${
                  isSuccess ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                }`}>
                  <div className={`flex items-center gap-2 text-sm font-medium mb-2 ${
                    isSuccess ? 'text-green-800' : 'text-red-800'
                  }`}>
                    <ToolIcon className="w-4 h-4" />
                    {isSuccess ? (
                      <CheckCircle className="w-4 h-4" />
                    ) : (
                      <XCircle className="w-4 h-4" />
                    )}
                    Tool: {toolName}()
                  </div>
                  <div className={`text-xs bg-white p-2 rounded border max-h-32 overflow-y-auto ${
                    isSuccess ? 'text-green-700' : 'text-red-700'
                  }`}>
                    {isSuccess ? (
                      <pre>{JSON.stringify(log.details?.content?.params || {}, null, 2)}</pre>
                    ) : (
                      <div>{log.details?.content?.error || 'Unknown error'}</div>
                    )}
                  </div>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
};

export default DetailedExecution;