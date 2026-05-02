import { useState, useEffect } from 'react';
import { Copy, Check, ExternalLink } from 'lucide-react';

// Brand icons from Simple Icons (https://simpleicons.org)
const AppleIcon = ({ className = "w-6 h-6" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12.152 6.896c-.948 0-2.415-1.078-3.96-1.04-2.04.027-3.91 1.183-4.961 3.014-2.117 3.675-.546 9.103 1.519 12.09 1.013 1.454 2.208 3.09 3.792 3.039 1.52-.065 2.09-.987 3.935-.987 1.831 0 2.35.987 3.96.948 1.637-.026 2.676-1.48 3.676-2.948 1.156-1.688 1.636-3.325 1.662-3.415-.039-.013-3.182-1.221-3.22-4.857-.026-3.04 2.48-4.494 2.597-4.559-1.429-2.09-3.623-2.324-4.39-2.376-2-.156-3.675 1.09-4.61 1.09zM15.53 3.83c.843-1.012 1.4-2.427 1.245-3.83-1.207.052-2.662.805-3.532 1.818-.78.896-1.454 2.338-1.273 3.714 1.338.104 2.715-.688 3.559-1.701" />
  </svg>
);

const WindowsIcon = ({ className = "w-6 h-6" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M0 3.449L9.75 2.1v9.451H0m10.949-9.602L24 0v11.4H10.949M0 12.6h9.75v9.451L0 20.699M10.949 12.6H24V24l-12.9-1.801" />
  </svg>
);

const LinuxIcon = ({ className = "w-6 h-6" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12.504 0c-.155 0-.315.008-.48.021-4.226.333-3.105 4.807-3.17 6.298-.076 1.092-.3 1.953-1.05 3.02-.885 1.051-2.127 2.75-2.716 4.521-.278.832-.41 1.684-.287 2.489a.424.424 0 00-.11.135c-.26.268-.45.6-.663.839-.199.199-.485.267-.797.4-.313.136-.658.269-.864.68-.09.189-.136.394-.132.602 0 .199.027.4.055.536.058.399.116.728.04.97-.249.68-.28 1.145-.106 1.484.174.334.535.47.94.601.81.2 1.91.135 2.774.6.926.466 1.866.67 2.616.47.526-.116.97-.464 1.208-.946.587-.003 1.23-.269 2.26-.334.699-.058 1.574.267 2.577.2.025.134.063.198.114.333l.003.003c.391.778 1.113 1.132 1.884 1.071.771-.06 1.592-.536 2.257-1.306.631-.765 1.683-1.084 2.378-1.503.348-.199.629-.469.649-.853.023-.4-.2-.811-.714-1.376v-.097l-.003-.003c-.17-.2-.25-.535-.338-.926-.085-.401-.182-.786-.492-1.046h-.003c-.059-.054-.123-.067-.188-.135a.357.357 0 00-.19-.064c.431-1.278.264-2.55-.173-3.694-.533-1.41-1.465-2.638-2.175-3.483-.796-1.005-1.576-1.957-1.56-3.368.026-2.152.236-6.133-3.544-6.139zm.529 3.405h.013c.213 0 .396.062.584.198.19.135.33.332.438.533.105.259.158.459.166.724 0-.02.006-.04.006-.06v.105a.086.086 0 01-.004-.021l-.004-.024a1.807 1.807 0 01-.15.706.953.953 0 01-.213.335.71.71 0 00-.088-.042c-.104-.045-.198-.064-.284-.133a1.312 1.312 0 00-.22-.066c.05-.06.146-.133.183-.198.053-.128.082-.264.088-.402v-.02a1.21 1.21 0 00-.061-.4c-.045-.134-.101-.2-.183-.333-.084-.066-.167-.132-.267-.132h-.016c-.093 0-.176.03-.262.132a.8.8 0 00-.205.334 1.18 1.18 0 00-.09.4v.019c.002.089.008.179.02.267-.193-.067-.438-.135-.607-.202a1.635 1.635 0 01-.018-.2v-.02a1.772 1.772 0 01.15-.768c.082-.22.232-.406.43-.533a.985.985 0 01.594-.2zm-2.962.059h.036c.142 0 .27.048.399.135.146.129.264.288.344.465.09.199.14.4.153.667v.004c.007.134.006.2-.002.266v.08c-.03.007-.056.018-.083.024-.152.055-.274.135-.393.2.012-.09.013-.18.003-.267v-.015c-.012-.133-.04-.2-.082-.333a.613.613 0 00-.166-.267.248.248 0 00-.183-.064h-.021c-.071.006-.13.04-.186.132a.552.552 0 00-.12.27.944.944 0 00-.023.33v.015c.012.135.037.2.08.334.046.134.098.2.166.268.01.009.02.018.034.024-.07.057-.117.07-.176.136a.304.304 0 01-.131.068 2.62 2.62 0 01-.275-.402 1.772 1.772 0 01-.155-.667 1.759 1.759 0 01.08-.668 1.43 1.43 0 01.283-.535c.128-.133.26-.2.418-.2zm1.37 1.706c.332 0 .733.065 1.216.399.293.2.523.269 1.052.468h.003c.255.136.405.266.478.399v-.131a.571.571 0 01.016.47c-.123.31-.516.643-1.063.842v.002c-.268.135-.501.333-.775.465-.276.135-.588.292-1.012.267a1.139 1.139 0 01-.448-.067 3.566 3.566 0 01-.322-.198c-.195-.135-.363-.332-.612-.465v-.005h-.005c-.4-.246-.616-.512-.686-.71-.07-.268-.005-.47.193-.6.224-.135.38-.271.483-.336.104-.074.143-.102.176-.131h.002v-.003c.169-.202.436-.47.839-.601.139-.036.294-.065.466-.065zm2.8 2.142c.358 1.417 1.196 3.475 1.735 4.473.286.534.855 1.659 1.102 3.024.156-.005.33.018.513.064.646-1.671-.546-3.467-1.089-3.966-.22-.2-.232-.335-.123-.335.59.534 1.365 1.572 1.646 2.757.13.535.16 1.104.021 1.67.067.028.135.06.205.067 1.032.534 1.413.938 1.23 1.537v-.002c-.06-.135-.12-.2-.283-.334-.381-.336-.767-.532-1.229-.468l-.004-.002c-.487.065-.802.336-.91.803-.18.133-.29.333-.37.465-.08.135-.12.2-.16.267-.394-.464-.777-.936-1.048-1.272-.473-.6-1.103-1.202-1.511-2.006-.406-.801-.596-1.67-.106-2.803-.222-.2-.453-.333-.725-.4-.303.935-.243 1.67.063 2.537.247.667.715 1.335 1.262 2.005-.896.135-1.478.2-1.978.467-.99.535-1.058 1.47-.21 2.338.035-.067.086-.2.152-.333.064-.135.14-.267.263-.4.088-.066.194-.2.347-.2.086 0 .171.065.239.132.07.066.11.135.084.2a.606.606 0 00.102.2.521.521 0 00.194.132c-.097-.066-.12-.197-.12-.333 0-.066.022-.135.065-.2.043-.066.11-.135.205-.135.176 0 .355.135.465.267a.6.6 0 01.118.333c0 .2-.073.465-.323.6-.066.066-.163.132-.262.2h-.002a.534.534 0 01-.357.067c-.197 0-.394-.067-.59-.2a1.4 1.4 0 01-.323-.333c-.066-.066-.13-.2-.195-.333-.065-.135-.13-.267-.195-.4a4.064 4.064 0 00-.228-.467c-.1-.135-.18-.27-.313-.4-.12-.135-.28-.267-.49-.4a9.6 9.6 0 01-.365-.333c-.13-.135-.258-.267-.323-.465-.13-.4-.108-.866.196-1.404.087-.066.194-.2.323-.333l.164-.165a4.482 4.482 0 01-.2-1.269c-.02-.8.147-1.672.537-2.542-.186.002-.376.022-.554.062-.534.135-.99.468-1.37.802-.378.333-.696.735-.934 1.137a3.867 3.867 0 00-.466 1.403c.018-.066.044-.132.063-.2l.02-.064a4.09 4.09 0 01.21-.467c.13-.267.29-.533.512-.8.224-.267.474-.533.81-.733.168-.1.352-.2.553-.267l.133-.04a2.265 2.265 0 01.413-.067c.132 0 .265.014.397.045l.017.004c.067.017.132.036.2.057l.001.001c.006.002.014.003.02.005l.022.007c.04.014.08.028.118.043.072.028.142.06.21.094.018.009.036.019.053.028l.065.035c.06.033.118.067.175.104.015.01.031.02.046.03l.023.016c.018.012.034.024.05.035.096.07.186.144.27.22.026.023.052.046.076.07a1.875 1.875 0 01.26.291c.105.126.205.26.29.394.026.042.053.086.077.129.007.011.014.023.02.035l.02.034c.134.229.222.476.292.598.1.135.2.335.347.467-.098-.4-.17-.67-.27-1.003z" />
  </svg>
);

const MobileIcon = ({ className = "w-6 h-6" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="5" y="2" width="14" height="20" rx="2" ry="2" />
    <line x1="12" y1="18" x2="12.01" y2="18" />
  </svg>
);

const BrowserIcon = ({ className = "w-6 h-6" }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="2" y1="12" x2="22" y2="12" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
  </svg>
);

type Platform = 'windows' | 'macos' | 'linux' | 'mobile' | 'browser';

const platforms: { id: Platform; label: string; icon: React.ReactNode }[] = [
  { id: 'windows', label: 'Windows', icon: <WindowsIcon className="w-8 h-8" /> },
  { id: 'macos', label: 'macOS', icon: <AppleIcon className="w-8 h-8" /> },
  { id: 'linux', label: 'Linux', icon: <LinuxIcon className="w-8 h-8" /> },
  { id: 'mobile', label: 'Mobile', icon: <MobileIcon className="w-8 h-8" /> },
  { id: 'browser', label: 'Browser', icon: <BrowserIcon className="w-8 h-8" /> },
];

// Copy button with feedback
const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="p-2 text-gray-500 hover:text-white transition-colors"
      title="Copy to clipboard"
    >
      {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
    </button>
  );
};

// Command block with copy functionality
const CommandBlock = ({ command }: { command: string }) => (
  <div className="flex items-center justify-between bg-[#1a1f2e] rounded-lg px-4 py-3 font-mono text-sm text-gray-300 border border-white/10">
    <code>{command}</code>
    <CopyButton text={command} />
  </div>
);

// Download button
const DownloadButton = ({
  children,
  onClick,
  disabled,
  primary,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
}) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`
      px-6 py-3 rounded-full font-medium transition-all
      ${disabled
        ? 'bg-white/5 text-gray-600 cursor-not-allowed'
        : primary
          ? 'bg-white text-gray-900 hover:bg-gray-100'
          : 'bg-white/10 text-white hover:bg-white/20'
      }
    `}
  >
    {children}
  </button>
);

// Tooltip for disabled buttons
const DisabledTooltip = ({ children }: { children: React.ReactNode }) => (
  <div className="group relative inline-block">
    {children}
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 text-xs text-white bg-gray-800 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
      Coming soon!
      <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-800" />
    </div>
  </div>
);

const DownloadsSection = () => {
  const [version, setVersion] = useState<string | null>(null);
  const [activePlatform, setActivePlatform] = useState<Platform>('windows');

  useEffect(() => {
    fetch('https://api.github.com/repos/Roy3838/Observer/releases/latest')
      .then((res) => res.json())
      .then((data) => setVersion(data.tag_name?.replace(/^v/, '') ?? '2.1.1'))
      .catch(() => setVersion('2.1.1'));
  }, []);

  const downloadFile = (url: string) => {
    if (!version) return;
    window.location.href = url;
  };

  const openExternal = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const urls = {
    windowsExe: `https://github.com/Roy3838/Observer/releases/download/v${version}/Observer_${version}_x64-setup.exe`,
    windowsMsi: `https://github.com/Roy3838/Observer/releases/download/v${version}/Observer_${version}_x64_en-US.msi`,
    macSilicon: `https://github.com/Roy3838/Observer/releases/download/v${version}/Observer_${version}_aarch64.dmg`,
    macIntel: `https://github.com/Roy3838/Observer/releases/download/v${version}/Observer_${version}_x64.dmg`,
    deb: `https://github.com/Roy3838/Observer/releases/download/v${version}/Observer_${version}_amd64.deb`,
    rpm: `https://github.com/Roy3838/Observer/releases/download/v${version}/Observer-${version}-1.x86_64.rpm`,
    appImage: `https://github.com/Roy3838/Observer/releases/download/v${version}/Observer_${version}_amd64.AppImage`,
    ios: 'https://apps.apple.com/mx/app/observer-ai/id6758222050?l=en-GB',
    web: 'https://app.observer-ai.com',
  };

  const renderContent = () => {
    switch (activePlatform) {
      case 'windows':
        return (
          <div className="flex flex-col items-center gap-4">
            <div className="flex flex-wrap justify-center gap-3">
              <DownloadButton primary onClick={() => downloadFile(urls.windowsExe)}>
                Download .exe
              </DownloadButton>
              <DownloadButton onClick={() => downloadFile(urls.windowsMsi)}>
                Download .msi
              </DownloadButton>
            </div>
            <p className="text-sm text-gray-500">Windows 10 or later</p>
          </div>
        );

      case 'macos':
        return (
          <div className="flex flex-col items-center gap-4">
            <div className="flex flex-wrap justify-center gap-3">
              <DownloadButton primary onClick={() => downloadFile(urls.macSilicon)}>
                Download for Apple Silicon
              </DownloadButton>
              <DownloadButton onClick={() => downloadFile(urls.macIntel)}>
                Download for Intel
              </DownloadButton>
            </div>
            <p className="text-sm text-gray-500">macOS 11 Big Sur or later</p>
          </div>
        );

      case 'linux':
        return (
          <div className="flex flex-col items-center gap-8 w-full max-w-2xl">
            {/* Package Downloads */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full">
              {/* .deb */}
              <div className="flex flex-col items-center gap-2">
                <DownloadButton primary onClick={() => downloadFile(urls.deb)}>
                  Download .deb
                </DownloadButton>
                <p className="text-xs text-gray-500 text-center">
                  Debian · Ubuntu · Mint · Pop!_OS
                </p>
              </div>

              {/* .rpm */}
              <div className="flex flex-col items-center gap-2">
                <DownloadButton primary onClick={() => downloadFile(urls.rpm)}>
                  Download .rpm
                </DownloadButton>
                <p className="text-xs text-gray-500 text-center">
                  Fedora · openSUSE · RHEL · CentOS
                </p>
              </div>

              {/* .AppImage */}
              <div className="flex flex-col items-center gap-2">
                <DownloadButton onClick={() => downloadFile(urls.appImage)}>
                  Download .AppImage
                </DownloadButton>
                <p className="text-xs text-gray-500 text-center">
                  Works on any distro
                </p>
              </div>
            </div>

            {/* AUR Section */}
            <div className="w-full border-t border-white/10 pt-6">
              <p className="text-sm text-gray-400 text-center mb-3">
                Arch Linux · Manjaro · EndeavourOS · CachyOS
              </p>
              <CommandBlock command="git clone https://aur.archlinux.org/observer-ai.git && cd observer-ai && makepkg -si" />
            </div>
          </div>
        );

      case 'mobile':
        return (
          <div className="flex flex-col items-center gap-4">
            <div className="flex flex-wrap justify-center gap-3">
              <DownloadButton primary onClick={() => openExternal(urls.ios)}>
                Download for iOS
              </DownloadButton>
              <DisabledTooltip>
                <DownloadButton disabled onClick={() => {}}>
                  Download for Android
                </DownloadButton>
              </DisabledTooltip>
            </div>
            <p className="text-sm text-gray-500">iOS 15 or later</p>
          </div>
        );

      case 'browser':
        return (
          <div className="flex flex-col items-center gap-4">
            <p className="text-gray-400 text-center max-w-md">
              Works in any modern browser. No installation required.
            </p>
            <DownloadButton primary onClick={() => openExternal(urls.web)}>
              <span className="flex items-center gap-2">
                Open Web App
                <ExternalLink className="w-4 h-4" />
              </span>
            </DownloadButton>
          </div>
        );
    }
  };

  return (
    <section className="py-24 md:py-32 bg-[#0D1321]" id="downloads">
      <div className="container mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-2">
            Download Observer
          </h2>
          {version && (
            <p className="text-sm text-gray-500">v{version}</p>
          )}
        </div>

        {/* Platform Tabs */}
        <div className="flex justify-center mb-10">
          {/* Desktop: single row */}
          <div className="hidden md:inline-flex gap-1 p-1.5 bg-white/5 rounded-2xl border border-white/10">
            {platforms.map((platform) => (
              <button
                key={platform.id}
                onClick={() => setActivePlatform(platform.id)}
                className={`
                  flex flex-col items-center gap-2 px-6 py-4 rounded-xl transition-all
                  ${activePlatform === platform.id
                    ? 'bg-white/10 text-white'
                    : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                  }
                `}
              >
                {platform.icon}
                <span className="text-sm font-medium">{platform.label}</span>
              </button>
            ))}
          </div>

          {/* Mobile: two rows */}
          <div className="md:hidden flex flex-col gap-1 p-1.5 bg-white/5 rounded-2xl border border-white/10">
            <div className="flex gap-1">
              {platforms.slice(0, 3).map((platform) => (
                <button
                  key={platform.id}
                  onClick={() => setActivePlatform(platform.id)}
                  className={`
                    flex flex-col items-center gap-2 px-4 py-3 rounded-xl transition-all
                    ${activePlatform === platform.id
                      ? 'bg-white/10 text-white'
                      : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                    }
                  `}
                >
                  {platform.icon}
                  <span className="text-xs font-medium">{platform.label}</span>
                </button>
              ))}
            </div>
            <div className="flex gap-1 justify-center">
              {platforms.slice(3).map((platform) => (
                <button
                  key={platform.id}
                  onClick={() => setActivePlatform(platform.id)}
                  className={`
                    flex flex-col items-center gap-2 px-4 py-3 rounded-xl transition-all
                    ${activePlatform === platform.id
                      ? 'bg-white/10 text-white'
                      : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                    }
                  `}
                >
                  {platform.icon}
                  <span className="text-xs font-medium">{platform.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content Panel */}
        <div className="flex justify-center">
          <div className="min-h-[200px] flex items-center justify-center w-full">
            {renderContent()}
          </div>
        </div>
      </div>
    </section>
  );
};

export default DownloadsSection;
