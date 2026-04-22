use dialoguer::{theme::ColorfulTheme, Input, Select};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NotifyChannel {
    Discord,
    Telegram,
    Sms,
    WhatsApp,
    Call,
    Email,
}

impl NotifyChannel {
    pub fn display_name(&self) -> &'static str {
        match self {
            NotifyChannel::Discord => "Discord (webhook)",
            NotifyChannel::Telegram => "Telegram",
            NotifyChannel::Sms => "SMS",
            NotifyChannel::WhatsApp => "WhatsApp",
            NotifyChannel::Call => "Voice Call",
            NotifyChannel::Email => "Email",
        }
    }

    pub fn requires_login(&self) -> bool {
        !matches!(self, NotifyChannel::Discord)
    }

    pub fn all() -> &'static [NotifyChannel] {
        &[
            NotifyChannel::Discord,
            NotifyChannel::Telegram,
            NotifyChannel::Sms,
            NotifyChannel::WhatsApp,
            NotifyChannel::Call,
            NotifyChannel::Email,
        ]
    }

    pub fn is_phone_based(&self) -> bool {
        matches!(
            self,
            NotifyChannel::Sms | NotifyChannel::WhatsApp | NotifyChannel::Call
        )
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub default_channel: Option<NotifyChannel>,
    #[serde(default)]
    pub discord: DiscordConfig,
    #[serde(default)]
    pub telegram: TelegramConfig,
    #[serde(default)]
    pub phone: PhoneConfig,
    #[serde(default)]
    pub email: EmailConfig,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DiscordConfig {
    pub webhook_url: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TelegramConfig {
    pub chat_id: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PhoneConfig {
    pub number: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EmailConfig {
    pub address: Option<String>,
}

impl Config {
    /// Get the config directory path (~/.config/observe/)
    fn config_dir() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("observe"))
    }

    /// Get the config file path
    fn config_path() -> Option<PathBuf> {
        Self::config_dir().map(|p| p.join("config.toml"))
    }

    /// Load config from file, or return default if not found
    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };

        match fs::read_to_string(&path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Save config to file
    pub fn save(&self) -> io::Result<()> {
        let Some(dir) = Self::config_dir() else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Could not determine config directory",
            ));
        };

        fs::create_dir_all(&dir)?;

        let path = dir.join("config.toml");
        let contents = toml::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        fs::write(path, contents)
    }

    /// Check if this is a first run (no default channel configured)
    pub fn is_first_run(&self) -> bool {
        self.default_channel.is_none()
    }

    /// Show interactive TUI to select notification channel
    pub fn select_channel_interactive(&mut self) -> io::Result<NotifyChannel> {
        eprintln!();
        eprintln!("  Welcome to Observer CLI!");
        eprintln!("  Select your notification method:");
        eprintln!();

        let channels = NotifyChannel::all();
        let items: Vec<&str> = channels.iter().map(|c| c.display_name()).collect();

        let selection = Select::with_theme(&ColorfulTheme::default())
            .items(&items)
            .default(0)
            .interact()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let channel = channels[selection];
        Ok(channel)
    }

    /// Prompt for the secret/config value for a given channel using TUI
    pub fn prompt_channel_secret_interactive(&mut self, channel: NotifyChannel) -> io::Result<String> {
        let (prompt, hint) = match channel {
            NotifyChannel::Discord => (
                "Discord webhook URL",
                "Create one in Discord: Server Settings > Integrations > Webhooks",
            ),
            NotifyChannel::Telegram => (
                "Telegram chat ID",
                "Message @observer_ai_bot on Telegram to get your chat ID",
            ),
            NotifyChannel::Sms | NotifyChannel::WhatsApp | NotifyChannel::Call => (
                "Phone number (E.164 format)",
                "Example: +15551234567 - Must be whitelisted via Observer bot first",
            ),
            NotifyChannel::Email => (
                "Email address",
                "Enter the email address to receive notifications",
            ),
        };

        eprintln!();
        eprintln!("  {}", hint);
        eprintln!();

        let value: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt(prompt)
            .interact_text()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        if value.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Value cannot be empty",
            ));
        }

        let value = value.trim().to_string();

        // Validate Discord webhook URL
        if channel == NotifyChannel::Discord
            && !value.starts_with("https://discord.com/api/webhooks/")
            && !value.starts_with("https://discordapp.com/api/webhooks/")
        {
            eprintln!("  Warning: URL doesn't look like a Discord webhook, but proceeding anyway.");
        }

        // Save the value to the appropriate config field
        match channel {
            NotifyChannel::Discord => self.discord.webhook_url = Some(value.clone()),
            NotifyChannel::Telegram => self.telegram.chat_id = Some(value.clone()),
            NotifyChannel::Sms | NotifyChannel::WhatsApp | NotifyChannel::Call => {
                self.phone.number = Some(value.clone())
            }
            NotifyChannel::Email => self.email.address = Some(value.clone()),
        }

        // Set as default channel and save
        self.default_channel = Some(channel);
        self.save()?;

        eprintln!();
        eprintln!("  Saved to ~/.config/observe/config.toml");
        eprintln!();

        Ok(value)
    }

    /// Run the full first-run setup flow: select channel + enter secret
    pub fn run_first_time_setup(&mut self) -> io::Result<NotifyChannel> {
        let channel = self.select_channel_interactive()?;
        self.prompt_channel_secret_interactive(channel)?;
        Ok(channel)
    }

    /// Get the secret for a channel, or None if not configured
    pub fn get_channel_secret(&self, channel: NotifyChannel) -> Option<String> {
        match channel {
            NotifyChannel::Discord => self.discord.webhook_url.clone(),
            NotifyChannel::Telegram => self.telegram.chat_id.clone(),
            NotifyChannel::Sms | NotifyChannel::WhatsApp | NotifyChannel::Call => {
                self.phone.number.clone()
            }
            NotifyChannel::Email => self.email.address.clone(),
        }
    }

    /// Ensure we have the secret for a channel, prompting if needed
    pub fn ensure_channel_secret(&mut self, channel: NotifyChannel) -> io::Result<String> {
        if let Some(secret) = self.get_channel_secret(channel) {
            return Ok(secret);
        }
        self.prompt_channel_secret_interactive(channel)
    }

}
