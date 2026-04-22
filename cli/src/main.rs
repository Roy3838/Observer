mod auth;
mod config;
mod notify;
mod preflight;
mod runner;

use clap::{Parser, Subcommand};
use config::{Config, NotifyChannel};

#[derive(Parser)]
#[command(name = "observe")]
#[command(about = "Wrap commands and get notified when they complete")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Configure Discord webhook and use it for this notification
    #[arg(long)]
    discord: bool,

    /// Configure Telegram and use it for this notification (requires login)
    #[arg(long)]
    telegram: bool,

    /// Configure SMS and use it for this notification (requires login)
    #[arg(long)]
    sms: bool,

    /// Configure WhatsApp and use it for this notification (requires login)
    #[arg(long)]
    whatsapp: bool,

    /// Configure voice call and use it for this notification (requires login)
    #[arg(long)]
    call: bool,

    /// Configure email and use it for this notification (requires login)
    #[arg(long)]
    email: bool,

    /// The command to run (and its arguments)
    #[arg(trailing_var_arg = true)]
    args: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Login to Observer AI (required for Telegram, SMS, WhatsApp, Call, Email)
    Login,
    /// Logout from Observer AI
    Logout,
    /// Show current auth status
    Whoami,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Handle subcommands
    if let Some(cmd) = cli.command {
        match cmd {
            Commands::Login => {
                if let Err(e) = auth::login().await {
                    eprintln!("Login failed: {}", e);
                    std::process::exit(1);
                }
                return;
            }
            Commands::Logout => {
                if let Err(e) = auth::logout() {
                    eprintln!("Logout failed: {}", e);
                    std::process::exit(1);
                }
                return;
            }
            Commands::Whoami => {
                auth::whoami();
                return;
            }
        }
    }

    // No subcommand - run a command and notify
    if cli.args.is_empty() {
        eprintln!("Usage: observe [OPTIONS] <COMMAND>...");
        eprintln!("       observe login");
        eprintln!("       observe logout");
        eprintln!("       observe whoami");
        eprintln!();
        eprintln!("Options:");
        eprintln!("       --discord     Configure Discord webhook");
        eprintln!("       --telegram    Configure Telegram (requires login)");
        eprintln!("       --sms         Configure SMS (requires login)");
        eprintln!("       --whatsapp    Configure WhatsApp (requires login)");
        eprintln!("       --call        Configure voice call (requires login)");
        eprintln!("       --email       Configure email (requires login)");
        eprintln!();
        eprintln!("On first run, you'll be prompted to select a notification method.");
        eprintln!("Use --<method> flags to reconfigure a specific method.");
        eprintln!();
        eprintln!("Run 'observe --help' for more information.");
        std::process::exit(1);
    }

    // Load config
    let mut config = Config::load();

    // Determine notification channel and whether to reconfigure
    let (channel, reconfigure) = if cli.discord {
        (NotifyChannel::Discord, true)
    } else if cli.telegram {
        (NotifyChannel::Telegram, true)
    } else if cli.sms {
        (NotifyChannel::Sms, true)
    } else if cli.whatsapp {
        (NotifyChannel::WhatsApp, true)
    } else if cli.call {
        (NotifyChannel::Call, true)
    } else if cli.email {
        (NotifyChannel::Email, true)
    } else if config.is_first_run() {
        // First run: show interactive TUI
        match config.run_first_time_setup() {
            Ok(ch) => (ch, false), // Already configured in setup
            Err(e) => {
                eprintln!("Setup error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Use saved default
        (config.default_channel.unwrap(), false)
    };

    // If reconfiguring (flag was passed), prompt for the new secret
    if reconfigure {
        if let Err(e) = config.prompt_channel_secret_interactive(channel) {
            eprintln!("Configuration error: {}", e);
            std::process::exit(1);
        }
    }

    // For channels that require login, verify auth BEFORE running the command
    let access_token = if channel.requires_login() {
        let mut tokens = match auth::AuthTokens::load() {
            Some(t) => t,
            None => {
                eprintln!("Not logged in. Run 'observe login' first.");
                std::process::exit(1);
            }
        };

        match tokens.get_valid_token().await {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("Auth error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Get the channel secret BEFORE running the command
    let secret = match config.ensure_channel_secret(channel) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Configuration error: {}", e);
            std::process::exit(1);
        }
    };

    // For phone-based channels, verify the number is whitelisted
    if channel.is_phone_based() {
        if let Err(e) = preflight::wait_for_whitelist(&secret, channel).await {
            eprintln!("Whitelist error: {}", e);
            std::process::exit(1);
        }
    }

    // Format command for display
    let command_str = cli.args.join(" ");

    // Now run the command
    let result = match runner::run_command(&cli.args) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to run command: {}", e);
            std::process::exit(1);
        }
    };

    // Send notification based on channel
    let notify_result = match channel {
        NotifyChannel::Discord => {
            notify::send_discord_notification(&secret, &command_str, &result).await
        }
        NotifyChannel::Telegram => {
            notify::send_telegram_notification(
                &secret,
                &command_str,
                &result,
                access_token.as_ref().unwrap(),
            )
            .await
        }
        NotifyChannel::Sms => {
            notify::send_sms_notification(
                &secret,
                &command_str,
                &result,
                access_token.as_ref().unwrap(),
            )
            .await
        }
        NotifyChannel::WhatsApp => {
            notify::send_whatsapp_notification(
                &secret,
                &command_str,
                &result,
                access_token.as_ref().unwrap(),
            )
            .await
        }
        NotifyChannel::Call => {
            notify::send_call_notification(
                &secret,
                &command_str,
                &result,
                access_token.as_ref().unwrap(),
            )
            .await
        }
        NotifyChannel::Email => {
            notify::send_email_notification(
                &secret,
                &command_str,
                &result,
                access_token.as_ref().unwrap(),
            )
            .await
        }
    };

    if let Err(e) = notify_result {
        eprintln!("Failed to send notification: {}", e);
    }

    // Exit with the same code as the wrapped command
    std::process::exit(result.exit_code.unwrap_or(1));
}
