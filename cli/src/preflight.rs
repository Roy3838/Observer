use crate::auth::AuthTokens;
use crate::config::NotifyChannel;
use serde::Deserialize;
use std::io::{self, Write};
use std::time::Duration;

const API_BASE: &str = "https://api.observer-ai.com";

#[derive(Debug, Deserialize)]
struct WhitelistResponse {
    is_whitelisted: bool,
}

/// Check if a phone number is whitelisted for the given channel
pub async fn check_whitelist(
    phone_number: &str,
    channel: NotifyChannel,
    access_token: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    let channel_param = match channel {
        NotifyChannel::WhatsApp => Some("whatsapp"),
        NotifyChannel::Sms => Some("sms"),
        NotifyChannel::Call => Some("voice"),
        _ => None,
    };

    let mut body = serde_json::json!({
        "phone_number": phone_number
    });

    if let Some(ch) = channel_param {
        body["channel"] = serde_json::json!(ch);
    }

    let response = client
        .post(format!("{}/tools/is-whitelisted", API_BASE))
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", access_token))
        .json(&body)
        .send()
        .await?;

    if !response.status().is_success() {
        return Ok(false);
    }

    let data: WhitelistResponse = response.json().await?;
    Ok(data.is_whitelisted)
}

/// Wait for a phone number to be whitelisted, polling every second
/// Shows instructions and a spinner while waiting
pub async fn wait_for_whitelist(
    phone_number: &str,
    channel: NotifyChannel,
) -> io::Result<()> {
    // Get auth token first
    let mut tokens = match AuthTokens::load() {
        Some(t) => t,
        None => {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Not logged in. Run 'observe login' first.",
            ));
        }
    };

    let access_token = match tokens.get_valid_token().await {
        Ok(t) => t,
        Err(e) => {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Auth error: {}", e),
            ));
        }
    };

    // Check if already whitelisted
    match check_whitelist(phone_number, channel, &access_token).await {
        Ok(true) => return Ok(()),
        Ok(false) => {}
        Err(e) => {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to check whitelist: {}", e),
            ));
        }
    }

    // Not whitelisted - show instructions and poll
    let channel_name = match channel {
        NotifyChannel::Sms => "SMS",
        NotifyChannel::WhatsApp => "WhatsApp",
        NotifyChannel::Call => "voice calls",
        _ => "notifications",
    };

    let bot_instructions = match channel {
        NotifyChannel::WhatsApp => {
            "  Text +1 (555) 783-4727 on WhatsApp to whitelist your number"
        }
        _ => {
            "  Text or call +1 (863) 208-5341 to whitelist your number"
        }
    };

    eprintln!();
    eprintln!("  Phone number {} is not whitelisted for {}.", phone_number, channel_name);
    eprintln!();
    eprintln!("  To whitelist your number:");
    eprintln!("{}", bot_instructions);
    eprintln!();
    eprint!("  Waiting for whitelist confirmation ");
    io::stderr().flush()?;

    let spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let mut spinner_idx = 0;

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Update spinner
        eprint!("\x1b[1D{}", spinner_chars[spinner_idx]);
        io::stderr().flush()?;
        spinner_idx = (spinner_idx + 1) % spinner_chars.len();

        // Re-check whitelist status
        match check_whitelist(phone_number, channel, &access_token).await {
            Ok(true) => {
                eprintln!("\x1b[1D✓");
                eprintln!();
                eprintln!("  Phone number whitelisted!");
                eprintln!();
                return Ok(());
            }
            Ok(false) => {
                // Still not whitelisted, keep polling
                continue;
            }
            Err(_) => {
                // Network error, keep trying
                continue;
            }
        }
    }
}
