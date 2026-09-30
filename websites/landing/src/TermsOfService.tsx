import { ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

const TermsOfService = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gray-900 text-white py-6">
        <div className="container mx-auto px-6">
          <Link to="/" className="flex items-center space-x-2 text-gray-300 hover:text-white transition w-fit">
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </Link>
        </div>
      </header>

      {/* Content */}
      <main className="py-16">
        <div className="container mx-auto px-6 max-w-4xl">
          <h1 className="text-4xl font-bold mb-4 text-black">Terms of Service</h1>
          <p className="text-black mb-8">Last updated: September 2026</p>

          <div className="space-y-6 text-gray-700">
            <p>
              Please read these Terms of Service ("Terms") carefully before using the Observer AI application
              and related services (the "Service") operated by Observer AI LLC ("Observer AI," "we," "us," or "our").
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">1. Acceptance of Terms</h2>
            <p>
              By accessing or using the Service, you agree to be bound by these Terms. If you disagree with
              any part of the Terms, you may not access the Service.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">2. Description of Service</h2>
            <p>
              Observer AI provides a platform for creating and running micro-agents that can observe your
              digital environment (screen, microphone, camera, clipboard), process information using AI models,
              and take actions such as sending notifications. The Service includes desktop applications, mobile
              applications, and web-based interfaces.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">3. Open Source License</h2>
            <p>
              Observer AI is open source software. You have an implied license to compile, modify, and run your
              own version of Observer from the <a href="https://github.com/Roy3838/Observer" className="text-blue-600 hover:underline">source code</a>.
              When using a self-compiled version with local inference providers and without our cloud services,
              no data passes through our servers and these Terms do not apply to such usage.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">4. User Accounts</h2>
            <p>
              To use certain features of the Service, you must create an account. You are responsible for:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Maintaining the confidentiality of your account credentials</li>
              <li>All activities that occur under your account</li>
              <li>Providing accurate and complete information</li>
              <li>Notifying us immediately of any unauthorized use</li>
            </ul>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">5. User Content and License Grant</h2>
            <p>
              You retain ownership of any agents, prompts, code, and other content you create ("User Content").
              By sharing User Content through community features, you grant Observer AI a worldwide, non-exclusive,
              royalty-free license to use, reproduce, modify, adapt, publish, translate, distribute, and display
              such content in connection with the Service.
            </p>
            <p>
              You represent and warrant that you have all rights necessary to grant this license and that your
              User Content does not infringe any third-party rights.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">6. Acceptable Use</h2>
            <p>You agree not to use the Service to:</p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Violate any applicable laws or regulations</li>
              <li>Infringe on the rights of others, including privacy and intellectual property rights</li>
              <li>Monitor or capture data from individuals without their consent</li>
              <li>Send spam, unsolicited messages, or harassing communications</li>
              <li>Distribute malware, viruses, or harmful code</li>
              <li>Attempt to gain unauthorized access to systems or data</li>
              <li>Interfere with or disrupt the Service or servers</li>
              <li>Use the Service for any illegal or unauthorized purpose</li>
              <li>Abuse notification services (SMS, WhatsApp, email, etc.) in ways that violate carrier policies</li>
            </ul>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">7. Third-Party Services</h2>
            <p>
              The Service integrates with third-party AI inference providers (including Google AI Studio,
              OpenRouter, and Fireworks.ai) and notification services (including Twilio, SendGrid, and
              Telegram), as well as analytics and monitoring services (including Datadog) that help us
              understand and improve the Service. Your use of these services is subject to their respective
              terms and privacy policies, and our use of analytics is described in our Privacy Policy.
            </p>
            <p>
              <strong>We are not responsible for the practices, policies, or actions of third-party service
              providers.</strong> This includes how AI inference providers may use, store, log, or process
              the data (including prompts, images, and other content) sent to them through our Service.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">8. Intellectual Property</h2>
            <p>
              The Service and its original content (excluding User Content), features, and functionality are
              owned by Observer AI and are protected by copyright, trademark, and other intellectual property laws.
              Our trademarks may not be used without our prior written consent.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">9. Payment and Subscriptions</h2>
            <p>
              Some features of the Service require a paid subscription. By subscribing, you agree to pay the
              applicable fees. Subscriptions automatically renew unless cancelled. Refunds are handled according
              to our refund policy and applicable app store policies.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">10. Cloud Monitoring Credits and Quotas</h2>
            <p>
              Cloud monitoring usage is measured in "credits." Each credit represents one agent loop execution
              (one cycle of data capture and AI inference processed through our cloud infrastructure).
            </p>
            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Credit Allocation by Tier</h3>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Free tier:</strong> 60 credits per day, up to 1,200 credits per month (about 10 hours at 30-second loop intervals)</li>
              <li><strong>Plus tier:</strong> 480 credits per day, up to 4,800 credits per month (about 4 hours per day and 40 hours per month at 30-second loop intervals)</li>
              <li><strong>Pro tier:</strong> 1,440 credits per day, up to 12,000 credits per month (about 100 hours at 30-second loop intervals)</li>
              <li><strong>Max tier:</strong> 2,880 credits per day with no monthly cap (effectively unlimited at 30-second loop intervals)</li>
            </ul>
            <p className="mt-2">
              Daily and monthly limits apply together: usage stops when either one is reached, whichever
              comes first.
            </p>
            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">How Credits Translate to Monitoring Time</h3>
            <p>
              The actual monitoring duration depends on your configured loop interval. For example, with 60 daily credits:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>30-second loop interval = 30 minutes of monitoring (60 loops)</li>
              <li>1-minute loop interval = 1 hour of monitoring (60 loops)</li>
              <li>3-minute loop interval = 3 hours of monitoring (60 loops)</li>
              <li>30-minute loop interval = 30 hours of monitoring (exceeds 24 hours, so effectively all day)</li>
            </ul>
            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Credit Usage Rules</h3>
            <ul className="list-disc pl-6 space-y-1">
              <li>Daily limits reset at midnight UTC</li>
              <li>Monthly limits reset on the first day of each calendar month (UTC), not on your billing date</li>
              <li>Unused credits do not roll over to the next day or month</li>
              <li>Each running agent consumes credits independently</li>
              <li>Local inference (using your own models) does not consume cloud credits</li>
              <li>We reserve the right to modify credit allocations with reasonable notice</li>
            </ul>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Service Limits (Anti-Abuse Measures)</h3>
            <p>
              To ensure fair usage and prevent abuse, all tiers have daily limits on notifications and other
              services, and some services also have monthly limits. Daily limits reset at midnight UTC; monthly
              limits reset on the first day of each calendar month (UTC). In addition, metered requests are
              limited to 30 per minute per account across all services.
            </p>

            <h4 className="font-medium mt-4 mb-2 text-gray-800">Free Tier Limits</h4>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Cloud Monitoring:</strong> 60 credits per day, 1,200 per month</li>
              <li><strong>Agent Builder (MCP) messages:</strong> 45 per day, 150 per month</li>
              <li><strong>SMS, WhatsApp, Voice Calls:</strong> 5 each per day, within a shared $0.50 monthly notification budget (see below)</li>
              <li><strong>Email, Telegram, Pushover:</strong> 2,880 each per day</li>
            </ul>

            <h4 className="font-medium mt-4 mb-2 text-gray-800">Plus Tier Limits</h4>
            <p>
              Same as the Free tier, except Cloud Monitoring: 480 credits per day and 4,800 per month (about
              4 hours per day and 40 hours per month at 30-second loop intervals).
            </p>

            <h4 className="font-medium mt-4 mb-2 text-gray-800">Pro Tier Limits</h4>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Cloud Monitoring:</strong> 1,440 credits per day, 12,000 per month</li>
              <li><strong>Agent Builder (MCP) messages:</strong> 1,000 per day, 1,500 per month</li>
              <li><strong>SMS, WhatsApp, Voice Calls:</strong> 300 each per day, within a shared $5.00 monthly notification budget (see below)</li>
              <li><strong>Email, Telegram, Pushover:</strong> 2,880 each per day</li>
            </ul>

            <h4 className="font-medium mt-4 mb-2 text-gray-800">Max Tier Limits</h4>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Cloud Monitoring:</strong> 2,880 credits per day, no monthly cap (effectively unlimited at 30-second intervals)</li>
              <li><strong>Agent Builder (MCP) messages:</strong> 1,000 per day, 3,000 per month</li>
              <li><strong>SMS, WhatsApp, Voice Calls:</strong> 300 each per day, within a shared $5.00 monthly notification budget (see below)</li>
              <li><strong>Email, Telegram, Pushover:</strong> 2,880 each per day</li>
            </ul>

            <h4 className="font-medium mt-4 mb-2 text-gray-800">Organization Plans</h4>
            <p>
              Limits for organization and enterprise plans are set by the agreement for that plan, which
              takes precedence over the tier limits above where they differ.
            </p>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">SMS, WhatsApp and Voice Call Budget</h3>
            <p>
              SMS, WhatsApp messages and voice calls cost us real money for every message and every minute,
              and that cost varies widely by country. Instead of a fixed number of messages, these three
              channels share one monthly dollar budget per account:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Free tier:</strong> $0.50 per month</li>
              <li><strong>Plus tier:</strong> $0.50 per month</li>
              <li><strong>Pro tier:</strong> $5.00 per month</li>
              <li><strong>Max tier:</strong> $5.00 per month</li>
            </ul>
            <p className="mt-2">
              Each message or call draws down the budget by the price our messaging provider (Twilio)
              charges us for it. We do not add a markup. That price depends on the destination country, the
              message length and, for calls, the duration.
            </p>
            <p className="mt-2">
              Because the provider reports its price only after a message is sent or a call ends, each send
              first reserves an estimated amount from your budget. The reservation is replaced with the actual
              price once the provider reports it, usually within a few minutes. While reservations are pending,
              a send may be refused shortly before your budget is fully used.
            </p>
            <ul className="list-disc pl-6 space-y-1 mt-2">
              <li>SMS messages are shortened to at most 3 segments (about 459 plain characters, or about 201 characters if the message contains emoji or other special characters)</li>
              <li>Videos sent by SMS are delivered as links</li>
              <li>The budget resets on the first day of each calendar month (UTC) and unused budget does not roll over</li>
            </ul>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Connecting a Phone for Notifications</h3>
            <ul className="list-disc pl-6 space-y-1">
              <li>SMS, WhatsApp and voice notifications are only sent to a phone you have connected by sending your 4-word Observer code to the Observer bot on WhatsApp. Notifications cannot be sent to arbitrary phone numbers.</li>
              <li>By connecting a phone, you confirm that it is your phone (or that you have its owner's permission) and consent to receive notifications from your agents on it.</li>
              <li>WhatsApp only allows us to message you within 24 hours of your last message to the Observer bot. After that, send the bot any message to resume WhatsApp notifications.</li>
            </ul>

            <p className="mt-4 text-sm text-gray-600">
              <strong>Note:</strong> SMS, WhatsApp, and Voice Call limits are capped due to carrier costs and anti-spam regulations.
              If you require higher limits for legitimate use cases, please contact us at{' '}
              <a href="mailto:help@observer-ai.com" className="text-blue-600 hover:underline">help@observer-ai.com</a>.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">11. Disclaimer of Warranties</h2>
            <p>
              THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
              OR IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
              PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
            </p>
            <p>
              We do not warrant that the Service will be uninterrupted, secure, or error-free, that defects will
              be corrected, or that the AI-generated responses will be accurate, complete, or reliable.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">12. Limitation of Liability</h2>
            <p>
              TO THE MAXIMUM EXTENT PERMITTED BY LAW, OBSERVER AI SHALL NOT BE LIABLE FOR ANY INDIRECT,
              INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, OR ANY LOSS OF PROFITS OR REVENUES,
              WHETHER INCURRED DIRECTLY OR INDIRECTLY, OR ANY LOSS OF DATA, USE, GOODWILL, OR OTHER INTANGIBLE
              LOSSES RESULTING FROM:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Your use or inability to use the Service</li>
              <li>Any unauthorized access to or use of our servers and/or any personal information stored therein</li>
              <li>Any interruption or cessation of transmission to or from the Service</li>
              <li>Any bugs, viruses, or other harmful code that may be transmitted through the Service</li>
              <li>Any errors, inaccuracies, or omissions in AI-generated content</li>
              <li>Actions taken by third-party service providers</li>
            </ul>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">13. Indemnification</h2>
            <p>
              You agree to indemnify, defend, and hold harmless Observer AI and its officers, directors,
              employees, and agents from any claims, damages, losses, liabilities, and expenses (including
              legal fees) arising out of your use of the Service, violation of these Terms, or infringement
              of any third-party rights.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">14. Termination</h2>
            <p>
              We may terminate or suspend your account and access to the Service immediately, without prior
              notice, for conduct that we believe violates these Terms or is harmful to other users, us, or
              third parties, or for any other reason at our sole discretion.
            </p>
            <p>
              You may delete your account at any time through the application settings.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">15. Age Requirement</h2>
            <p>
              You must be at least 13 years old to use the Service. By using the Service, you represent that
              you are at least 13 years of age.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">16. Governing Law</h2>
            <p>
              These Terms shall be governed by and construed in accordance with the laws of the United States,
              without regard to conflict of law principles.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">17. Changes to Terms</h2>
            <p>
              We reserve the right to modify these Terms at any time. We will provide notice of significant
              changes by posting the new Terms on this page and updating the "Last updated" date. Your continued
              use of the Service after changes constitutes acceptance of the modified Terms.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">18. Contact Us</h2>
            <p>
              If you have questions about these Terms, please contact us at:{' '}
              <a href="mailto:help@observer-ai.com" className="text-blue-600 hover:underline">help@observer-ai.com</a>
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="container mx-auto px-6 text-center text-gray-400">
          <p>&copy; 2026 Observer AI. Open source and community driven.</p>
        </div>
      </footer>
    </div>
  );
};

export default TermsOfService;
