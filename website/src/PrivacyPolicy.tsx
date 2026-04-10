import { ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

const PrivacyPolicy = () => {
  return (
    <div className="min-h-screen bg-white">
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
          <h1 className="text-4xl font-bold mb-4 text-black">Privacy Policy</h1>
          <p className="text-black mb-8">Last updated: January 2026</p>

          <div className="space-y-6 text-gray-700">
            {/* Privacy-First Option */}
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-6 mb-8">
              <h2 className="text-xl font-semibold mb-4 text-gray-900">Want Maximum Privacy? You Can Have It.</h2>
              <p className="mb-4">
                Observer AI is <strong>100% open source</strong>. If you care deeply about your privacy, you have an
                implied license to compile and run your own version of Observer. Here's how to use Observer with
                <strong> zero data going to our servers</strong>:
              </p>
              <ol className="list-decimal pl-6 space-y-2 mb-4">
                <li><strong>Compile Observer from source</strong> — Clone the <a href="https://github.com/Roy3838/Observer" className="text-blue-600 hover:underline">GitHub repo</a> and build it yourself. (Use Tor if you don't even want GitHub to know you exist.)</li>
                <li><strong>Set up a local inference server</strong> — Use <a href="https://ollama.ai" className="text-blue-600 hover:underline">Ollama</a>, llama.cpp, vLLM, or any local model server.</li>
                <li><strong>Use Discord notifications</strong> — Discord webhooks go directly from your client to Discord's servers, completely bypassing our API.</li>
              </ol>
              <p className="text-gray-600 italic">
                If you do this, we have literally no way of knowing you exist. Your data stays 100% on your machine.
              </p>
            </div>

            <p className="text-lg font-medium text-gray-800">
              If you choose to use our convenient cloud inference and other notification services, the following policy applies:
            </p>

            <hr className="my-6 border-gray-200" />

            <p>
              Observer AI LLC ("Observer AI," "we," "us," or "our") is committed to protecting your privacy.
              This Privacy Policy explains how we collect, use, disclose, and safeguard your information when
              you use the Observer AI application and related services (the "Service").
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">1. Information We Collect</h2>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Account Information</h3>
            <p>When you create an account, we collect:</p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Email address</li>
              <li>Name</li>
              <li>Authentication identifiers (via Auth0)</li>
              <li>Payment information (via Stripe) if you subscribe to paid features</li>
            </ul>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Sensor and Content Data</h3>
            <p>
              Observer AI allows you to use various sensors (screen capture, microphone, camera, clipboard).
              When you use features that require server processing, this data may be temporarily cached on our
              servers for up to 24 hours to facilitate the service. This includes:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Screenshots and screen content</li>
              <li>Audio transcriptions</li>
              <li>Agent memory and context</li>
              <li>Notification content (messages, images, videos)</li>
            </ul>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Usage Data</h3>
            <p>
              We maintain ephemeral logs of API requests to monitor service health and ensure proper usage.
              These logs are not stored in a persistent database and are used solely for operational purposes.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">2. How We Use Your Information</h2>
            <p>We use the information we collect to:</p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Provide, maintain, and improve the Service</li>
              <li>Process AI inference requests through our supported providers</li>
              <li>Send notifications via your configured channels (SMS, email, WhatsApp, Telegram, etc.)</li>
              <li>Process payments and manage subscriptions</li>
              <li>Respond to your inquiries and provide support</li>
              <li>Monitor and analyze usage patterns to improve the Service</li>
            </ul>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">3. Third-Party Service Providers</h2>
            <p>
              To provide the Service, we share data with the following categories of third-party providers:
            </p>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">AI Inference Providers</h3>
            <p className="mb-3">
              Your prompts and sensor data (such as screenshots) are sent to AI inference providers to generate responses.
              We select providers that maintain industry-standard data protection practices. The specific providers we use are:
            </p>
            <ul className="list-disc pl-6 space-y-2 mb-3">
              <li>
                <strong>Google AI Studio (Gemini API)</strong> — We use the paid API tier, which does not use your data for model training.{' '}
                <a href="https://ai.google.dev/gemini-api/terms" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Privacy Policy</a>
              </li>
              <li>
                <strong>Fireworks.ai</strong> — Explicitly commits to zero data retention for prompts and does not use your data for training without opt-in.{' '}
                <a href="https://fireworks.ai/privacy-policy" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Privacy Policy</a>
              </li>
              <li>
                <strong>OpenRouter</strong> — Routes requests to various AI providers. Data handling depends on the underlying model provider.{' '}
                <a href="https://openrouter.ai/privacy" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Privacy Policy</a>
              </li>
            </ul>
            <p className="text-sm text-gray-600">
              <strong>For maximum privacy:</strong> Use local models (via Ollama or similar) — your data never leaves your device.
              The app clearly indicates whether each model is cloud-based or local, and links to the relevant provider's privacy policy.
            </p>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Notification Providers</h3>
            <p>
              When you use notification features, your message content and media are transmitted through:
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Twilio (SMS, WhatsApp, phone calls)</li>
              <li>SendGrid (email)</li>
              <li>Telegram Bot API</li>
              <li>Discord (direct from client to Discord servers)</li>
            </ul>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Other Providers</h3>
            <ul className="list-disc pl-6 space-y-1">
              <li>Auth0 (authentication)</li>
              <li>Stripe (payment processing)</li>
            </ul>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">4. Data Retention</h2>
            <ul className="list-disc pl-6 space-y-1">
              <li><strong>Temporarily cached data</strong> (screenshots, audio, notification content): Up to 24 hours</li>
              <li><strong>Account information</strong>: Retained until you delete your account</li>
              <li><strong>Payment records</strong>: Retained as required by law and for accounting purposes</li>
            </ul>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">5. Data Security</h2>
            <p>
              We implement appropriate technical and organizational measures to protect your personal information.
              However, no method of transmission over the Internet or electronic storage is 100% secure, and we
              cannot guarantee absolute security.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">6. Your Rights and Choices</h2>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">Account Deletion</h3>
            <p>
              You can delete your account at any time through the application settings. Upon deletion, we will
              remove your personal information from our systems, except as required by law.
            </p>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">For EU Residents (GDPR)</h3>
            <p>You have the right to:</p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Access your personal data</li>
              <li>Rectify inaccurate personal data</li>
              <li>Request erasure of your personal data</li>
              <li>Restrict or object to processing</li>
              <li>Data portability</li>
              <li>Lodge a complaint with a supervisory authority</li>
            </ul>

            <h3 className="text-lg font-medium mt-6 mb-2 text-gray-800">For California Residents (CCPA)</h3>
            <p>You have the right to:</p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Know what personal information we collect</li>
              <li>Request deletion of your personal information</li>
              <li>Opt-out of the sale of personal information (we do not sell your data)</li>
              <li>Non-discrimination for exercising your rights</li>
            </ul>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">7. Data Sales</h2>
            <p>
              <strong>We do not sell your personal information.</strong> We only share data with third-party
              providers as necessary to deliver the Service.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">8. Children's Privacy</h2>
            <p>
              The Service is not intended for children under 13 years of age. We do not knowingly collect
              personal information from children under 13. If you are a parent or guardian and believe your
              child has provided us with personal information, please contact us.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">9. International Data Transfers</h2>
            <p>
              Your information may be transferred to and processed in countries other than your country of
              residence. These countries may have different data protection laws. By using the Service, you
              consent to such transfers.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">10. Changes to This Policy</h2>
            <p>
              We may update this Privacy Policy from time to time. We will notify you of any changes by posting
              the new Privacy Policy on this page and updating the "Last updated" date.
            </p>

            <h2 className="text-xl font-semibold mt-8 mb-4 text-gray-900">11. Contact Us</h2>
            <p>
              If you have questions about this Privacy Policy or our data practices, please contact us at:{' '}
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

export default PrivacyPolicy;
