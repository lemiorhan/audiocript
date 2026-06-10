# Security Policy

## Supported versions

This project is actively developed on the `main` branch. Security fixes are
applied to `main`.

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report them privately using GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
("Report a vulnerability" under the repository's **Security** tab), or contact the
maintainers directly.

When reporting, please include:

- A description of the issue and its potential impact
- Steps to reproduce
- Affected version / commit and your environment (macOS and Python versions)

We will acknowledge your report as soon as possible and keep you informed of the
progress toward a fix.

## Scope notes

This app runs locally and processes audio on your machine. It requires macOS
**Microphone** and **System Audio Recording** permissions to capture audio, and
it downloads transcription models from the Hugging Face Hub on first use. No
recorded audio or transcripts are sent anywhere by the application.
