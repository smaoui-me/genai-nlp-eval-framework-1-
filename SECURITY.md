# Security policy

## Supported scope

The current Docker distribution is a single-instance application for local
evaluation and controlled pilots. It does not provide authentication,
authorization, tenant isolation, encrypted storage, or centralized audit
retention.

Do not expose it directly to the public internet or use it with confidential
data without adding the controls required by your organization.

## Secrets and data

- Supply credentials only through `.env`, Streamlit secrets, or an approved
  runtime secret manager.
- Never commit API keys, reviewed exports, or private datasets.
- Verify the retention, geographic processing, and training policies of an LLM
  endpoint before sending organizational data.
- Prefer an approved private endpoint or local model for sensitive material.

## Reporting a vulnerability

Use GitHub's private security advisory feature for the repository. Do not place
credentials, private data, or exploit details in a public issue.
