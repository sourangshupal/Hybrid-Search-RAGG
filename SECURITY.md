# Security Policy

## Supported Versions

We actively support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security vulnerabilities by emailing:

**security@mongodb.com**

Or use GitHub's private vulnerability reporting:

1. Go to the [Security tab](../../security) of this repository
2. Click "Report a vulnerability"
3. Fill out the form with details

### What to Include

Please include as much of the following information as possible:

- **Type of vulnerability** (e.g., injection, authentication bypass, data exposure)
- **Location** - Full path to the affected source file(s)
- **Configuration** - Any special configuration required to reproduce
- **Steps to reproduce** - Step-by-step instructions
- **Proof of concept** - Code or commands that demonstrate the issue
- **Impact** - What an attacker could achieve

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days (depending on severity)

### What to Expect

1. **Acknowledgment** - We'll confirm receipt of your report
2. **Assessment** - We'll evaluate the vulnerability and determine severity
3. **Updates** - We'll keep you informed of our progress
4. **Fix** - We'll develop and test a fix
5. **Disclosure** - We'll coordinate disclosure timing with you
6. **Credit** - We'll credit you in the release notes (unless you prefer anonymity)

## Security Best Practices

When using HybridRAG, follow these security practices:

### Environment Variables

```bash
# NEVER commit .env files
# Use .env.example as a template

# Required secrets (keep these secure)
MONGODB_URI=mongodb+srv://...
VOYAGE_API_KEY=pa-...
ANTHROPIC_API_KEY=sk-ant-...
```

### MongoDB Atlas Security

1. **IP Whitelist** - Only allow known IP addresses
2. **VPC Peering** - Use private networking for production
3. **Database Users** - Create users with minimal required permissions
4. **Encryption** - Enable encryption at rest and in transit (default in Atlas)

### API Key Management

```python
# DO: Use environment variables
import os
api_key = os.environ.get("VOYAGE_API_KEY")

# DON'T: Hardcode keys
api_key = "pa-xxxxx"  # NEVER do this
```

### Input Validation

HybridRAG includes input validation, but always:

1. **Sanitize user input** before passing to queries
2. **Validate session IDs** match expected format
3. **Limit query length** to prevent abuse
4. **Rate limit** API endpoints

### Production Checklist

- [ ] All API keys stored in secret manager (not environment variables)
- [ ] MongoDB user has minimal required permissions
- [ ] IP whitelist configured in Atlas
- [ ] TLS 1.2+ enforced
- [ ] Rate limiting enabled on API endpoints
- [ ] Input validation on all user inputs
- [ ] Logging excludes sensitive data
- [ ] Regular dependency updates scheduled

## Known Security Considerations

### NoSQL Injection

HybridRAG uses parameterized queries and the filter builder pattern to prevent NoSQL injection. The filter builders validate input and prevent dangerous operators:

```python
# Safe: Using filter builders
from hybridrag import VectorSearchFilterConfig

config = VectorSearchFilterConfig(
    equality_filters={"category": user_input}  # Sanitized
)

# The filter builder prevents injection attempts
```

### Embedding Storage

Vector embeddings can potentially be used to reconstruct original text. Consider:

- Access controls on the chunks collection
- Encryption at rest (enabled by default in Atlas)
- Data retention policies

### Conversation Memory

Session data stored in MongoDB includes conversation history:

- Implement TTL indexes for automatic cleanup
- Consider PII detection and masking
- Encrypt sensitive conversations

## Dependencies

We regularly audit dependencies for vulnerabilities:

```bash
# Check for vulnerable packages
pip audit

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Security Updates

Security updates are released as patch versions (e.g., 0.3.1) and announced through:

- GitHub Security Advisories
- Release notes
- MongoDB Developer Center

Subscribe to repository notifications to stay informed.

---

Thank you for helping keep HybridRAG and its users secure!
