# Domain Knowledge and Architecture Decisions

## Context Map (Multi-Context)

This project uses a multi-context domain documentation layout. This means that while we maintain a central \`GLOSSARY-MAP.md\`, specific domains should keep their glossary entries within their own local context for better organization.

### Structure:
1.  **Root Level**: \`docs/agents/\` contains this map and general rules.
2.  **Domain Glossaries**: Specific crates or components (e.g., \`crates/abd-clam/\`, \`crates/distances/\`) should maintain their own dedicated glossary documentation (\`GLOSSARY.md\`). These files will be referenced by the main \`GLOSSARY-MAP.md\`.

### Rules:
*   **Central Map**: Always reference a domain's glossary via \`[Domain Name] in [Path to Glossary]\` within the appropriate local context or the central map.
*   **ADRs**: Architecture Decision Records (ADRs) should be stored in their respective component directory's \`docs/adr/\` folder, not at the root level.

This structure ensures that domain knowledge is both centralized for quick reference and decentralized for ownership clarity.