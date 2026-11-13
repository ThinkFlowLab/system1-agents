# Security

Report a vulnerability through GitHub's private advisory form for this repository
(Security tab, "Report a vulnerability"). Include the agent or command involved,
the input that triggers it and what an attacker gains.

What this repository does with untrusted input: the browser agents read live web pages and hand their text to
Jev and to the chat model. The `injection_guard` rail is the one defence in the tree against instructions planted
in such pages; [its runtime policy](docs/agents.md#injection-guard-runtime-policy) says what it inspects and what
passes on each verdict. The reports of most interest: a bypass of that rail, an agent leaking the keys in `.env`, or the
MCP server running anything beyond its three declared tools.
