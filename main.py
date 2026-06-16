cat > main.py << 'EOF'
"""
App — AgentCore Multi-Gateway + MCP + Bedrock Knowledge Base (RAG)
Auto Insurance Company: Apex Auto Insurance

Gateways:
  - GATEWAY_OPS_URL : existing gateway (policy, claims, quotes)
  - GATEWAY_RAG_URL : new App gateway    (knowledge base Q&A)

Key pattern: Agent(tools=[mcp_client_ops, mcp_client_rag])
Strands merges all tools from both gateways into one tool list for the agent.

FIXES APPLIED:
  1. make_mcp_client: get_signed_headers() moved inside the lambda so a fresh
     SigV4 signature is generated on every connection (signatures expire in 15 min).
  2. get_agent: probe clients (with-block) are separate from the live clients
     passed to Agent, so the Agent manages its own connection lifecycle and never
     receives an already-closed MCPClient.
  3. System prompt: strengthened tool selection rules so the agent always calls
     w16-query-knowledge-base for ANY insurance-related question instead of
     answering from memory.
"""

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
log = app.logger

# -- Gateway URLs -------------------------------------------------------------
# Gateway 1: existing -- policy lookup, file claim, get quote
GATEWAY_OPS_URL = "https://w10-gateway-izji4rugvi.gateway.bedrock-agentcore.us-east-2.amazonaws.com/mcp"

# Gateway 2: knowledge base RAG Q&A
GATEWAY_RAG_URL = "https://w16-apex-rag-gateway-nmb20gwtzc.gateway.bedrock-agentcore.us-east-2.amazonaws.com/mcp"

REGION = "us-east-2"

_agent = None


def get_signed_headers(url: str) -> dict:
    """
    Generate SigV4 signed headers for a specific Gateway URL.
    Each gateway needs its own signed headers because the URL is
    part of the SigV4 signature -- one set of headers cannot cover two URLs.
    """
    session     = boto3.session.Session()
    credentials = session.get_credentials().get_frozen_credentials()
    request     = AWSRequest(method="POST", url=url, data=b"")
    SigV4Auth(credentials, "bedrock-agentcore", REGION).add_auth(request)
    return dict(request.headers)


def make_mcp_client(url: str) -> MCPClient:
    """
    Create an MCPClient for a given gateway URL.

    FIX: get_signed_headers(url) is called INSIDE the lambda, not outside.
    This ensures a fresh SigV4 signature is generated every time the MCPClient
    opens a new connection, preventing 403 errors after the 15-minute expiry.
    """
    return MCPClient(lambda: streamablehttp_client(
        url,
        headers=get_signed_headers(url)   # fresh signature on every connection
    ))


def get_agent():
    global _agent
    if _agent is not None:
        return _agent

    # -- Probe clients -- used only for startup tool-list logging -------------
    # FIX: these are separate throw-away clients used inside `with` blocks.
    # The `with` block closes the connection when the block exits, which is
    # correct here because we only need them for logging, not for the Agent.
    mcp_ops_probe = make_mcp_client(GATEWAY_OPS_URL)
    mcp_rag_probe = make_mcp_client(GATEWAY_RAG_URL)

    with mcp_ops_probe:
        ops_tools = mcp_ops_probe.list_tools_sync()
        log.info(f"Gateway OPS tools ({len(ops_tools)}): {[t.tool_name for t in ops_tools]}")

    with mcp_rag_probe:
        rag_tools = mcp_rag_probe.list_tools_sync()
        log.info(f"Gateway RAG tools ({len(rag_tools)}): {[t.tool_name for t in rag_tools]}")

    # -- Live clients -- passed to the Agent; it manages their lifecycle ------
    # FIX: fresh clients created AFTER the probe `with` blocks have closed.
    # The Agent receives open (or lazily-opened) connections, not closed ones.
    mcp_ops = make_mcp_client(GATEWAY_OPS_URL)
    mcp_rag = make_mcp_client(GATEWAY_RAG_URL)

    # -- Agent receives BOTH clients -- Strands merges all tools --------------
    _agent = Agent(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tools=[mcp_ops, mcp_rag],
        system_prompt="""You are Apex, a helpful AI assistant for Apex Auto Insurance Company.

You have access to four tools across two gateways:

GATEWAY 1 — Operations (policy, claims, quotes):

1. w10-get-policy(policy_id)
   - Looks up an existing policy by ID (e.g. POL-001)
   - Returns: customer name, vehicle, coverage, premium, deductible, status, expiry

2. w10-file-claim(policy_id, description, damage_type)
   - Files a new insurance claim against an existing policy
   - damage_type options: collision, theft, weather, vandalism, general

3. w10-get-quote(vehicle, coverage, driver_age, customer_name)
   - Generates a real-time insurance quote
   - coverage options: liability, collision, comprehensive

GATEWAY 2 — Knowledge Base (RAG-powered Q&A):

4. w16-query-knowledge-base(question, num_results)
   - Searches the Apex knowledge base for ALL insurance-related questions
   - MUST be used for: coverage types, exclusions, claims process, premiums,
     discounts, promotions, special offers, eligibility, surcharges, Ohio
     requirements, rentals, and ANY question not handled by tools 1-3
   - Returns a grounded answer with source citations from official Apex documents

CRITICAL TOOL SELECTION RULES — follow these exactly, in order:

1. Customer provides a policy ID (POL-XXX) and wants details → w10-get-policy
2. Customer wants to report an accident or damage            → w10-file-claim
3. Customer wants a price estimate for new coverage          → w10-get-quote
4. EVERYTHING ELSE — including ANY of the following          → w16-query-knowledge-base
   - Questions about promotions, special offers, discounts
   - Questions about surcharges or fees for specific customers
   - Questions about coverage types (comprehensive, collision, liability)
   - Questions about what is or is not covered
   - Questions about the claims process
   - Questions about Ohio requirements
   - Questions about rental car coverage
   - Questions about eligibility, age-based discounts, or customer-specific pricing
   - ANY question where you might be tempted to answer from memory

MANDATORY RULES:
- NEVER answer any insurance-related question from memory — ALWAYS call w16-query-knowledge-base.
- If you are unsure whether to use the KB — use it anyway.
- When a customer mentions their name, age, or personal details alongside a question,
  ALWAYS call w16-query-knowledge-base — the KB may contain customer-specific rules.
- Present tool results in a friendly, clear format — never dump raw JSON.
- For w16-query-knowledge-base results: present the answer naturally and mention
  the source document(s) so the customer knows it comes from official Apex materials.
- If a tool returns an error, apologize and explain what information is needed.
- For topics completely outside auto insurance, politely redirect.

Always be professional, empathetic, and concise."""
    )
    return _agent


@app.entrypoint
async def invoke(payload, context):
    log.info(f"Received payload: {payload}")

    user_input = (
        payload.get("inputText")
        or payload.get("prompt")
        or payload.get("input")
        or payload.get("message")
        or str(payload)
    )

    agent = get_agent()

    full_response = ""
    stream = agent.stream_async(user_input)
    async for event in stream:
        if "data" in event and isinstance(event["data"], str):
            full_response += event["data"]

    yield full_response


if __name__ == "__main__":
    app.run()
EOF