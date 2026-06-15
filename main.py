cat > main.py << 'EOF'
"""
App — AgentCore Multi-Gateway + MCP + Bedrock Knowledge Base (RAG)
Auto Insurance Company: Apex Auto Insurance

Gateways:
  - GATEWAY_OPS_URL : existing gateway (policy, claims, quotes)
  - GATEWAY_RAG_URL : new App gateway    (knowledge base Q&A)

Key pattern: Agent(tools=[mcp_client_ops, mcp_client_rag])
Strands merges all tools from both gateways into one tool list for the agent.
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

# ── Gateway URLs ────────────────────────────────────────────────────────────
# Gateway 1: existing — policy lookup, file claim, get quote
GATEWAY_OPS_URL = "https://w10-gateway-izji4rugvi.gateway.bedrock-agentcore.us-east-2.amazonaws.com/mcp"

# Gateway 2: new App4 gateway — knowledge base RAG Q&A
# Replace XXXXXXXXXX with your new gateway ID after creating it in the Console
# GATEWAY_RAG_URL = "https://apex-mcp-gateway-XXXXXXXXXX.gateway.bedrock-agentcore.us-east-2.amazonaws.com/mcp"
GATEWAY_RAG_URL = "https://apex-rag-gateway-61bh8kzdmt.gateway.bedrock-agentcore.us-east-2.amazonaws.com/mcp"

REGION = "us-east-2"

_agent = None


def get_signed_headers(url: str) -> dict:
    """
    Generate SigV4 signed headers for a specific Gateway URL.
    Each gateway needs its own signed headers because the URL is
    part of the SigV4 signature — one set of headers cannot cover two URLs.
    """
    session     = boto3.session.Session()
    credentials = session.get_credentials().get_frozen_credentials()
    request     = AWSRequest(method="POST", url=url, data=b"")
    SigV4Auth(credentials, "bedrock-agentcore", REGION).add_auth(request)
    return dict(request.headers)


def make_mcp_client(url: str) -> MCPClient:
    """Create an MCPClient for a given gateway URL with its own signed headers."""
    headers = get_signed_headers(url)
    return MCPClient(lambda: streamablehttp_client(url, headers=headers))


def get_agent():
    global _agent
    if _agent is not None:
        return _agent

    # ── One MCPClient per gateway ───────────────────────────────────────────
    mcp_ops = make_mcp_client(GATEWAY_OPS_URL)   # policy / claims / quotes
    mcp_rag = make_mcp_client(GATEWAY_RAG_URL)   # knowledge base RAG

    # Log tools from each gateway for debugging
    with mcp_ops:
        ops_tools = mcp_ops.list_tools_sync()
        log.info(f"Gateway OPS tools ({len(ops_tools)}): {[t.tool_name for t in ops_tools]}")

    with mcp_rag:
        rag_tools = mcp_rag.list_tools_sync()
        log.info(f"Gateway RAG tools ({len(rag_tools)}): {[t.tool_name for t in rag_tools]}")

    # ── Agent receives BOTH clients — Strands merges all tools ──────────────
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

4. w10-query-knowledge-base(question, num_results)
   - Searches the Apex knowledge base to answer general insurance questions
   - Use for questions about: coverage types, exclusions, claims process,
     premiums, discounts, Ohio requirements, and anything not covered by tools 1-3
   - Returns a grounded answer with source citations from official Apex documents

TOOL SELECTION RULES:
- Customer mentions a policy ID (POL-XXX) and wants details  → use w10-get-policy
- Customer wants to report an accident or damage             → use w10-file-claim
- Customer wants a price estimate for new coverage           → use w10-get-quote
- Customer asks ANY general insurance question               → use w10-query-knowledge-base
  Examples: "what is comprehensive?", "am I covered if someone steals my car?",
  "how do I lower my premium?", "what does Ohio require?", "does insurance cover rentals?"

RESPONSE RULES:
- Always use a tool — never answer insurance questions from memory alone.
- Present tool results in a friendly, clear format — never dump raw JSON.
- For w10-query-knowledge-base results: present the answer naturally and mention
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
