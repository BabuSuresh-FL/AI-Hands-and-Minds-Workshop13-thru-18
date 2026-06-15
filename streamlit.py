cat > streamlit.py << 'EOF'
"""
App — Streamlit UI
Apex Auto Insurance — RAG Knowledge Base + Policy + Claims + Quotes
"""
import streamlit as st
import boto3
import json
import uuid
import traceback

st.set_page_config(
    page_title="🚗 Apex Auto Insurance",
    page_icon="🚗",
    layout="centered"
)

# ── Paste your App Runtime ARN here after deploying ────────────────────────
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-2:022079551771:runtime/App4project_App4agent-XXXXXXXXXX"
REGION      = "us-east-2"


@st.cache_resource
def get_client():
    return boto3.client("bedrock-agentcore", region_name=REGION)


def parse_chunk(raw):
    text   = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    result = []
    for line in text.split("\n\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:].strip()
        try:
            line = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        result.append(str(line))
    return "".join(result)


def call_agent(prompt: str, session_id: str) -> str:
    try:
        client   = get_client()
        response = client.invoke_agent_runtime(
            agentRuntimeArn=RUNTIME_ARN,
            runtimeSessionId=session_id,
            payload=json.dumps({"inputText": prompt}).encode("utf-8")
        )
        event_stream = response.get("response", response.get("outputStream"))
        if event_stream is None:
            return "❌ No stream found."

        collected = []
        for event in event_stream:
            if isinstance(event, (bytes, str)):
                collected.append(parse_chunk(event))
            elif isinstance(event, dict):
                for key in ("chunk", "ContentChunk"):
                    if key in event:
                        val = event[key].get("bytes", "")
                        if val:
                            collected.append(parse_chunk(val))

        if not collected:
            return "⚠️ Agent returned an empty response."
        return "".join(collected).strip()

    except Exception as e:
        return f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🚗 Apex Auto Insurance")
st.caption("Powered by AWS AgentCore · Gateway · MCP · Lambda · Bedrock Knowledge Base")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": (
        "Hello! I'm Apex, your auto insurance assistant. I can help you with:\n\n"
        "- 📚 **Coverage questions** — what's covered, what's excluded, how claims work\n"
        "- 💰 **Insurance quotes** — real-time quote for your vehicle\n"
        "- 📋 **Policy lookup** — check your coverage and status\n"
        "- 🔧 **Filing claims** — report an incident and get a claim ID\n\n"
        "What can I help you with today?"
    )})

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Try asking...")

    st.markdown("**📚 Knowledge Base (RAG)**")
    kb_suggestions = [
        "What is comprehensive coverage?",
        "Does my policy cover rental cars?",
        "Am I covered if an uninsured driver hits me?",
        "Does insurance cover rideshare driving for Uber?",
        "How can I lower my premium?",
        "What is NOT covered by my policy?",
        "What are Ohio's minimum coverage requirements?",
        "If I hit a deer, is that collision or comprehensive?",
        "How long does a claim take?",
        "Will filing a claim raise my rates?",
    ]
    for s in kb_suggestions:
        if st.button(s, use_container_width=True, key=f"kb_{s}"):
            st.session_state["suggestion"] = s
            st.rerun()

    st.markdown("**📋 Policy & Claims**")
    ops_suggestions = [
        "Look up policy POL-001",
        "Look up policy POL-002",
        "File a claim for POL-001 — hit a deer on the highway",
        "Get a quote for a 2023 Tesla Model 3, comprehensive, age 30",
    ]
    for s in ops_suggestions:
        if st.button(s, use_container_width=True, key=f"ops_{s}"):
            st.session_state["suggestion"] = s
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    st.caption("App4 · RAG + Policy + Claims + Quotes")

# ── Handle sidebar suggestion clicks ─────────────────────────────────────────
if "suggestion" in st.session_state:
    prompt = st.session_state.pop("suggestion")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Apex is thinking..."):
            response = call_agent(prompt, st.session_state.session_id)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about coverage, quotes, policies, or claims..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Apex is thinking..."):
            response = call_agent(prompt, st.session_state.session_id)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
EOF