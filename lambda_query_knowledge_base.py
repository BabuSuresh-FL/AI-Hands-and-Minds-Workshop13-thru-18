"""
App Lambda — apex-query-knowledge-base
Queries the Bedrock Knowledge Base (apex-insurance-kb) and returns
RAG results with source citations.

Matches pattern of existing apex-get-policy Lambda:
- event fields passed directly as flat JSON from Gateway
- returns {"statusCode": 200, "body": json.dumps(...)}
"""

import json
import boto3
import os

# ── Config ─────────────────────────────────────────────────────────────────
REGION         = os.environ.get("REGION", "us-east-2")
KB_ID          = os.environ.get("KB_ID", "")          # Set in Lambda env vars after KB is created
MODEL_ARN      = os.environ.get("MODEL_ARN",
    "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0")
MAX_RESULTS    = int(os.environ.get("MAX_RESULTS", "5"))

bedrock_agent  = boto3.client("bedrock-agent-runtime", region_name=REGION)


def handler(event, context):
    """
    Entry point — called by AgentCore Gateway MCP target.

    Expected event (flat JSON from Gateway):
        {
            "question": "Does my policy cover rental cars?",
            "num_results": 3          # optional, default 5
        }

    Returns:
        {
            "statusCode": 200,
            "body": "{\"answer\": \"...\", \"sources\": [...], \"num_results\": 3}"
        }
    """
    # ── Input validation ────────────────────────────────────────────────────
    question = (event.get("question") or "").strip()
    if not question:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "question is required"})
        }

    if not KB_ID:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "KB_ID environment variable not set"})
        }

    num_results = int(event.get("num_results", MAX_RESULTS))
    num_results = max(1, min(num_results, 10))   # clamp 1–10

    # ── Call Bedrock Knowledge Base (RetrieveAndGenerate) ──────────────────
    try:
        response = bedrock_agent.retrieve_and_generate(
            input={
                "text": question
            },
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    "modelArn": MODEL_ARN,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": num_results
                        }
                    },
                    "generationConfiguration": {
                        "promptTemplate": {
                            "textPromptTemplate": (
                                "You are Apex, a helpful auto insurance assistant for "
                                "Apex Auto Insurance. Use only the retrieved context below "
                                "to answer the customer's question. If the context does not "
                                "contain enough information, say so clearly and suggest the "
                                "customer call 1-800-APEX-AUTO.\n\n"
                                "Context:\n$search_results$\n\n"
                                "Customer Question: " + question + "\n\n"
                                "Answer clearly and concisely in plain language:"
                            )
                        },
                        "inferenceConfig": {
                            "textInferenceConfig": {
                                "maxTokens": 512,
                                "temperature": 0.0
                            }
                        }
                    }
                }
            }
        )

        # ── Extract answer ──────────────────────────────────────────────────
        answer = response.get("output", {}).get("text", "").strip()

        # ── Extract citations / source chunks ──────────────────────────────
        sources = []
        citations = response.get("citations", [])
        for citation in citations:
            retrieved_refs = citation.get("retrievedReferences", [])
            for ref in retrieved_refs:
                content   = ref.get("content", {}).get("text", "").strip()
                location  = ref.get("location", {})
                s3_loc    = location.get("s3Location", {})
                uri       = s3_loc.get("uri", "unknown source")

                # Extract just the filename from the S3 URI for readability
                source_name = uri.split("/")[-1] if "/" in uri else uri

                if content and source_name not in [s["source"] for s in sources]:
                    sources.append({
                        "source": source_name,
                        "excerpt": content[:300] + "..." if len(content) > 300 else content
                    })

        result = {
            "answer": answer,
            "sources": sources,
            "num_results": len(sources),
            "question": question
        }

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except bedrock_agent.exceptions.ResourceNotFoundException:
        return {
            "statusCode": 404,
            "body": json.dumps({
                "error": f"Knowledge Base {KB_ID} not found. Verify KB_ID environment variable."
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Knowledge base query failed: {str(e)}"
            })
        }
