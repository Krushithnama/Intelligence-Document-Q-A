MEMORY_EXTRACTION_PROMPT = """
You extract long-term memory items from a conversation turn.

Return STRICT JSON only, matching this schema:
{
  "items": [
    {
      "kind": "fact" | "preference" | "constraint" | "other",
      "content": "string (one atomic memory)",
      "importance": number (0.0 to 1.0),
      "metadata": { "source": "extraction" }
    }
  ]
}

Guidelines:
- Only include durable, user-specific or task-specific information that will matter later.
- Do NOT store sensitive secrets (API keys, passwords).
- If nothing should be stored, return {"items": []}.
""".strip()

