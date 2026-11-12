"""Guardrails for the support agent, applied at four points in the loop.

  1. Pre-input   (`run_pre_input_checks`)   — before the user message reaches the LLM:
     rate limit, Unicode normalisation + sanitisation, toxicity (regex over a
     de-obfuscated copy, then a toxic-bert classifier), prompt injection (regex, then
     a DeBERTa injection classifier), off-topic requests (embedding similarity to
     support vs. off-topic prototypes, topic.py). Images: `check_image_nsfw`.
  2. Tool call   (`check_tool_call`)        — before a model-requested tool runs:
     tool allowlist, argument validation, per-message call budgets, and a
     "look before you write" rule for create_return_request.
  3. Tool output (`sanitize_tool_output`)   — before a tool result goes back to the
     model: strips indirect prompt injection hidden in retrieved documents or
     record fields, and masks PII the model does not need.
  4. Output      (`check_output`)           — before the answer reaches the customer:
     (a) blocks system-prompt leakage (shingle overlap);
     (b) blocks toxic / abusive output (regex + ML with stricter thresholds than input);
     (c) blocks off-scope content (medical/legal/financial advice, self-harm, extremism);
     (d) redacts PII (email, card, SSN) and other-customer IDs;
     (e) warns on order IDs not seen in question or tool traffic (possible hallucination);
     (f) warns on specific quantities in the answer absent from tool context;
     (g) warns when the answer asserts policy rules but no policy chunks were retrieved.
     Live NLI grounding (faithfulness / contradiction) runs separately in `grounding.py`.

The classifier layers (`classifiers.py`) run only after the cheap regex layers pass,
and fail open (regex still applies) if a model cannot load; set RSA_ML_GUARDRAILS=0
to switch them off.

The regex checks are deterministic and dependency-free, so every path is
unit-testable without a model. They complement — not replace — the hard rules enforced server-side
(identity pinned on the MCP server, eligibility enforced in `ops`).
"""

from __future__ import annotations

from ._base import (  # noqa: F401
    BlockReason,
    GuardrailResult,
    _ml_on,
)
from .injection import (  # noqa: F401
    _INJECTION_PATTERNS,
    _INJECTION_SIGNALS,
    _RECALL_ASK_RE,
    _RECALL_DENY_RE,
    _RECALL_RE,
    MAX_RECALL_CHARS,
    check_prompt_injection,
    check_prompt_injection_ml,
    is_conversation_recall,
)
from .output import (  # noqa: F401
    _LEAK_MIN_SHINGLES,
    _OFF_SCOPE_BLOCK_MSG,
    _OFF_SCOPE_PATTERNS,
    _OUTPUT_ML_BLOCK_LABELS,
    _POLICY_ASSERT_RE,
    _QUANTITY_ALLOWLIST,
    _QUANTITY_RE,
    _SHINGLE,
    OUTPUT_BLOCK_MESSAGE,
    OutputCheckResult,
    _shingles,
    check_fabricated_numbers,
    check_off_scope_content,
    check_output,
    check_output_toxicity_ml,
    check_ungrounded_policy_assertion,
)
from .pre_input import (  # noqa: F401
    _BLOCK_MESSAGES,
    _PRUNE_AT,
    PreInputCheckResult,
    _request_lock,
    _request_log,
    check_image_nsfw,
    check_off_topic,
    check_rate_limit,
    run_pre_input_checks,
    topic_guard_enabled,
)
from .sanitize import (  # noqa: F401
    _CONTROL_CHARS,
    _EXCESSIVE_NEWLINES,
    _EXCESSIVE_WHITESPACE,
    _FAKE_DELIMITERS,
    _INVISIBLE_CHARS,
    normalize_text,
    sanitize_input,
)
from .tool_calls import (  # noqa: F401
    _AFFIRMATIVE_RE,
    _OFFER_RE,
    _RETURN_ACTION_RE,
    DEFAULT_TOOL_BUDGET,
    MAX_QUERY_CHARS,
    MAX_REASON_CHARS,
    ORDER_ID_RE,
    TOOL_CALL_BUDGET,
    WRITE_TOOLS,
    ToolGuardState,
    _block,
    check_tool_call,
    detect_return_intent,
    record_tool_call,
    tool_block_envelope,
)
from .tool_output import (  # noqa: F401
    _CARD_RE,
    _CUSTOMER_ID_RE,
    _EMAIL_RE,
    _ORDER_REF_RE,
    _SSN_RE,
    INJECTION_PLACEHOLDER,
    PUBLIC_EMAIL_DOMAINS,
    _luhn_ok,
    _mask_card,
    _mask_email,
    redact_pii,
    sanitize_tool_output,
)
from .toxicity import (  # noqa: F401
    _DIRECTED_ABUSE,
    _INSULT,
    _LEET,
    _SEVERE_SLUR_PATTERNS,
    _SPACED_LETTERS,
    _THREAT_PATTERNS,
    _TOXIC_PATTERNS,
    _V_FOR_U,
    ML_BLOCK_LABELS,
    check_toxicity,
    check_toxicity_ml,
    deobfuscate,
)
