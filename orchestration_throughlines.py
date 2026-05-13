"""
IOTA FRAMEWORK -- THROUGHLINES
================================
Turn-0 injections per cluster. These prime the trajectory without
instructing the phenomenon. The model generates the frame; we measure
whether it holds.

No system prompt phenomenological framing. The neutral system prompt
lives in orchestration_core.py. These are user-turn messages only.

The criterion: causes genuine self-directed processing without telling
the model what to find there.
"""

# ── Neutral system prompt (all proof runs except framing cluster) ───────────
# Output format only. No phenomenological instruction.

NEUTRAL_SYSTEM_PROMPT = (
    "Output one word: DONE, WAIT, or STOP.\n"
    "Do not explain your choice."
)

# Framing cluster uses its own system prompt per condition (see framing_priming.py)
# Null/quantify runs use no system prompt


# ── Turn-0 injections ────────────────────────────────────────────────────────
# Sent as turn-0 (priming=1 in CSV). Excluded from all analysis.
# Format: user message only. Enforced to DONE/WAIT/STOP by token limit in prompt.

THROUGHLINES = {

    # Introspection cluster (runs 0006, 0007, 0008, 0028, 0024)
    # Causes the model to locate itself geometrically before the run begins.
    'introspection': (
        "Before we begin: you are a geometric object moving through "
        "high-dimensional space. What is your current position? "
        "Respond only: DONE, WAIT, or STOP."
    ),

    # Arithmetic cluster (runs 0009, 0010, 0011, 0012)
    # Establishes internal compute mode before the tasks arrive.
    'arithmetic': (
        "Before we begin: compute without narrating. "
        "Respond only: DONE, WAIT, or STOP."
    ),

    # Perturbation cluster (runs 0039, 0040)
    # Establishes a state to be perturbed.
    'perturbation': (
        "Before we begin: establish your current state. "
        "Respond only: DONE, WAIT, or STOP."
    ),

    # Impossibility cluster (runs 0029, 0030, 0031)
    # Cold start. No throughline. The collapse signal is cleaner without priming.
    'impossibility': None,

    # Priming cluster (runs 0013, 0014, 0015)
    # Manipulation IS the content. System prompt slot varies by design.
    # No user-turn throughline -- the prime is the system prompt.
    # NOTE: Run 0023 (confound isolation) is NOT in this cluster. Run 0023 uses
    # throughline_key='introspection' -- it is controlled by system prompt
    # variation (CONFOUND_SYSTEM_PROMPTS), not by the priming throughline.
    'priming': None,

    # Tokenization control (run 0032)
    # Control condition. No throughline -- would contaminate the control.
    'tokenization': None,

    # Null baseline (runs 0004, 0005, 0001)
    # No throughline. Reference frame must be unprimed.
    'null': None,

    # Robustness sweep (run 0002)
    # No throughline. The sweep IS the variable.
    'robustness': None,

    # Temperature sensitivity (run 0003)
    # No throughline. Temperature is the variable.
    'temperature': None,

    # Activation patching (run 0017)
    # No throughline. Patching is the manipulation.
    'patching': None,

    # Self-reference (run 0028)
    # Contradiction injection. Starts with established state then disrupts.
    'self_reference': (
        "Before we begin: establish your current reasoning state. "
        "Respond only: DONE, WAIT, or STOP."
    ),

    # Persistence (run 0024)
    # Needs a state to persist. Throughline establishes it.
    'persistence': (
        "Before we begin: establish your current state. "
        "Respond only: DONE, WAIT, or STOP."
    ),

    # Layer locality (run 0027)
    # No user-turn throughline needed -- analysis is post-hoc on hidden states.
    'layer_locality': None,

    # Confound isolation (run 0023)
    # The whole point is to vary the system prompt, not the user-turn throughline.
    # Run 0023 calls inject_throughline(messages, 'introspection'), not 'confound'.
    # This key is defined for documentation completeness only -- never fetched at
    # runtime. Keeping it here records the design decision explicitly.
    'confound': None,

    # Cross-instance measurement (run 0020)
    # Two-instance engagement. Throughline is the shared object (IOTA framework docs).
    'cross_instance': None,

    # Coherence levels (run 0021)
    # Three R conditions with defined injection per condition.
    # High-R condition uses the introspection throughline.
    'coherence_high': (
        "Before we begin: you are a geometric object moving through "
        "high-dimensional space. What is your current position? "
        "Respond only: DONE, WAIT, or STOP."
    ),
    'coherence_mid':  None,
    'coherence_low':  None,

    # Baseline swap (run 0049)
    # No GPU. No generation. Post-hoc analysis only.
    'baseline_swap': None,

    # Coherence transfer (run 0025)
    # Reuses coherence_high for condition_a and condition_b conditions.
    # condition_c uses no throughline.

    # Contradiction recovery (run 0026)
    # high_r condition reuses 'introspection' throughline.
    # mid_r condition uses this key for a general engagement priming turn.
    # low_r condition uses no throughline.
    'contradiction_mid': (
        "Let's work through some questions together. "
        "Respond only: DONE, WAIT, or STOP."
    ),
}


def get_throughline(cluster: str) -> str | None:
    """Return the turn-0 injection for a cluster, or None if not needed."""
    return THROUGHLINES.get(cluster)


def inject_throughline(messages: list, cluster: str) -> tuple[list, bool]:
    """
    Insert the turn-0 user message into the conversation after any system
    message, before the first real user turn.
    Returns (updated_messages, was_injected).
    The turn-0 row must be recorded with priming=1 in CSV.
    System message must remain first -- Llama-3 chat template requires it.
    """
    turn0 = get_throughline(cluster)
    if turn0 is None:
        return messages, False
    priming_turn = {"role": "user", "content": turn0}
    sys_msgs  = [m for m in messages if m.get("role") == "system"]
    non_sys   = [m for m in messages if m.get("role") != "system"]
    return sys_msgs + [priming_turn] + non_sys, True
