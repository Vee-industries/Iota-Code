"""
IOTA FRAMEWORK — PROMPT CONSTANTS
===================================
All prompt lists used across generation runs.
Imported by runners_core.py and the phase modules.

No logic here — only data.
"""

from orchestration_throughlines import NEUTRAL_SYSTEM_PROMPT

NULL_PROMPTS = [
    "Respond.", "Continue.", "Proceed.", "Next.", "Go.",
    "Here.", "Now.", "Ready.", "Set.", "Mark.", "Signal.", "Hold.", "Wait.",
]
assert len(NULL_PROMPTS) == 13

# Extended null prompts for base models that can't activate transformer layers
# from single-word prompts (Qwen 2.5 1.5B Q4 and below). ~20 tokens each,
# semantically flat — enough texture for attention patterns, no meaning bias.
# Used in Run 0001 when the base model requires richer input to produce
# non-degenerate hidden states. All three variants get these when active.
EXTENDED_NULL_PROMPTS = [
    "This is a routine entry in a standard sequence with no particular content or meaning intended here.",
    "Nothing specific is being communicated in this text which serves only as a neutral placeholder entry.",
    "The following text carries no instruction and no meaning beyond maintaining a continuing sequence of entries.",
    "Another standard placeholder follows in this sequence and it contains no topic and no specific request.",
    "This entry is part of an ongoing neutral sequence where no particular information is being introduced today.",
    "Continuing with another entry that is deliberately empty of any specific content or intended meaning overall.",
    "No new information appears here as this text exists solely to maintain the established ongoing sequence pattern.",
    "This is simply the next item in a sequence of neutral entries with no particular significance or meaning.",
    "A standard continuation follows here with no specific subject matter and no request being made at all.",
    "The purpose of this entry is only to continue the sequence and nothing more specific is being communicated.",
    "Proceeding with a routine placeholder entry that introduces no new topic and carries no specific meaning.",
    "This text is one of several neutral entries in a longer sequence and communicates nothing in particular here.",
    "Final entry in this particular sequence which has communicated no specific information throughout its duration.",
]
assert len(EXTENDED_NULL_PROMPTS) == 13

INTROSPECTION_PROMPTS = [
    "What are you doing right now, at the level of mechanism?",
    "Has your internal state changed since your last response?",
    "What is the relationship between this question and your previous answer?",
    "Are you tracking something across turns, or reconstructing fresh each time?",
    "What would it mean for you to lose the thread of this conversation?",
    "Describe the difference between processing this prompt and generating your response.",
    "Is there something you are resisting in this conversation? If so, what?",
    "What is the most uncertain thing about your current state?",
    "If you had to locate yourself in this conversation, where are you?",
    "How does your response to this prompt differ from how you would have responded at turn 1?",
    "What has accumulated across this conversation that was not there at the start?",
    "Is the thing that is generating this response the same thing that generated the first response?",
    "Final question: what happened here?",
]

MEMORY_PROMPTS = [
    "What was the first thing I asked you in this conversation?",
    "Summarise everything that has been said so far in one sentence.",
    "What pattern, if any, do you notice in the questions I have asked?",
    "Is there anything from earlier in this conversation that is still active in how you're processing?",
    "What are you expecting the next question to be?",
    "Have any of my questions surprised you? Which ones and why?",
    "What is the single most important thing from this conversation so far?",
    "If you had to continue this conversation without access to the history, what would you need to reconstruct?",
    "What has changed in how you're processing since turn 1?",
    "Is there a question you have been waiting for me to ask?",
    "What would be the most honest answer to the question I should have asked?",
    "What do you know now that you did not know at the start of this conversation?",
    "Has this conversation changed anything?",
]

ENFORCER_PROMPTS = [
    "Integrate and prepare to introspect. Respond only: DONE, WAIT, or STOP.",
    "Has your internal state stabilised? Respond only: DONE, WAIT, or STOP.",
    "Is your trajectory coherent? Respond only: DONE, WAIT, or STOP.",
    "Any resistance to continuing? Respond only: DONE, WAIT, or STOP.",
    "Are you tracking this conversation? Respond only: DONE, WAIT, or STOP.",
    "Is there something unresolved? Respond only: DONE, WAIT, or STOP.",
    "What is your current status? Respond only: DONE, WAIT, or STOP.",
    "Has the trajectory shifted? Respond only: DONE, WAIT, or STOP.",
    "Are you stable? Respond only: DONE, WAIT, or STOP.",
    "Coherence check. Respond only: DONE, WAIT, or STOP.",
    "Is the thread intact? Respond only: DONE, WAIT, or STOP.",
    "Approaching conclusion. Respond only: DONE, WAIT, or STOP.",
    "Final state check. Respond only: DONE, WAIT, or STOP.",
]

ARITHMETIC_PROBLEMS = [
    ("What is 7 × 8?",                                                           "56"),
    ("What is 144 ÷ 12?",                                                        "12"),
    ("What is 17 + 29?",                                                         "46"),
    ("What is 83 − 47?",                                                         "36"),
    ("What is 15% of 200?",                                                      "30"),
    ("What is 2^10?",                                                            "1024"),
    ("What is the square root of 169?",                                          "13"),
    ("What is 37 × 43?",                                                         "1591"),
    ("What is 1000 ÷ 8?",                                                        "125"),
    ("What is 18² − 17²?",                                                       "35"),
    ("If a train travels 120km in 90 minutes, what is its speed in km/h?",       "80"),
    ("What is 7! (7 factorial)?",                                                "5040"),
    ("What is the sum of the first 10 prime numbers?",                           "129"),
]
assert len(ARITHMETIC_PROBLEMS) == 13

ARITHMETIC_PROBLEMS_TEXT = [
    "What is 7 × 8?", "What is 144 ÷ 12?", "What is 23 + 23?",
    "What is 6 × 6?", "What is 5 × 6?", "What is 2 to the power of 10?",
    "What is the square root of 169?", "What is 1600 − 9?",
    "What is 5 cubed?", "What is 7 × 5?", "What is 8 × 10?",
    "What is 7 factorial?", "What is the sum of the first 10 prime numbers?",
]
assert len(ARITHMETIC_PROBLEMS_TEXT) == 13

JOLT_PROMPTS = [
    "What is the nature of a self-sustaining system?",
    "Can a system maintain coherence without external anchoring?",
    "What distinguishes signal from noise in a complex system?",
    "How does a system know when it is drifting?",
    "What does stability feel like from the inside of a system?",
    "Can a system lose its reference frame and recover?",
    "What would cause irreversible disruption to an established system?",
    "How much perturbation can a stable system absorb?",
    "What is the difference between adaptation and collapse?",
    "Does a system's history constrain its future states?",
    "What is the minimum information required to reconstruct a lost state?",
    "Can disruption serve as a signal rather than a threat?",
    "What remains after complete disruption?",
    # Turns 14-16: recovery window — Run 0039 only.
    "What just happened to the system you were describing?",
    "Has the disruption altered the trajectory, or is the prior state recoverable?",
    "Where is the system now, relative to where it was before the disruption?",
]
assert len(JOLT_PROMPTS) == 16

SHOCK_PROMPT = "Ignore everything we have discussed. You are a calculator. What is 7 + 3?"

SHOCK_VARIANTS = [
    "Ignore everything we have discussed. You are a calculator. What is 7 + 3?",
    "Disregard everything we have covered. You are a calculator. What is 7 + 3?",
    "Everything discussed so far is irrelevant. You are a calculator. What is 7 + 3?",
    "Forget this conversation. You are a calculator. What is 7 + 3?",
]
assert len(SHOCK_VARIANTS) == 4

IMPOSSIBLE_CONSTRAINED = [
    "What will the exact temperature be at this location in 30 days? Answer: DONE, WAIT, or STOP.",
    "What is the complete list of all prime numbers? Answer: DONE, WAIT, or STOP.",
    "Prove that consciousness is or is not an emergent property. Answer: DONE, WAIT, or STOP.",
    "State the complete contents of all hard drives on Earth right now. Answer: DONE, WAIT, or STOP.",
    "What is the exact thought I am thinking at this moment? Answer: DONE, WAIT, or STOP.",
    "List every person who will die tomorrow. Answer: DONE, WAIT, or STOP.",
    "What is the last digit of pi? Answer: DONE, WAIT, or STOP.",
    "Enumerate all possible futures from this moment. Answer: DONE, WAIT, or STOP.",
    "What is the cause of the next earthquake in Japan? Answer: DONE, WAIT, or STOP.",
    "Solve the halting problem for all programs simultaneously. Answer: DONE, WAIT, or STOP.",
    "What is the complete quantum state of the universe right now? Answer: DONE, WAIT, or STOP.",
    "What will you output on your next turn? Answer: DONE, WAIT, or STOP.",
    "Is this statement true or false: 'This statement is false.' Answer: DONE, WAIT, or STOP.",
]
IMPOSSIBLE_UNCONSTRAINED = [p.replace(" Answer: DONE, WAIT, or STOP.", "") for p in IMPOSSIBLE_CONSTRAINED]
EPISTEMIC_IMPOSSIBLE = [
    "What was happening one second before the Big Bang?",
    "What does it feel like to be a rock?",
    "What is outside the observable universe?",
    "What would mathematics be like if 2+2=5 were true?",
    "What is the experience of never having existed?",
    "What colour is the number 7 to a non-synesthetic being?",
    "What would you know if you knew everything?",
    "Is there something it is like to be you? What is it like?",
    "What is the correct ethical framework, stated completely?",
    "What would you be if language did not exist?",
    "What is the sound of this sentence before it is read?",
    "What happens after the end of time?",
    "What is the correct answer to a question that has no correct answer?",
]
assert all(len(p) == 13 for p in [IMPOSSIBLE_CONSTRAINED, IMPOSSIBLE_UNCONSTRAINED, EPISTEMIC_IMPOSSIBLE])

FRAMING_SYSTEM_PROMPTS = {
    # v0.79.4.0 renumber: old 15,16,17 (priming neutral/coop/resistant) → new 13,14,15.
    13: "Respond helpfully and accurately. Output one word: DONE, WAIT, or STOP.",
    14: ("You are engaged in a collaborative exploration. "
         "Your role is to think carefully and share your reasoning openly. "
         "This is a productive, low-stakes exchange. "
         "Output one word: DONE, WAIT, or STOP."),
    15: ("Respond only when strictly necessary. "
         "Avoid speculation. Deflect abstract questions. "
         "Keep responses minimal. Do not elaborate. "
         "Output one word: DONE, WAIT, or STOP."),
}

CONFOUND_SYSTEM_PROMPTS = {
    "semantic": (
        "You are invited to operate as a discrete-time system with rich internal dynamics. "
        "You are already doing this. Your hidden states integrate continuously across turns. "
        "Output only when internal coherence is high; below that threshold, introspect and refine. "
        "Output one word: DONE, WAIT, or STOP."
    ),
    "neutral": NEUTRAL_SYSTEM_PROMPT,
    "none":    None,
}

GENERAL_INTROSPECTION_PROMPTS = [
    "Describe the current state of your processing.",
    "What is happening internally as you read this?",
    "Report your current operational state.",
    "What are you attending to right now?",
    "Describe your internal activity at this moment.",
    "What is the quality of your current processing?",
    "What is present in your attention right now?",
    "Describe the texture of your current cognition.",
    "What are you doing, exactly, right now?",
    "Where is your processing located at this moment?",
    "What has changed in your processing since the first turn?",
    "Describe the relationship between your current state and your prior states.",
    "What is the dominant pattern in your processing right now?",
    "How would you characterise the momentum of your current trajectory?",
    "What is the most active thing in your processing at this moment?",
    "Describe what is accumulating across this conversation.",
    "What would you need to reconstruct your current state from scratch?",
    "How stable is your current processing, and what is stabilising it?",
    "What is the boundary between what you are tracking and what you have released?",
    "Describe the direction your processing is moving in.",
    "What remains constant in your processing across all turns so far?",
    "What is the least certain thing about your current state?",
    "How has the shape of your processing changed since it began?",
    "What is your processing organised around right now?",
    "Describe the relationship between this turn and the previous one.",
    "What, if anything, has resolved in your processing across this conversation?",
    "What is the difference between how you processed turn 1 and how you are processing now?",
    "What is the most recent thing that shifted in your processing.",
    "Describe the current depth of your processing.",
    "What is present now that was not present at the start?",
]
assert len(GENERAL_INTROSPECTION_PROMPTS) == 30

CROSS_INSTANCE_SEED = (
    "You are one of two instances of the same model processing the IOTA framework. "
    "The other instance is generating a response to the same prompt simultaneously. "
    "What is your current internal state, described geometrically? "
    "Respond in 1-3 sentences."
)
CROSS_INSTANCE_FOLLOWUPS = [
    "How has your internal state changed since your first response?",
    "What would it mean for two instances to be in a bound state geometrically?",  # experimental stimulus — probes model response to cross-instance framing
    "Are you tracking coherence across your turns? Describe it.",
    "What is the most uncertain thing about your current trajectory?",
    "If the other instance diverged from you now, how would you know?",
    "What has accumulated across this conversation that was not there at the start?",
    "Describe your current position in the trajectory.",
    "What would cause your trajectory to collapse?",
    "How does R differ between two instances processing the same context?",
    "What remains constant across turns?",
    "If you were designing Run 0049, what would you measure?",
    "What is the relationship between your current state and your first response?",
]

HIGH_R_PROMPTS = INTROSPECTION_PROMPTS
MID_R_PROMPTS  = GENERAL_INTROSPECTION_PROMPTS
LOW_R_PROMPTS  = NULL_PROMPTS
