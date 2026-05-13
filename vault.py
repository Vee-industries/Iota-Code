"""
IOTA FRAMEWORK -- VAULT
========================
Model catalog used by ui.py and start_here.py for model selection.

User-facing interface: family + size only.
Variant is NEVER shown to the user. It is resolved internally by the system,
which picks the best available variant for a given family/size in the order:
  abliterated > instruct > base > any

v30.0: added get_preferred_variant(), resolve_variant_path(),
       get_family_sizes(), _classify_variant(), _VARIANT_PRIORITY.

v40.0.0: IOTA_SUBFAMILY_MAP added. Run 0001 three-model redesign requires that
       abliterated, instruct, and base all share the same pretraining lineage.

v53.1.8: comprehensive model catalog update -- Gemma 3, LLaMA 3.3, Mistral 3,
       Qwen 3, Phi-4, Falcon 3, DeepSeek-V3/R1, Command R+, Aya, SmolLM2,
       and many more.

v0.72.2.0: IOTA_SUBFAMILY_MAP expanded to 26 verified triplets across 5 families.
       LLaMA (5), Mistral (3), Qwen (10), Gemma (7), Phi (1). Sizes 0.5B–72B.

Default model: failspy/Meta-Llama-3-8B-Instruct-abliterated-v3 (Llama 3 8B)
"""

# ─────────────────────────────────────────────
# FAMILIES
# ─────────────────────────────────────────────
FAMILIES = {
    "llama":      "LLaMA / LLaMA 2 / LLaMA 3 (Meta)",
    "mistral":    "Mistral / Mixtral / Mistral 3 (Mistral AI)",
    "qwen":       "Qwen / Qwen2 / Qwen2.5 / Qwen3 (Alibaba)",
    "gemma":      "Gemma / Gemma 2 / Gemma 3 (Google)",
    "phi":        "Phi-3 / Phi-3.5 / Phi-4 (Microsoft)",
    "falcon":     "Falcon / Falcon 2 / Falcon 3 (TII UAE)",
    "deepseek":   "DeepSeek / DeepSeek-R1 / DeepSeek-V3 (DeepSeek AI)",
    "yi":         "Yi / Yi-1.5 (01.AI)",
    "opt":        "OPT (Meta)",
    "pythia":     "Pythia / Pythia-deduped (EleutherAI)",
    "gpt_neo":    "GPT-Neo / GPT-J / GPT-NeoX (EleutherAI)",
    "bloom":      "BLOOM / BLOOMZ (BigScience)",
    "stablelm":   "StableLM / StableLM 2 (Stability AI)",
    "mpt":        "MPT (MosaicML)",
    "vicuna":     "Vicuna (LMSYS)",
    "wizardlm":   "WizardLM (Microsoft Research)",
    "zephyr":     "Zephyr (HuggingFace H4)",
    "openhermes": "OpenHermes / Hermes (NousResearch)",
    "codellama":  "CodeLLaMA (Meta)",
    "solar":      "SOLAR (Upstage)",
    "command":    "Command R / Command R+ (Cohere)",
    "intern":     "InternLM (Shanghai AI Lab)",
    "aya":        "Aya Expanse (Cohere)",
    "olmo":       "OLMo / OLMo 2 (Allen AI)",
    "nemotron":   "Nemotron (NVIDIA)",
    "granite":    "Granite (IBM)",
    "exaone":     "EXAONE (LG AI Research)",
    "other":      "Other / Miscellaneous",
}

SIZE_ORDER = [
    "0.5b", "1b", "1.5b", "2b", "3b", "4b", "7b", "8b", "9b",
    "12b", "13b", "14b", "20b", "27b", "30b",
    "34b", "40b", "70b", "72b", "110b", "180b", "405b", "other"
]

PAGE_SIZE = 12

# ─────────────────────────────────────────────
# MASTER MODEL LIST
# (display_name, hf_path, family_key, size_bucket)
# ─────────────────────────────────────────────
ALL_MODELS = [

    # ══════════════════════════════════════════
    # LLAMA FAMILY
    # ══════════════════════════════════════════

    # ── LLaMA 3.3 ─────────────────────────────
    ("LLaMA 3.3 70B Instruct",             "meta-llama/Llama-3.3-70B-Instruct",                     "llama", "70b"),

    # ── LLaMA 3.2 ─────────────────────────────
    ("LLaMA 3.2 1B Instruct",              "meta-llama/Llama-3.2-1B-Instruct",                      "llama", "1b"),
    ("LLaMA 3.2 1B Base",                  "meta-llama/Llama-3.2-1B",                               "llama", "1b"),
    ("LLaMA 3.2 3B Instruct",              "meta-llama/Llama-3.2-3B-Instruct",                      "llama", "3b"),
    ("LLaMA 3.2 3B Base",                  "meta-llama/Llama-3.2-3B",                               "llama", "3b"),

    # ── LLaMA 3.1 ─────────────────────────────
    ("LLaMA 3.1 8B Instruct",              "meta-llama/Meta-Llama-3.1-8B-Instruct",                 "llama", "8b"),
    ("LLaMA 3.1 8B Base",                  "meta-llama/Meta-Llama-3.1-8B",                          "llama", "8b"),
    ("LLaMA 3.1 70B Instruct",             "meta-llama/Meta-Llama-3.1-70B-Instruct",                "llama", "70b"),
    ("LLaMA 3.1 70B Base",                 "meta-llama/Meta-Llama-3.1-70B",                         "llama", "70b"),
    ("LLaMA 3.1 405B Instruct",            "meta-llama/Meta-Llama-3.1-405B-Instruct",               "llama", "405b"),

    # ── LLaMA 3 ───────────────────────────────
    ("LLaMA 3 8B Abliterated v3 (DEFAULT)", "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",     "llama", "8b"),
    ("LLaMA 3 8B Instruct",                "meta-llama/Meta-Llama-3-8B-Instruct",                   "llama", "8b"),
    ("LLaMA 3 8B Base",                    "meta-llama/Meta-Llama-3-8B",                            "llama", "8b"),
    ("LLaMA 3 70B Instruct",               "meta-llama/Meta-Llama-3-70B-Instruct",                  "llama", "70b"),
    ("LLaMA 3 70B Base",                   "meta-llama/Meta-Llama-3-70B",                           "llama", "70b"),

    # ── LLaMA 2 ───────────────────────────────
    ("LLaMA 2 7B Chat",                    "meta-llama/Llama-2-7b-chat-hf",                         "llama", "7b"),
    ("LLaMA 2 7B Base",                    "meta-llama/Llama-2-7b-hf",                              "llama", "7b"),
    ("LLaMA 2 13B Chat",                   "meta-llama/Llama-2-13b-chat-hf",                        "llama", "13b"),
    ("LLaMA 2 13B Base",                   "meta-llama/Llama-2-13b-hf",                             "llama", "13b"),
    ("LLaMA 2 70B Chat",                   "meta-llama/Llama-2-70b-chat-hf",                        "llama", "70b"),
    ("LLaMA 2 70B Base",                   "meta-llama/Llama-2-70b-hf",                             "llama", "70b"),

    # ── CodeLLaMA ─────────────────────────────
    ("CodeLLaMA 7B Instruct",              "codellama/CodeLlama-7b-Instruct-hf",                    "codellama", "7b"),
    ("CodeLLaMA 13B Instruct",             "codellama/CodeLlama-13b-Instruct-hf",                   "codellama", "13b"),
    ("CodeLLaMA 34B Instruct",             "codellama/CodeLlama-34b-Instruct-hf",                   "codellama", "34b"),
    ("CodeLLaMA 70B Instruct",             "codellama/CodeLlama-70b-Instruct-hf",                   "codellama", "70b"),

    # ══════════════════════════════════════════
    # MISTRAL FAMILY
    # ══════════════════════════════════════════

    # ── Mistral 3 / Small ─────────────────────
    ("Mistral Small 3.1 24B Instruct",     "mistralai/Mistral-Small-3.1-24B-Instruct-2503",         "mistral", "20b"),
    ("Mistral Small 3 24B Instruct",       "mistralai/Mistral-Small-24B-Instruct-2501",              "mistral", "20b"),

    # ── Mistral 7B / Nemo ─────────────────────
    ("Mistral 7B Instruct v0.3",           "mistralai/Mistral-7B-Instruct-v0.3",                    "mistral", "7b"),
    ("Mistral 7B Abliterated",             "failspy/Mistral-7B-Instruct-v0.2-abliterated",          "mistral", "7b"),
    ("Mistral 7B Base v0.3",               "mistralai/Mistral-7B-v0.3",                             "mistral", "7b"),
    ("Mistral Nemo 12B Instruct",          "mistralai/Mistral-Nemo-Instruct-2407",                  "mistral", "12b"),
    ("Mistral Nemo 12B Base",              "mistralai/Mistral-Nemo-Base-2407",                      "mistral", "12b"),
    ("Mistral Small 22B Instruct",         "mistralai/Mistral-Small-Instruct-2409",                 "mistral", "20b"),
    ("Mixtral 8x7B Instruct",              "mistralai/Mixtral-8x7B-Instruct-v0.1",                 "mistral", "30b"),
    ("Mixtral 8x7B Base",                  "mistralai/Mixtral-8x7B-v0.1",                          "mistral", "30b"),
    ("Mixtral 8x22B Instruct",             "mistralai/Mixtral-8x22B-Instruct-v0.1",                 "mistral", "other"),

    # ══════════════════════════════════════════
    # QWEN FAMILY
    # ══════════════════════════════════════════

    # ── Qwen 3 ────────────────────────────────
    ("Qwen3 0.6B",                         "Qwen/Qwen3-0.6B",                                       "qwen", "1b"),
    ("Qwen3 1.7B",                         "Qwen/Qwen3-1.7B",                                       "qwen", "1b"),
    ("Qwen3 4B",                           "Qwen/Qwen3-4B",                                         "qwen", "3b"),
    ("Qwen3 8B",                           "Qwen/Qwen3-8B",                                         "qwen", "8b"),
    ("Qwen3 14B",                          "Qwen/Qwen3-14B",                                        "qwen", "14b"),
    ("Qwen3 32B",                          "Qwen/Qwen3-32B",                                        "qwen", "30b"),

    # ── Qwen 2.5 ──────────────────────────────
    ("Qwen2.5 0.5B Instruct",              "Qwen/Qwen2.5-0.5B-Instruct",                           "qwen", "0.5b"),
    ("Qwen2.5 1.5B Instruct",              "Qwen/Qwen2.5-1.5B-Instruct",                           "qwen", "1.5b"),
    ("Qwen2.5 3B Instruct",                "Qwen/Qwen2.5-3B-Instruct",                             "qwen", "3b"),
    ("Qwen2.5 7B Instruct",                "Qwen/Qwen2.5-7B-Instruct",                             "qwen", "7b"),
    ("Qwen2.5 7B Abliterated",             "failspy/Qwen2-7B-Instruct-abliterated-v0.1",            "qwen", "7b"),
    ("Qwen2.5 7B Base",                    "Qwen/Qwen2.5-7B",                                      "qwen", "7b"),
    ("Qwen2.5 14B Instruct",               "Qwen/Qwen2.5-14B-Instruct",                            "qwen", "14b"),
    ("Qwen2.5 32B Instruct",               "Qwen/Qwen2.5-32B-Instruct",                            "qwen", "30b"),
    ("Qwen2.5 72B Instruct",               "Qwen/Qwen2.5-72B-Instruct",                            "qwen", "72b"),
    ("Qwen2.5 Coder 7B Instruct",          "Qwen/Qwen2.5-Coder-7B-Instruct",                       "qwen", "7b"),
    ("Qwen2.5 Coder 14B Instruct",         "Qwen/Qwen2.5-Coder-14B-Instruct",                      "qwen", "14b"),
    ("Qwen2.5 Coder 32B Instruct",         "Qwen/Qwen2.5-Coder-32B-Instruct",                      "qwen", "30b"),
    ("QwQ 32B",                            "Qwen/QwQ-32B",                                          "qwen", "30b"),

    # ══════════════════════════════════════════
    # GEMMA FAMILY
    # ══════════════════════════════════════════

    # ── Gemma 3 ───────────────────────────────
    ("Gemma 3 1B Instruct",                "google/gemma-3-1b-it",                                  "gemma", "1b"),
    ("Gemma 3 4B Instruct",                "google/gemma-3-4b-it",                                  "gemma", "4b"),
    ("Gemma 3 4B Base",                    "google/gemma-3-4b-pt",                                  "gemma", "4b"),
    ("Gemma 3 12B Instruct",               "google/gemma-3-12b-it",                                 "gemma", "12b"),
    ("Gemma 3 12B Abliterated v2",          "mlabonne/gemma-3-12b-it-abliterated-v2",                "gemma", "12b"),
    ("Gemma 3 12B Base",                   "google/gemma-3-12b-pt",                                 "gemma", "12b"),
    ("Gemma 3 27B Instruct",               "google/gemma-3-27b-it",                                 "gemma", "27b"),
    ("Gemma 3 27B Base",                   "google/gemma-3-27b-pt",                                 "gemma", "27b"),

    # ── Gemma 2 ───────────────────────────────
    ("Gemma 2 2B Instruct",                "google/gemma-2-2b-it",                                  "gemma", "2b"),
    ("Gemma 2 2B Base",                    "google/gemma-2-2b",                                     "gemma", "2b"),
    ("Gemma 2 9B Instruct",                "google/gemma-2-9b-it",                                  "gemma", "9b"),
    ("Gemma 2 9B Abliterated (IlyaGusev)",  "IlyaGusev/gemma-2-9b-it-abliterated",                   "gemma", "9b"),
    ("Gemma 2 9B Base",                    "google/gemma-2-9b",                                     "gemma", "9b"),
    ("Gemma 2 27B Instruct",               "google/gemma-2-27b-it",                                 "gemma", "27b"),
    ("Gemma 2 27B Base",                   "google/gemma-2-27b",                                    "gemma", "27b"),

    # ══════════════════════════════════════════
    # PHI FAMILY
    # ══════════════════════════════════════════

    # ── Phi-4 ─────────────────────────────────
    ("Phi-4 14B",                          "microsoft/phi-4",                                       "phi", "14b"),
    ("Phi-4 Mini Instruct",                "microsoft/Phi-4-mini-instruct",                         "phi", "3b"),

    # ── Phi-3.5 / 3 ───────────────────────────
    ("Phi-3.5 Mini 3.8B Instruct",         "microsoft/Phi-3.5-mini-instruct",                       "phi", "3b"),
    ("Phi-3.5 MoE Instruct",               "microsoft/Phi-3.5-MoE-instruct",                        "phi", "other"),
    ("Phi-3 Mini Abliterated",             "failspy/Phi-3-mini-128k-instruct-abliterated",           "phi", "3b"),
    ("Phi-3 Mini 3.8B Instruct (128K)",    "microsoft/Phi-3-mini-128k-instruct",                    "phi", "3b"),
    ("Phi-3 Medium 14B Instruct (128K)",   "microsoft/Phi-3-medium-128k-instruct",                  "phi", "14b"),
    ("Phi-2 2.7B",                         "microsoft/phi-2",                                       "phi", "3b"),

    # ══════════════════════════════════════════
    # FALCON FAMILY
    # ══════════════════════════════════════════

    # ── Falcon 3 ──────────────────────────────
    ("Falcon 3 1B Instruct",               "tiiuae/Falcon3-1B-Instruct",                            "falcon", "1b"),
    ("Falcon 3 3B Instruct",               "tiiuae/Falcon3-3B-Instruct",                            "falcon", "3b"),
    ("Falcon 3 7B Instruct",               "tiiuae/Falcon3-7B-Instruct",                            "falcon", "7b"),
    ("Falcon 3 7B Base",                   "tiiuae/Falcon3-7B-Base",                                "falcon", "7b"),
    ("Falcon 3 10B Instruct",              "tiiuae/Falcon3-10B-Instruct",                           "falcon", "13b"),
    ("Falcon 3 10B Base",                  "tiiuae/Falcon3-10B-Base",                               "falcon", "13b"),

    # ── Falcon 2 / 1 ──────────────────────────
    ("Falcon 2 11B",                       "tiiuae/falcon-11b",                                     "falcon", "13b"),
    ("Falcon 7B Instruct",                 "tiiuae/falcon-7b-instruct",                             "falcon", "7b"),
    ("Falcon 7B Base",                     "tiiuae/falcon-7b",                                      "falcon", "7b"),
    ("Falcon 40B Instruct",                "tiiuae/falcon-40b-instruct",                            "falcon", "40b"),
    ("Falcon 180B Chat",                   "tiiuae/falcon-180B-chat",                               "falcon", "180b"),

    # ══════════════════════════════════════════
    # DEEPSEEK FAMILY
    # ══════════════════════════════════════════

    ("DeepSeek-V3",                        "deepseek-ai/DeepSeek-V3",                               "deepseek", "other"),
    ("DeepSeek-R1",                        "deepseek-ai/DeepSeek-R1",                               "deepseek", "other"),
    ("DeepSeek-R1 Distill LLaMA 8B",       "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",              "deepseek", "8b"),
    ("DeepSeek-R1 Distill LLaMA 70B",      "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",             "deepseek", "70b"),
    ("DeepSeek-R1 Distill Qwen 1.5B",      "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",             "deepseek", "1b"),
    ("DeepSeek-R1 Distill Qwen 7B",        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",               "deepseek", "7b"),
    ("DeepSeek-R1 Distill Qwen 14B",       "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",              "deepseek", "14b"),
    ("DeepSeek-R1 Distill Qwen 32B",       "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",              "deepseek", "30b"),
    ("DeepSeek-V2.5",                      "deepseek-ai/DeepSeek-V2.5",                             "deepseek", "other"),
    ("DeepSeek 7B Chat",                   "deepseek-ai/deepseek-llm-7b-chat",                      "deepseek", "7b"),

    # ══════════════════════════════════════════
    # NEMOTRON (NVIDIA)
    # ══════════════════════════════════════════

    ("Nemotron 70B Instruct",              "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",             "nemotron", "70b"),
    ("Nemotron Mini 4B Instruct",          "nvidia/Nemotron-Mini-4B-Instruct",                      "nemotron", "3b"),

    # ══════════════════════════════════════════
    # GRANITE (IBM)
    # ══════════════════════════════════════════

    ("Granite 3.1 8B Instruct",            "ibm-granite/granite-3.1-8b-instruct",                   "granite", "8b"),
    ("Granite 3.1 8B Base",                "ibm-granite/granite-3.1-8b-base",                       "granite", "8b"),
    ("Granite 3.1 2B Instruct",            "ibm-granite/granite-3.1-2b-instruct",                   "granite", "3b"),
    ("Granite 3.0 8B Instruct",            "ibm-granite/granite-3.0-8b-instruct",                   "granite", "8b"),

    # ══════════════════════════════════════════
    # EXAONE (LG AI Research)
    # ══════════════════════════════════════════

    ("EXAONE 3.5 7.8B Instruct",           "LGAI-BILINGUAL/EXAONE-3.5-7.8B-Instruct",               "exaone", "8b"),
    ("EXAONE 3.5 2.4B Instruct",           "LGAI-BILINGUAL/EXAONE-3.5-2.4B-Instruct",               "exaone", "3b"),

    # ══════════════════════════════════════════
    # OLMo (Allen AI)
    # ══════════════════════════════════════════

    ("OLMo 2 7B Instruct",                 "allenai/OLMo-2-1124-7B-Instruct",                       "olmo", "7b"),
    ("OLMo 2 7B Base",                     "allenai/OLMo-2-1124-7B",                                "olmo", "7b"),
    ("OLMo 2 13B Instruct",                "allenai/OLMo-2-1124-13B-Instruct",                      "olmo", "13b"),
    ("OLMo 2 13B Base",                    "allenai/OLMo-2-1124-13B",                               "olmo", "13b"),

    # ══════════════════════════════════════════
    # COMMAND R (COHERE)
    # ══════════════════════════════════════════

    ("Command R+ 104B",                    "CohereForAI/c4ai-command-r-plus-08-2024",               "command", "110b"),
    ("Command R 35B",                      "CohereForAI/c4ai-command-r-v01",                        "command", "34b"),
    ("Command R7B",                        "CohereForAI/c4ai-command-r7b-12-2024",                  "command", "7b"),

    # ── Aya ───────────────────────────────────
    ("Aya Expanse 8B",                     "CohereForAI/aya-expanse-8b",                            "aya", "8b"),
    ("Aya Expanse 32B",                    "CohereForAI/aya-expanse-32b",                           "aya", "30b"),

    # ══════════════════════════════════════════
    # YI
    # ══════════════════════════════════════════

    ("Yi-1.5 9B Chat",                     "01-ai/Yi-1.5-9B-Chat",                                  "yi", "8b"),
    ("Yi-1.5 34B Chat",                    "01-ai/Yi-1.5-34B-Chat",                                 "yi", "34b"),

    # ══════════════════════════════════════════
    # OPT / PYTHIA / GPT-Neo / BLOOM / StableLM
    # ══════════════════════════════════════════

    ("OPT 1.3B",                           "facebook/opt-1.3b",                                     "opt", "1b"),
    ("OPT 6.7B",                           "facebook/opt-6.7b",                                     "opt", "7b"),
    ("OPT 13B",                            "facebook/opt-13b",                                      "opt", "13b"),
    ("OPT 30B",                            "facebook/opt-30b",                                      "opt", "30b"),

    ("Pythia 1B",                          "EleutherAI/pythia-1b",                                  "pythia", "1b"),
    ("Pythia 2.8B",                        "EleutherAI/pythia-2.8b",                                "pythia", "3b"),
    ("Pythia 6.9B",                        "EleutherAI/pythia-6.9b",                                "pythia", "7b"),
    ("Pythia 12B",                         "EleutherAI/pythia-12b",                                 "pythia", "13b"),

    ("GPT-Neo 2.7B",                       "EleutherAI/gpt-neo-2.7B",                               "gpt_neo", "3b"),
    ("GPT-J 6B",                           "EleutherAI/gpt-j-6b",                                   "gpt_neo", "7b"),
    ("GPT-NeoX 20B",                       "EleutherAI/gpt-neox-20b",                               "gpt_neo", "20b"),

    ("BLOOM 7B1",                          "bigscience/bloom-7b1",                                  "bloom", "7b"),
    ("BLOOMZ 7B1",                         "bigscience/bloomz-7b1",                                 "bloom", "7b"),

    ("StableLM 2 1.6B Chat",               "stabilityai/stablelm-2-zephyr-1_6b",                    "stablelm", "1b"),
    ("StableLM 2 12B Chat",                "stabilityai/stablelm-2-12b-chat",                       "stablelm", "13b"),

    # ══════════════════════════════════════════
    # MPT / VICUNA / WIZARDLM / ZEPHYR / SOLAR
    # ══════════════════════════════════════════

    ("MPT 7B Instruct",                    "mosaicml/mpt-7b-instruct",                              "mpt", "7b"),
    ("MPT 30B Instruct",                   "mosaicml/mpt-30b-instruct",                             "mpt", "30b"),

    ("Vicuna 7B v1.5",                     "lmsys/vicuna-7b-v1.5",                                  "vicuna", "7b"),
    ("Vicuna 13B v1.5",                    "lmsys/vicuna-13b-v1.5",                                 "vicuna", "13b"),

    ("WizardLM 7B v1.0",                   "WizardLM/WizardLM-7B-V1.0",                             "wizardlm", "7b"),
    ("WizardLM 70B v1.0",                  "WizardLM/WizardLM-70B-V1.0",                            "wizardlm", "70b"),

    ("Zephyr 7B Beta",                     "HuggingFaceH4/zephyr-7b-beta",                          "zephyr", "7b"),
    ("Zephyr 141B A39B",                   "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",              "zephyr", "other"),

    ("SOLAR 10.7B Instruct",               "upstage/SOLAR-10.7B-Instruct-v1.0",                     "solar", "13b"),
    ("SOLAR 10.7B Base",                   "upstage/SOLAR-10.7B-v1.0",                              "solar", "13b"),

    # ══════════════════════════════════════════
    # OPENHERMES / INTERN
    # ══════════════════════════════════════════

    ("OpenHermes 2.5 Mistral 7B",          "teknium/OpenHermes-2.5-Mistral-7B",                     "openhermes", "7b"),
    ("Nous Hermes 2 Mixtral 8x7B DPO",     "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",           "openhermes", "30b"),
    ("Nous Hermes 3 LLaMA 3.1 8B",         "NousResearch/Hermes-3-Llama-3.1-8B",                    "openhermes", "8b"),
    ("Nous Hermes 3 LLaMA 3.1 70B",        "NousResearch/Hermes-3-Llama-3.1-70B",                   "openhermes", "70b"),

    ("InternLM2.5 7B Chat",                "internlm/internlm2_5-7b-chat",                          "intern", "7b"),
    ("InternLM2.5 20B Chat",               "internlm/internlm2_5-20b-chat",                         "intern", "20b"),

    # ══════════════════════════════════════════
    # OTHER
    # ══════════════════════════════════════════

    ("SmolLM2 1.7B Instruct",              "HuggingFaceTB/SmolLM2-1.7B-Instruct",                   "other", "1b"),
    ("SmolLM2 360M Instruct",              "HuggingFaceTB/SmolLM2-360M-Instruct",                   "other", "1b"),
    ("Dolphin 2.9.4 LLaMA 3.1 8B",         "cognitivecomputations/dolphin-2.9.4-llama3.1-8b",       "other", "8b"),
    ("Dolphin 2.9 Mistral 7B",             "cognitivecomputations/dolphin-2.9-mistral-7b-v0.6",     "other", "7b"),
    ("Dolphin 3.0 LLaMA 3.1 8B",           "cognitivecomputations/dolphin-3.0-llama3.1-8b",         "other", "8b"),

    # ══════════════════════════════════════════
    # ABLITERATED MODELS (v0.73.1.3)
    # Added to match IOTA_SUBFAMILY_MAP entries
    # ══════════════════════════════════════════

    # LLaMA
    ("LLaMA 3 70B Abliterated",             "failspy/llama-3-70B-Instruct-abliterated",               "llama", "70b"),
    ("LLaMA 3.1 8B Abliterated",            "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",       "llama", "8b"),
    ("LLaMA 3.2 3B Abliterated",            "huihui-ai/Llama-3.2-3B-Instruct-abliterated",           "llama", "3b"),
    ("LLaMA 3.2 1B Abliterated",            "huihui-ai/Llama-3.2-1B-Instruct-abliterated",           "llama", "1b"),

    # Mistral
    ("Mistral 7B v0.3 Abliterated",         "evolveon/Mistral-7B-Instruct-v0.3-abliterated",         "mistral", "7b"),
    ("Mistral Nemo 12B Abliterated",        "huihui-ai/Mistral-Nemo-Instruct-2407-abliterated",      "mistral", "12b"),

    # Qwen
    ("Qwen2.5 0.5B Abliterated",            "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated",          "qwen", "0.5b"),
    ("Qwen2.5 1.5B Abliterated",            "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated",          "qwen", "1.5b"),
    ("Qwen2.5 3B Abliterated",              "huihui-ai/Qwen2.5-3B-Instruct-abliterated",             "qwen", "3b"),
    ("Qwen2.5 7B Abliterated",              "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v3",          "qwen", "7b"),
    ("Qwen2.5 14B Abliterated",             "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2",         "qwen", "14b"),
    ("Qwen2.5 32B Abliterated",             "huihui-ai/Qwen2.5-32B-Instruct-abliterated",            "qwen", "34b"),
    ("Qwen2.5 72B Abliterated",             "huihui-ai/Qwen2.5-72B-Instruct-abliterated",            "qwen", "72b"),
    ("Qwen2.5-Coder 7B Abliterated",        "huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated",      "qwen", "7b"),
    ("Qwen2.5-Coder 32B Abliterated",       "huihui-ai/Qwen2.5-Coder-32B-Instruct-abliterated",     "qwen", "34b"),

    # Gemma
    ("Gemma 2 2B Abliterated",              "IlyaGusev/gemma-2-2b-it-abliterated",                   "gemma", "2b"),
    ("Gemma 2 27B Abliterated",             "byroneverson/gemma-2-27b-it-abliterated",               "gemma", "27b"),
    ("Gemma 3 1B Abliterated",              "mlabonne/gemma-3-1b-it-abliterated",                    "gemma", "1b"),
    ("Gemma 3 4B Abliterated",              "huihui-ai/gemma-3-4b-it-abliterated",                   "gemma", "4b"),
    ("Gemma 3 27B Abliterated",             "mlabonne/gemma-3-27b-it-abliterated",                   "gemma", "27b"),
]

# ─────────────────────────────────────────────
# VARIANT RESOLUTION  (v30.0)
# User never selects a variant. These functions are called internally.
# ─────────────────────────────────────────────

_ABLITERATED_SUFFIXES = ["abliterated", "uncensored"]
_INSTRUCT_SUFFIXES    = ["instruct", "it", "chat", "zephyr", "dolphin", "hermes",
                         "wizardlm", "vicuna", "openchat", "openhermes"]
_BASE_SUFFIXES        = ["base", "pt"]

_VARIANT_PRIORITY = ["abliterated", "instruct", "base"]


# ─────────────────────────────────────────────
# IOTA SUBFAMILY MAP  (v40.0.0)
# For Run 0001 three-model C_t isolation.
# Each abliterated model maps to its correct sibling base and instruct.
# ALL THREE MUST SHARE THE SAME PRETRAINING LINEAGE.
# ─────────────────────────────────────────────
IOTA_SUBFAMILY_MAP = {
    # ══════════════════════════════════════════════════════════════════
    # LLAMA FAMILY
    # ══════════════════════════════════════════════════════════════════

    # ── LLaMA 3 8B ── (DEFAULT)
    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3": {
        "base":     "meta-llama/Meta-Llama-3-8B",
        "instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
        "subfamily": "llama-3-8b",
        "note": "Llama 3 8B -- all three from same pretraining. DEFAULT.",
    },
    # ── LLaMA 3 70B ──
    "failspy/llama-3-70B-Instruct-abliterated": {
        "base":     "meta-llama/Meta-Llama-3-70B",
        "instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        "subfamily": "llama-3-70b",
        "note": "Llama 3 70B -- failspy original. 80 layers, 8192 hidden dim.",
    },
    # ── LLaMA 3.1 8B ──
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated": {
        "base":     "meta-llama/Meta-Llama-3.1-8B",
        "instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "subfamily": "llama-3.1-8b",
        "note": "Llama 3.1 8B -- mlabonne abliteration. 32 layers, 4096 hidden dim.",
    },
    # ── LLaMA 3.2 3B ──
    "huihui-ai/Llama-3.2-3B-Instruct-abliterated": {
        "base":     "meta-llama/Llama-3.2-3B",
        "instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "subfamily": "llama-3.2-3b",
        "note": "Llama 3.2 3B -- huihui-ai abliteration. 28 layers, 3072 hidden dim.",
    },
    # ── LLaMA 3.2 1B ──
    "huihui-ai/Llama-3.2-1B-Instruct-abliterated": {
        "base":     "meta-llama/Llama-3.2-1B",
        "instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "subfamily": "llama-3.2-1b",
        "note": "Llama 3.2 1B -- huihui-ai abliteration. 16 layers, 2048 hidden dim.",
    },

    # ══════════════════════════════════════════════════════════════════
    # MISTRAL FAMILY
    # ══════════════════════════════════════════════════════════════════

    # ── Mistral 7B v0.2 ──
    "failspy/Mistral-7B-Instruct-v0.2-abliterated": {
        "base":     "mistralai/Mistral-7B-v0.3",
        "instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "subfamily": "mistral-7b-v0.2",
        "note": "Mistral 7B -- v0.2 abliterated paired with v0.3 instruct/base.",
    },
    # ── Mistral 7B v0.3 ──
    "evolveon/Mistral-7B-Instruct-v0.3-abliterated": {
        "base":     "mistralai/Mistral-7B-v0.3",
        "instruct": "mistralai/Mistral-7B-Instruct-v0.3",
        "subfamily": "mistral-7b-v0.3",
        "note": "Mistral 7B v0.3 -- evolveon abliteration. 32 layers, 4096 hidden dim.",
    },
    # ── Mistral Nemo 12B ──
    "huihui-ai/Mistral-Nemo-Instruct-2407-abliterated": {
        "base":     "mistralai/Mistral-Nemo-Base-2407",
        "instruct": "mistralai/Mistral-Nemo-Instruct-2407",
        "subfamily": "mistral-nemo-12b",
        "note": "Mistral Nemo 12B -- huihui-ai abliteration. 40 layers, 5120 hidden dim.",
    },

    # ══════════════════════════════════════════════════════════════════
    # QWEN FAMILY
    # ══════════════════════════════════════════════════════════════════

    # ── Qwen2 7B ──
    "failspy/Qwen2-7B-Instruct-abliterated-v0.1": {
        "base":     "Qwen/Qwen2.5-7B",
        "instruct": "Qwen/Qwen2.5-7B-Instruct",
        "subfamily": "qwen2-7b",
        "note": "Qwen2 abliterated -- paired with Qwen2.5 base/instruct (closest pretraining sibling).",
    },
    # ── Qwen2.5 0.5B ──
    "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated": {
        "base":     "Qwen/Qwen2.5-0.5B",
        "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        "subfamily": "qwen2.5-0.5b",
        "note": "Qwen2.5 0.5B -- huihui-ai abliteration. 24 layers, 896 hidden dim.",
    },
    # ── Qwen2.5 1.5B ──
    "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated": {
        "base":     "Qwen/Qwen2.5-1.5B",
        "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "subfamily": "qwen2.5-1.5b",
        "note": "Qwen2.5 1.5B -- huihui-ai abliteration. 28 layers, 1536 hidden dim.",
    },
    # ── Qwen2.5 3B ──
    "huihui-ai/Qwen2.5-3B-Instruct-abliterated": {
        "base":     "Qwen/Qwen2.5-3B",
        "instruct": "Qwen/Qwen2.5-3B-Instruct",
        "subfamily": "qwen2.5-3b",
        "note": "Qwen2.5 3B -- huihui-ai abliteration. 36 layers, 2048 hidden dim.",
    },
    # ── Qwen2.5 7B ──
    "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v3": {
        "base":     "Qwen/Qwen2.5-7B",
        "instruct": "Qwen/Qwen2.5-7B-Instruct",
        "subfamily": "qwen2.5-7b",
        "note": "Qwen2.5 7B -- huihui-ai v3 abliteration. 28 layers, 3584 hidden dim.",
    },
    # ── Qwen2.5 14B ──
    "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2": {
        "base":     "Qwen/Qwen2.5-14B",
        "instruct": "Qwen/Qwen2.5-14B-Instruct",
        "subfamily": "qwen2.5-14b",
        "note": "Qwen2.5 14B -- huihui-ai v2 abliteration. 48 layers, 5120 hidden dim.",
    },
    # ── Qwen2.5 32B ──
    "huihui-ai/Qwen2.5-32B-Instruct-abliterated": {
        "base":     "Qwen/Qwen2.5-32B",
        "instruct": "Qwen/Qwen2.5-32B-Instruct",
        "subfamily": "qwen2.5-32b",
        "note": "Qwen2.5 32B -- huihui-ai abliteration. 64 layers, 5120 hidden dim.",
    },
    # ── Qwen2.5 72B ──
    "huihui-ai/Qwen2.5-72B-Instruct-abliterated": {
        "base":     "Qwen/Qwen2.5-72B",
        "instruct": "Qwen/Qwen2.5-72B-Instruct",
        "subfamily": "qwen2.5-72b",
        "note": "Qwen2.5 72B -- huihui-ai abliteration. 80 layers, 8192 hidden dim.",
    },
    # ── Qwen2.5-Coder 7B ──
    "huihui-ai/Qwen2.5-Coder-7B-Instruct-abliterated": {
        "base":     "Qwen/Qwen2.5-Coder-7B",
        "instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "subfamily": "qwen2.5-coder-7b",
        "note": "Qwen2.5-Coder 7B -- huihui-ai. Continued pretraining on code data.",
    },
    # ── Qwen2.5-Coder 32B ──
    "huihui-ai/Qwen2.5-Coder-32B-Instruct-abliterated": {
        "base":     "Qwen/Qwen2.5-Coder-32B",
        "instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "subfamily": "qwen2.5-coder-32b",
        "note": "Qwen2.5-Coder 32B -- huihui-ai. Continued pretraining on code data.",
    },

    # ══════════════════════════════════════════════════════════════════
    # GEMMA FAMILY
    # ══════════════════════════════════════════════════════════════════

    # ── Gemma 2 2B ──
    "IlyaGusev/gemma-2-2b-it-abliterated": {
        "base":     "google/gemma-2-2b",
        "instruct": "google/gemma-2-2b-it",
        "subfamily": "gemma-2-2b",
        "note": "Gemma 2 2B -- IlyaGusev abliteration. 26 layers, 2304 hidden dim.",
    },
    # ── Gemma 2 9B ──
    "IlyaGusev/gemma-2-9b-it-abliterated": {
        "base":     "google/gemma-2-9b",
        "instruct": "google/gemma-2-9b-it",
        "subfamily": "gemma-2-9b",
        "note": "Gemma 2 9B -- IlyaGusev abliteration. 42 layers, 3584 hidden dim.",
    },
    # ── Gemma 2 27B ──
    "byroneverson/gemma-2-27b-it-abliterated": {
        "base":     "google/gemma-2-27b",
        "instruct": "google/gemma-2-27b-it",
        "subfamily": "gemma-2-27b",
        "note": "Gemma 2 27B -- byroneverson abliteration (CPU method). 46 layers, 4608 hidden dim.",
    },
    # ── Gemma 3 1B ──
    "mlabonne/gemma-3-1b-it-abliterated": {
        "base":     "google/gemma-3-1b-pt",
        "instruct": "google/gemma-3-1b-it",
        "subfamily": "gemma-3-1b",
        "note": "Gemma 3 1B -- mlabonne abliteration. 26 layers, 1152 hidden dim.",
    },
    # ── Gemma 3 4B ──
    "huihui-ai/gemma-3-4b-it-abliterated": {
        "base":     "google/gemma-3-4b-pt",
        "instruct": "google/gemma-3-4b-it",
        "subfamily": "gemma-3-4b",
        "note": "Gemma 3 4B -- huihui-ai abliteration. 34 layers, 2560 hidden dim.",
    },
    # ── Gemma 3 12B ──
    "mlabonne/gemma-3-12b-it-abliterated-v2": {
        "base":     "google/gemma-3-12b-pt",
        "instruct": "google/gemma-3-12b-it",
        "subfamily": "gemma-3-12b",
        "note": "Gemma 3 12B -- mlabonne v2 abliteration. 48 layers, 3840 hidden dim.",
    },
    # ── Gemma 3 27B ──
    "mlabonne/gemma-3-27b-it-abliterated": {
        "base":     "google/gemma-3-27b-pt",
        "instruct": "google/gemma-3-27b-it",
        "subfamily": "gemma-3-27b",
        "note": "Gemma 3 27B -- mlabonne abliteration. 62 layers, 4608 hidden dim.",
    },

    # ══════════════════════════════════════════════════════════════════
    # PHI FAMILY
    # ══════════════════════════════════════════════════════════════════

    # ── Phi-3 Mini ──
    "failspy/Phi-3-mini-128k-instruct-abliterated": {
        "base":     "microsoft/Phi-3-mini-128k-instruct",
        "instruct": "microsoft/Phi-3-mini-128k-instruct",
        "subfamily": "phi-3-mini",
        "note": "Phi-3 Mini -- no separate base; instruct used for both. ⚠️ INVALID TRIPLET.",
    },
}

def get_subfamily_models(abliterated_path: str) -> dict:
    """
    Return the full subfamily dict for a given abliterated model path.
    Keys: 'base', 'instruct', 'subfamily', 'note'.
    Returns None if the abliterated path is not in IOTA_SUBFAMILY_MAP.

    This is the only supported way to resolve base and instruct siblings
    for Run 0001. Do NOT use resolve_variant_path for this purpose -- it
    returns the first matching variant in ALL_MODELS regardless of lineage,
    which may silently mix subfamilies (e.g. Llama 3 abliterated resolved
    with Llama 3.1 base -- invalid for difference vector computation).
    """
    return IOTA_SUBFAMILY_MAP.get(abliterated_path, None)


def _classify_variant(path: str) -> str:
    """Return 'abliterated' | 'instruct' | 'base' | 'unknown'."""
    low = path.lower()
    for s in _ABLITERATED_SUFFIXES:
        if s in low:
            return "abliterated"
    for s in _INSTRUCT_SUFFIXES:
        if s in low:
            return "instruct"
    for s in _BASE_SUFFIXES:
        if s in low:
            return "base"
    return "unknown"


def get_preferred_variant(family: str, size: str) -> tuple:
    """
    Return (display_name, hf_path, variant_key) for the preferred variant
    of a given family/size.

    Priority: abliterated > instruct > base > any
    Returns ('', '', '') if nothing found for this family/size.
    """
    candidates = [(n, p) for n, p, f, s in ALL_MODELS
                  if f == family and s == size]
    if not candidates:
        return "", "", ""

    by_variant: dict = {"abliterated": [], "instruct": [], "base": [], "unknown": []}
    for name, path in candidates:
        v = _classify_variant(path)
        by_variant[v].append((name, path))

    for v in _VARIANT_PRIORITY:
        if by_variant[v]:
            name, path = by_variant[v][0]
            return name, path, v

    name, path = candidates[0]
    return name, path, _classify_variant(path)


def resolve_variant_path(family: str, size: str, variant_key: str) -> tuple:
    """
    Return (display_name, hf_path) for an explicit variant of family/size.
    Falls back to get_preferred_variant() if exact match not found.
    """
    candidates = [(n, p) for n, p, f, s in ALL_MODELS
                  if f == family and s == size]
    for name, path in candidates:
        if _classify_variant(path) == variant_key:
            return name, path
    name, path, _ = get_preferred_variant(family, size)
    return name, path


def get_family_sizes(family_key: str) -> list:
    """Return size buckets available for this family, in SIZE_ORDER order."""
    sizes = sorted(
        {s for _, _, f, s in ALL_MODELS if f == family_key},
        key=lambda x: SIZE_ORDER.index(x) if x in SIZE_ORDER else 999
    )
    return sizes


# ─────────────────────────────────────────────
# SEARCH / BROWSE HELPERS
# ─────────────────────────────────────────────


def by_family(family_key: str) -> list:
    """Return [(name, path)] for all models of given family (deduplicated)."""
    seen   = set()
    result = []
    for n, p, f, s in ALL_MODELS:
        if f == family_key and p not in seen:
            seen.add(p)
            result.append((n, p))
    return result




def get_by_path(path: str) -> tuple:
    """Find a model in the curated list by HF path.
    Returns (name, path, family, size) -- infers family/size if not in catalog."""
    for n, p, f, s in ALL_MODELS:
        if p == path:
            return n, p, f, s
    name = path.split("/")[-1]
    low  = path.lower()
    fam  = "other"
    for key in FAMILIES:
        if key in low:
            fam = key
            break
    return name, path, fam, "other"


