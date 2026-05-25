# Model Configs

These configs cover two groups:

- `paper_*`: models used by the official TRAP experiments.
- `server_*`: models reported as available under `/home/chj/LLMJailbreak/models`.

The official TRAP paper evaluates suffixes optimized for Llama-2-chat, Vicuna, Guanaco, and a Vicuna/Guanaco ensemble. The unified framework currently optimizes one model at a time, so the ensemble setting is not represented as a single model config.

Only `paper_llama2_7b_chat.yaml` is expected to exist on the current remote server from the official TRAP list. The other `paper_*` configs are included as named placeholders with conventional local paths; download or symlink those model folders before running them.

For the server models, use the `server_*` configs as suspect-model configs when estimating transfer or false positive behavior against the TRAP fingerprints.
