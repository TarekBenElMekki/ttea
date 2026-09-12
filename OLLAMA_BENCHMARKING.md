# Controlled Ollama Benchmark Runtime

T/TEA launches a dedicated benchmark-owned `ollama serve` process on a validated loopback-only endpoint, defaulting to `127.0.0.1:11435`. It does not reuse or control a desktop/tray server on the default port. The owned subprocess alone receives `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_QUEUE=1`, an initial context limit of 4096, `OLLAMA_KEEP_ALIVE=0`, and `OLLAMA_NO_CLOUD=1`; global environment variables are unchanged.

Every HTTP and process operation is bounded. Short control-plane timeouts cover health, discovery, and unload operations separately from the generation request timeout. Generation is one model, one prompt, one non-streaming request with explicit context, output, and CPU-thread limits plus `keep_alive=0`. The model must already be installed locally: benchmark execution never pulls or downloads models.

The benchmark worker PID and owned Ollama server PID are separate persisted identities. The server and each descendant are tracked by PID and process creation time; cleanup never searches or kills by process name. The emergency PowerShell stop covers the verified worker and owned server tree. Cleanup attempts exact-model API unload, then `ollama stop <exact model>` against the dedicated server when needed, and finally verified shutdown of only the owned server process tree. Ownership state is removed only after both worker and server shutdown are confirmed.

`ResourceManager` preflight runs before server startup, and `BenchmarkSupervisor` performs a second fresh preflight immediately before generation. The controlled runtime cannot grant itself permission and includes the server-side model allocation in the supervisor cleanup hierarchy. Ctrl+C before or during supervised execution triggers bounded cleanup and preserves ownership state whenever server shutdown cannot be confirmed.

`python -m ttea.benchmarking --ollama-check` performs passive executable and default-loopback API discovery only. It does not launch a server, pull a model, or generate text.

T/TEA resolves Ollama from an explicit configured executable, PATH, or the standard current-user Windows installation location without modifying the user's global PATH.
