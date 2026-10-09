# Configuration reference

Environment variables configure decision models, the comparison chat model, logging paths, driver binaries, and evaluation datasets.

Variables can be exported in your shell or placed in a `.env` file at the root of the repository. Exported environment variables take precedence over values in `.env`.

## Environment variables

| Variable | Who reads it | Default | What it does |
|:---|:---|:---|:---|
| `TYPESAFE_API_KEY` | `jev` model | *(unset)* | API key for direct TypeSafe decisions endpoint (`https://api.typesafe.ai/v1/systemone`). |
| `TYPESAFE_API_URL` | `jev` model | `https://openrouter.ai/api/alpha/decisions` | Endpoint URL for decisions; defaults to OpenRouter proxy, or can be overridden to a custom proxy URL. |
| `TYPESAFE_MODEL` | `jev` model | `typesafe/jev-1.13` | Model identifier when proxying Jev decisions through OpenRouter. |
| `S1A_DECISION_TIMEOUT_S` | `jev` model | `5` | Seconds one decision may take, retries included. Raise it for a slower System One server behind `TYPESAFE_API_URL` (e.g. a local OneJev-27B or CLM-8B: `30`). |
| `OPENROUTER_API_KEY` | `jev` model (proxy), chat model fallback | *(unset)* | OpenRouter API key, used for proxying Jev decisions or as a fallback for `LLM_API_KEY`. |
| `OPENROUTER_BASE_URL` | chat model fallback | `https://openrouter.ai/api/v1` | Fallback base URL for the chat model when `LLM_BASE_URL` or `OPENAI_BASE_URL` is unset. |
| `MODEL_NAME` | chat model (`llm` model, rethink planner, browser agent) | *(unset)* | Model identifier for the chat model (e.g. `google/gemini-2.5-flash` or `claude-fable-5-1`). |
| `MODEL_PROVIDER` | chat model | `openai` | Provider protocol for the chat model (`openai` or `anthropic`). |
| `OPENAI_API_KEY` | chat model | *(unset)* | API key for OpenAI-compatible endpoints; checked alongside alias `LLM_API_KEY`. |
| `OPENAI_BASE_URL` | chat model | `https://api.openai.com/v1` | Base endpoint URL for OpenAI-compatible chat models; checked alongside alias `LLM_BASE_URL`. |
| `LLM_API_KEY` | chat model | *(unset)* | Alias for `OPENAI_API_KEY`; falls back to `OPENROUTER_API_KEY` when unset. |
| `LLM_BASE_URL` | chat model | *(unset)* | Alias for `OPENAI_BASE_URL`; falls back to `OPENROUTER_BASE_URL` when unset. |
| `ANTHROPIC_WORKSPACE_ID` | chat model | *(unset)* | Organization workspace identifier passed in the `anthropic-workspace-id` header when `MODEL_PROVIDER=anthropic`. |
| `S1A_HOME` | runtime, console, logging | Repository root | Root directory where agent runs, logs, and artifacts are written (`runs/logs/`). |
| `PLAYWRIGHT_MCP_ARGS` | browser agents (`allrecipes`, `flights`), `scripts/browser_showcase.sh` | `npx -y @playwright/mcp@0.0.78` | Command arguments passed when launching Playwright MCP, or custom flags like `--cdp-endpoint`. |
| `PLAYWRIGHT_MCP_COMMAND` | `scripts/browser_showcase.sh` | `node` | Executable used to launch the Playwright MCP server process in browser showcase scripts. |
| `ALFWORLD_DATA` | `alfworld` agent, replay | *(unset)* | Directory containing the downloaded ALFWorld benchmark dataset and game files. |
| `CHAT_USD_PER_M_INPUT` | pricing / accounting | *(catalogue)* | Price in USD per million prompt tokens when `MODEL_NAME` is not in OpenRouter's public catalogue. |
| `CHAT_USD_PER_M_OUTPUT` | pricing / accounting | *(catalogue)* | Price in USD per million completion tokens when `MODEL_NAME` is not in OpenRouter's public catalogue. |
| `CHAT_USD_PER_M_CACHED_INPUT` | pricing / accounting | `CHAT_USD_PER_M_INPUT` | Price in USD per million cached input tokens for non-catalogue models. |
| `LAYA_MODEL` | `laya` model | `convaiinnovations/laya` | Hugging Face repository ID or local path for the resident Laya decision model checkpoint. |
| `LAYA_SUBFOLDER` | `laya` model | *(unset)* | Optional subfolder in the checkpoint repo (e.g. `multilingual` or `typed-decisions`). |
| `LAYA_DEVICE` | `laya` model | `(library default)` | PyTorch device for Laya model evaluation; passes None so the library selects CUDA, MPS, or CPU. |
| `LAYA_MAX_LEN` | `laya` model | `(checkpoint default)` | Maximum token sequence length for Laya state representation; overrides checkpoint window only when set. Browser agents want `1536`. |
| `LAYA_HEAD_MAX_LEN` | `laya` model | `(checkpoint default)` | Maximum token sequence length for Laya decision head options; overrides checkpoint window only when set. Browser agents want `1024`. |
| `LAYA_COMPACT_BROWSER_STATE` | `laya` model | `1` | Folds a browser-front state and its questions to fit Laya's window (`laya_state`, `laya_browser_question`); `0`, `false` or `no` sends them as Jev gets them. |
| `CLM_URL` | `clm` model | *(unset, required)* | Base URL of a served CLM: `clm-serve` (`http://127.0.0.1:8091`). CLM's engine is a client of an `/v1/embeddings` server, so `clm-serve` needs one beside it; system1-omni's `recipe/clm/README.md` runs both. |
| `CLM_MODEL` | `clm` model | `clm-latest` | Name of the served model to ask: `clm-latest`, or `clm-raw` for the ablation that scores in the raw encoder space with no projection head. |
| `CLM_API_KEY` | `clm` model | *(unset)* | Bearer token, when `clm-serve` was started with one. |
| `CLM_TIMEOUT_S` | `clm` model | `30` | Deadline per decision in seconds, the one refused-connection retry included. A finite number above 0: `nan`, `inf` and `0` are configuration errors. Longer than the other served models because a cold candidate set is one 8B forward pass. |
| `LAYA_MPS_AMP_MIN_ROWS` | `laya` model | *(unset: fp32)* | Laya's own variable: on MPS, requests with at least this many questions run in fp16. Unset, `--model laya` keeps every request in fp32, as on CPU; `5` is Laya's default. fp16 moves the probabilities and can flip a close decision. A value that is not a whole number of at least 1 is a configuration error. |
| `LAYA_SERVED_URL` | `laya-served` model | *(unset, required)* | Base URL of a served Laya: the system1-omni worker (`http://127.0.0.1:8000`), its `omni-jev` frontend (`:8080`) or plain laya-serve. |
| `LAYA_SERVED_MODEL` | `laya-served` model | `english` | Name of the served checkpoint to ask: `english`, `multilingual` or `typed-decisions`. A name the server does not know raises instead of being routed by language. |
| `LAYA_SERVED_API_KEY` | `laya-served` model | *(unset)* | Bearer token, the server's `LAYA_API_KEY` when it sets one. |
| `LAYA_SERVED_TIMEOUT_S` | `laya-served` model | `5` | Deadline per decision in seconds, the one retry and any `/health` refresh included. A finite number above 0: `nan`, `inf` and `0` are configuration errors. |
| `LAYA_SERVED_MAX_LEN` | `laya-served` model | `512` | The served checkpoint's token window per question (`english` 512, `multilingual` 1024); a request that fills it raises. The server takes the window from the checkpoint, so set this to match and nothing higher. |
| `CUA_S1_CHECKPOINT` | `cua` model | `cua-ai/cua-s1-nano-0.1` | Hugging Face checkpoint ID or local directory for Cua-S1 Nano option scorer. |
| `CUA_S1_SUBFOLDER` | `cua` model | `text` | Subfolder within checkpoint directory containing text option scoring weights. |
| `CUA_S1_DEVICE` | `cua` model | `auto` | PyTorch device used for Cua-S1 Nano evaluation (`auto`, `cpu`, `cuda`, or `mps`). |
| `CUA_DRIVER_BIN` | `desktop` agent | `cua-driver` | Path to the `cua-driver` executable on Windows or macOS when not located on `PATH`. |
| `CUA_DRIVER_PERMISSION_MODE` | `desktop` agent | `standard` | Permission mode passed to `cua-driver mcp` (`standard`, or `bounded` for restricted capability manifests). |
| `HF_HOME` | Hugging Face runtime | `~/.cache/huggingface` | Cache directory where Laya and Cua-S1 checkpoints are downloaded on first run. |
| `HF_HUB_OFFLINE` | Hugging Face runtime | `0` | When set to `1`, prevents network requests and forces models to load exclusively from local cache. |
