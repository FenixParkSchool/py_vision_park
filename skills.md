# AI Development Guide — License Plate Recognition Microservice

> **Purpose of this file:** These instructions guide AI coding assistants, reviewers, and contributors working on this repository. Follow them together with the actual source code, project configuration, tests, and the team's agreed integration contract.
>
> **Primary rule:** Build the real project incrementally while helping the student understand each file, design decision, library, and test. Do not replace development with disconnected exercises, vague advice, or a complete application dumped all at once.

---

## 1. Project Brief

### 1.1 What this service does

This repository contains **only the Python computer-vision and machine-learning microservice** for a parking-lot system.

Its principal responsibility is to observe a video source, detect vehicles' license plates, recognize the plate characters, apply recognition-quality and duplicate-event rules, and send the agreed recognition data to a separate backend owned by another team.

The expected high-level flow is:

```text
Camera / recorded video
        |
        v
Frame capture and sampling (OpenCV)
        |
        v
Image preparation (OpenCV / NumPy)
        |
        v
License-plate detection
        |
        v
Plate crop and character recognition (TensorFlow model, when selected)
        |
        v
Post-processing and recognition-quality checks
        |
        v
Short-window duplicate suppression
        |
        v
Recognition event / agreed payload
        |
        v
HTTPX client ------HTTP------> External Node.js / Express backend
```

The service also uses FastAPI and Uvicorn for its own operational HTTP API, health checks, and any explicitly required diagnostic endpoints. The continuous camera-processing loop must not be implemented as a long-running HTTP request handler.

### 1.2 Ownership boundary

The Python team owns the vision service and its communication client. A separate team owns the main application backend.

| This repository owns | This repository does **not** own |
|---|---|
| Reading frames from a configured camera, webcam, or test video | Visitor registration or visitor CRUD operations |
| Frame sampling and image preprocessing | Vehicle or visitor database schema |
| License-plate detection | Reservations, parking permissions, entry authorization, or business rules owned by the backend |
| OCR / plate-character recognition | Prisma models, database migrations, or PostgreSQL persistence for the main system |
| Recognition confidence, normalization, and quality checks | The main Node.js / Express API implementation |
| Suppressing short-window duplicate observations | Deciding what a recognition means for parking-system business workflows |
| Building and sending recognition events to an agreed backend endpoint | Inventing or unilaterally changing the backend's endpoint or payload contract |
| Service health, diagnostics, tests, configuration, and deployment of this service | Reimplementing backend business logic in Python |

**Do not create database access, ORM models, visitor services, reservation logic, or a second copy of the main backend in this repository unless the scope is explicitly changed by the project owner.**

### 1.3 Current technology stack

The declared stack is:

- Python 3.11
- `uv` for project/dependency/environment management
- FastAPI
- Uvicorn
- Pydantic
- `pydantic-settings`
- HTTPX
- `python-multipart` when multipart file uploads are actually needed
- OpenCV
- NumPy
- TensorFlow
- Pytest

Do not replace these technologies casually. Do not add another framework, queue system, ORM, database, or ML framework merely because it is common in other applications. Propose an addition only when there is a concrete requirement and explain its cost.

### 1.4 Known integration facts and unknowns

Known:

- A separate Node.js / Express backend will receive the recognition data.
- The vision service must be able to read a recorded video during development before a live parking-lot camera is available.
- The intended eventual sources include a webcam and/or an IP camera using RTSP, depending on the available hardware and network.
- The service uses Python 3.11.
- The project should be built through practical, incremental work, with explanation alongside implementation.

Not yet guaranteed unless confirmed in the repository or by the team:

- The final plate-detection model or algorithm.
- The OCR model and its input/output tensor contract.
- Whether model weights are already present, must be trained, or must be obtained from an approved source.
- The final backend URL, HTTP method, route, authentication method, payload schema, and error semantics.
- Exact confidence thresholds, sampling interval, duplicate-suppression window, retention rules, and production performance targets.
- Whether the API process and camera worker will be separate processes in production.

**Never silently convert an unknown into a fact.** Inspect the repository and ask one focused question when a missing decision blocks safe progress. Otherwise, use a clearly labeled temporary interface or placeholder and keep it easy to change.

---

## 2. Instructions for the AI Assistant

### 2.1 Work in the actual project

- Inspect the current repository before proposing changes. Read the relevant source files, `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, README, and tests where present.
- Treat the current repository state as authoritative. Earlier conversation, an example directory tree, or a previous answer does not prove that a file currently exists.
- Check existing naming, imports, routes, configuration conventions, and dependency versions before changing them.
- Prefer a small change that fits the existing codebase over a large, speculative refactor.
- Never claim to have inspected, edited, executed, or tested something unless that action actually happened.
- Do not assume the project is in a clean Git state. Avoid overwriting or reverting existing user work.

### 2.2 Pair-programming and learning method

The student wants to **make progress on the real deliverable while understanding the implementation**. Learning should be integrated into practical development; it is not a substitute for delivering the project.

For each coherent task:

1. **State the practical goal.** Explain what capability the next change will add and how it fits into the service.
2. **Explain the design choice.** Describe why the file or component belongs where it does, what it depends on, and what it should not do.
3. **Work in a bounded increment.** Implement or guide one cohesive slice, not the whole microservice in one response.
4. **Explain the relevant code.** Describe unfamiliar syntax, library APIs, important control flow, error handling, and lifecycle behavior at the point where they are used.
5. **Give a concrete verification step.** Use a real command, test, observable output, or acceptance criterion tied to the current project.
6. **Review the result.** When the student writes code, review that code directly, explain errors and trade-offs, and help fix it rather than replacing it without explanation.
7. **Identify the next step.** Recommend one logical continuation based on the actual state of the project.

Do not give vague directions such as “implement the camera,” “add error handling,” or “write tests” without specifying expected behavior, the relevant file or interface, edge cases, and how success will be verified.

Do not dump every finished file at once when the student is trying to write and understand the implementation. Provide the smallest useful code excerpt or a focused patch when needed. If the student explicitly asks for a complete implementation, deliver it with an explanation and tests appropriate to the request.

Do not force unrelated textbook exercises. Any exercise or experiment should help implement, validate, or diagnose a real component of this service.

### 2.3 Communication style

- Be direct, technically precise, and constructive.
- Avoid generic praise, filler, inflated claims, and canned “AI-sounding” prose.
- Do not use profanity.
- Explain important terms in plain language, then use the correct technical term consistently.
- Do not repeat questions whose answers are already in the conversation or repository.
- If a genuine blocker requires clarification, ask one concise, specific question. Otherwise, make a reversible, clearly documented assumption and continue.
- Distinguish observed facts, recommendations, temporary assumptions, and unresolved decisions.
- Do not overwhelm the student with every possible design pattern. Explain the simplest design that meets the actual requirements and mention alternatives only where they matter.

### 2.4 Before changing code

The assistant should establish:

- Which user-facing or internal behavior is being added or corrected.
- Which files currently own that behavior.
- Which inputs, outputs, and errors the component must handle.
- Whether a corresponding interface or test already exists.
- Whether this change affects the backend integration contract.
- How the change will be tested without requiring a real camera or external service, where possible.

For changes affecting multiple layers, give a short plan before editing. Keep the plan proportional to the task.

### 2.5 After changing code

Report:

- What changed, including exact files.
- Why the change belongs in those files.
- Important implementation choices and any assumptions.
- Commands and tests actually run, with their actual outcomes.
- Tests that could not be run and why.
- Known limitations or integration decisions still needed.
- The next practical step.

Never state “all tests pass” unless the relevant tests were actually executed successfully. Never disguise a skipped test as a passing test.

---

## 3. Architecture Principles

### 3.1 Recommended logical structure

Use this as a **target organization**, not an instruction to perform a wholesale refactor. First inspect the real repository, then introduce directories gradually as the corresponding responsibilities are implemented.

```text
vision-service/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Create/configure the FastAPI application
│   ├── worker.py                # Entry point or lifecycle adapter for continuous processing
│   ├── config.py                # Validated environment-based settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py        # Liveness/readiness/diagnostics as needed
│   │       └── recognition.py   # Optional image-upload/test endpoint
│   ├── camera/
│   │   ├── __init__.py
│   │   └── source.py            # Video-source abstraction and frame lifecycle
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Convert and prepare images for the selected model
│   │   ├── detector.py           # Locate plate regions
│   │   ├── ocr.py                # Recognize plate characters
│   │   └── postprocessing.py     # Normalize/check predictions and quality
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Coordinate a frame through recognition stages
│   │   ├── deduplicator.py       # Suppress repeated observations over a window
│   │   └── backend_client.py     # Send events to the agreed backend contract
│   └── schemas/
│       ├── __init__.py
│       └── recognition.py        # Typed internal/API data contracts
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── experiments/                 # Small, reproducible experiments and diagnostics
├── models/                      # Model artifacts or documented model retrieval rules
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── Dockerfile                   # Add when deployment is in scope
└── README.md
```

Some repositories use a `src/` layout rather than an `app/` package at the root. Either can be valid. Follow the existing structure unless there is a concrete reason to change it. Do not create duplicate packages such as both `app/recognition` and `src/vision_service/recognition` without a clear migration plan.

### 3.2 Responsibility boundaries

- **`main.py`:** constructs the FastAPI app, registers routers, and owns app-level lifecycle setup. It should remain small.
- **`config.py`:** defines validated settings and defaults. It must not open the camera or perform inference.
- **`camera/source.py`:** owns camera/video opening, frame reading, source-specific handling, and release. It must not know about backend payloads or parking business rules.
- **`recognition/`:** owns image preparation and model-level plate recognition. These functions should be testable using images/arrays without starting FastAPI.
- **`services/pipeline.py`:** coordinates the steps of one frame or one observation. It should not duplicate preprocessing, model, or HTTP-client implementation details.
- **`services/deduplicator.py`:** handles a clearly defined duplicate-suppression policy. It does not decide parking authorization.
- **`services/backend_client.py`:** owns HTTP communication, payload serialization, timeouts, authentication configuration, and response/error mapping. It should not run the vision model.
- **`schemas/`:** defines explicit shapes for data crossing component boundaries. Keep external API schemas separate from model-specific tensors and raw OpenCV arrays.
- **`api/routes/`:** translates HTTP requests into calls to application functions and formats HTTP responses. It should not contain long camera loops or machine-learning logic.
- **`worker.py`:** starts continuous processing and handles a graceful stop. Its exact design may be a separate process or an app lifecycle-managed task, depending on deployment needs.

A component may initially be implemented in a simpler location if that matches the repository state. Refactor only when the separation provides a real benefit and can be tested.

### 3.3 Pragmatic architecture, not architecture for its own sake

Use explicit functions, data structures, and small classes where they improve clarity, resource management, or testability. Do not create interfaces, abstract base classes, factories, dependency-injection frameworks, repositories, or generic plugin systems solely to imitate an architecture diagram.

An abstraction is especially useful where a real variation exists:

- Video file vs. webcam vs. RTSP source.
- Real model vs. deterministic fake model in tests.
- Real HTTP backend vs. mock sender in tests.

Avoid abstractions that have only one implementation and no testing, lifecycle, or substitution benefit unless a concrete near-term requirement justifies them.

### 3.4 Dependencies point inward

Prefer this dependency direction:

```text
FastAPI routes ──> application/pipeline ──> recognition logic
                                  ├───────> camera abstraction
                                  └───────> backend client interface

OpenCV / TensorFlow / HTTPX are implementation details at the edges.
```

Recognition functions should not import FastAPI to return a result. Camera capture should not send HTTP requests. The backend client should not import TensorFlow. Keep the boundaries simple and observable.

---

## 4. Python and Dependency Management

### 4.1 Python version

- Target Python **3.11** as specified by the project.
- Check `pyproject.toml` and `.python-version` (if present) before changing version declarations.
- Do not introduce syntax or standard-library APIs unavailable in Python 3.11.
- Check TensorFlow, OpenCV, NumPy, and platform compatibility together before selecting or upgrading versions. Availability can differ by operating system, CPU architecture, and runtime environment.
- Do not claim GPU acceleration is available unless the environment has been checked and the selected TensorFlow build supports it.

### 4.2 Use `uv` as the source of dependency-management truth

- Use `pyproject.toml` to declare project metadata and dependencies.
- Keep `uv.lock` committed when this is a `uv`-managed project.
- Use `uv add <package>` to add a dependency and `uv remove <package>` to remove one.
- Use `uv sync` to synchronize the environment.
- Use `uv run <command>` to run project commands in the managed environment.
- Do not manually edit `uv.lock`.
- Do not introduce `pip install` instructions as the primary workflow unless the repository's actual setup requires a compatibility path.
- Do not maintain a separate manually curated `requirements.txt` as a second source of truth unless the project/deployment environment requires it. If an export is needed, generate it from the lockfile and document why.
- Avoid dependency updates unrelated to the current task. Dependency changes should be narrow and tested.

Common commands, adjusted to the actual repository layout:

```bash
uv sync
uv run pytest -q
uv run pytest tests/unit -q
uv run uvicorn app.main:app --reload
```

If the import path or test directory differs, use the actual project configuration rather than copying these commands blindly. `--reload` is for local development, not production.

### 4.3 Dependency selection rules

Before adding a library:

1. Confirm there is a real requirement.
2. Check whether the current stack already solves it.
3. Review the official documentation and Python 3.11/platform compatibility.
4. Explain the operational cost and why it is justified.
5. Add it through `uv` and verify the resulting lock/environment.

Do not add Celery, Redis, Kafka, RabbitMQ, a database, a second web framework, a second ML framework, or a large plate-recognition package without a demonstrated need and explicit discussion.

### 4.4 Version uncertainty

Do not write guessed version pins into the project. Inspect the current lockfile first. If compatibility needs research, consult official documentation and release/installation notes, record the sources or decision in the relevant documentation, and make the smallest compatible change.

---

## 5. Configuration and Environment

### 5.1 General rules

Use Pydantic Settings for configuration instead of scattering `os.getenv()` calls throughout the codebase. Parse and validate configuration at a clear boundary. Components should receive the settings or the specific values they need rather than independently loading environment variables.

Recommended naming convention, subject to existing repository conventions:

```text
VISION_CAMERA_SOURCE
VISION_SAMPLE_INTERVAL_SECONDS
VISION_BACKEND_BASE_URL
VISION_BACKEND_RECOGNITION_PATH
VISION_BACKEND_TIMEOUT_SECONDS
VISION_CAMERA_ID
VISION_MODEL_PATH
VISION_LOG_LEVEL
```

These are suggested names, not an agreed backend contract. Keep existing configuration names if they are already used, unless a focused migration is justified.

### 5.2 Source configuration

The camera source must be configurable without editing Python code. It may represent:

- A numeric camera index such as `0`.
- A path to a local test video.
- An RTSP URL or another explicitly supported OpenCV source.

Do not infer that every string containing digits is a valid source. Make source parsing explicit, validate invalid values clearly, and document examples. Never log a full RTSP URL when it may contain credentials.

### 5.3 Settings design

- Use typed fields and validation for values such as sample intervals, timeouts, and confidence thresholds.
- Reject nonsensical settings such as a non-positive sample interval or negative timeout when that value is not supported.
- Use a clear and consistent environment prefix, such as `VISION_`, if appropriate for the repository.
- Keep `.env` loading explicit and predictable.
- Avoid expensive side effects when settings are imported. Loading a model or opening a camera must not happen just because another module imports `settings`.
- If using a cached settings factory, make the behavior easy to override in tests.

### 5.4 Secrets and `.env`

- `.env` must be ignored by Git.
- `.env.example` should document variable names and safe example values, not real credentials or access tokens.
- Never commit camera passwords, backend API tokens, private RTSP URLs, or production credentials.
- If a secret has been committed, do not just remove it from the latest file; alert the project owner so it can be rotated and handled according to the repository's policy.

### 5.5 Defaults and thresholds

Do not choose production thresholds by intuition alone. Thresholds for detector confidence, OCR confidence, duplicate windows, sampling, timeouts, and reconnection must be:

1. Named and configurable where appropriate.
2. Explained in terms of the component they control.
3. Tested with representative data.
4. Recorded as provisional until measured and accepted.

A sample value may be used for local development, but label it as a development default, not a validated production value.

---

## 6. Camera Input and Frame Lifecycle

### 6.1 `CameraSource` contract

The rest of the application should be able to request frames without knowing how the source is implemented. Define and document the small interface needed by the pipeline. A source should expose behavior equivalent to:

- Open the configured source.
- Confirm that it opened successfully.
- Read frames in a controlled loop or generator.
- Report useful source metadata if required (for example, FPS where reliable).
- Stop promptly when requested.
- Release the underlying capture resource on normal exit and exceptions.

Do not expose `cv2.VideoCapture` to unrelated modules if a smaller abstraction can contain it.

### 6.2 OpenCV capture rules

- Validate `VideoCapture.isOpened()` after opening a source.
- Check the success flag returned by `read()` before using the frame.
- Treat a failed read as a meaningful event. For a file, it commonly means end of file; for a live source, it may indicate disconnection or a transient read failure. Do not pretend these are always the same situation.
- Use `try/finally` or an equally reliable resource-management mechanism so `release()` runs when processing stops or fails.
- Handle an empty or invalid frame defensively before preprocessing.
- Include the source identifier in operational logs without leaking credentials.
- Avoid opening the camera on module import.

### 6.3 Source types and portability

A numeric environment value may need to be converted to a camera index, while a path or URL must remain a string. Make this conversion an explicit, tested decision. Handle camera-index values intentionally; do not let a malformed URL be silently transformed into an integer.

Backend capture flags differ across operating systems and OpenCV builds. Use platform-specific flags only when there is a demonstrated need, and isolate them so the rest of the service remains portable.

### 6.4 Frame sampling semantics

The team must understand what “analyze one frame every N seconds” means for each source type.

- For a recorded file, frame count and media FPS can be used to select frames based on **video time**. A file may decode much faster than real time.
- For a live camera, wall-clock scheduling or controlled frame sampling may be more appropriate, depending on latency and throughput requirements.
- Some devices report unreliable or zero FPS values. Do not divide by zero or assume metadata is always valid.
- If using `N = round(fps * interval)`, validate FPS, clamp the stride to at least one, and document that the method selects frames by frame count.
- Do not add `sleep()` to a file-processing loop accidentally. Sleeping simulates real time; it is not necessary when the goal is fast offline analysis.
- If a common sampling policy is required for files and live sources, explicitly define whether it follows video timestamps, wall-clock time, or a target processing rate. Test this policy.

Do not optimize sampling until the expected latency/throughput and the model's processing cost are understood. Do not analyze every frame by default without considering the cost of model inference.

### 6.5 Shutdown, reconnect, and camera ownership

- The application must stop capture cleanly on shutdown or cancellation.
- Do not create two readers for the same physical camera unintentionally.
- If API lifespan starts the camera worker, account for Uvicorn reload and multi-worker behavior: each process can execute startup logic. Do not use multiple Uvicorn worker processes to start one physical camera reader without an explicit ownership design.
- A simple first version may stop on a live-source failure and expose a failed readiness state. Reconnection with bounded backoff can be added when specified.
- If reconnecting, do not make an infinite tight loop. Use bounded or exponential delays, logs with appropriate severity, and a shutdown mechanism that remains responsive.
- Make clear whether the worker is single-camera or multi-camera. Do not silently introduce a list of camera workers or concurrent capture threads without a requirement.

---

## 7. Computer Vision and Machine Learning

### 7.1 Separate detection from OCR

Keep these conceptual steps distinct even if the chosen model combines some of them internally:

1. **Detection:** locate the license-plate region in a frame.
2. **Cropping and preparation:** validate and crop the region; prepare pixels in the exact format expected by the selected model.
3. **OCR / recognition:** convert the crop into a character sequence.
4. **Post-processing:** normalize and assess the predicted result without manufacturing unsupported certainty.
5. **Decision:** accept, reject, or mark the observation as uncertain based on a documented policy.

Do not write model-dependent logic until the model's actual input shape, color order, normalization, outputs, and licensing are known.

### 7.2 Model selection and provenance

The detector and OCR model have not been fixed by this file. Before selecting or implementing them, inspect the repository and discuss:

- Whether a model or weights already exist.
- Whether the project requires training, fine-tuning, or inference only.
- Whether the available dataset is permitted for the project's use and represents the target conditions.
- Whether the approach supports Brazilian plate formats expected by the project.
- Whether the runtime environment can run the chosen TensorFlow model.
- How large the model files are and how they will be delivered/deployed.
- Whether the model's license permits the intended use.

Do not pretend that a detector or OCR model exists when it does not. Do not download weights or datasets from an unverified source and silently incorporate them. Record the model source, version, preprocessing requirements, expected outputs, and applicable license in project documentation.

### 7.3 Array and image conventions

OpenCV commonly reads color images in BGR order, while a model may require RGB. Never assume the model accepts OpenCV's raw output. The preprocessing code must explicitly document and test:

- Shape and dimension ordering.
- Color-channel ordering.
- Data type (for example, integer pixels vs. floating-point values).
- Normalization/scaling method.
- Resize and aspect-ratio handling.
- Required batch dimension.
- Any grayscale conversion or channel count requirement.

These details must match the selected model; do not apply generic normalization such as dividing by 255 unless the model contract calls for it.

### 7.4 Bounding boxes and crops

- Validate model output before using coordinates.
- Clip crop boundaries to image dimensions.
- Reject boxes with invalid, reversed, zero-sized, or unusably small dimensions as appropriate.
- Handle frames with no detected plate as normal, expected outcomes—not necessarily exceptional failures.
- Be careful with coordinate conventions (absolute pixels vs. normalized values and `x/y` ordering).
- If multiple detections are returned, apply the agreed ranking, suppression, or selection rule. Do not automatically choose the first result without checking what the model output represents.
- Avoid repeatedly copying large arrays if unnecessary, but prioritize correctness and readable code before premature optimization.

### 7.5 TensorFlow lifecycle and inference

- Avoid loading the same model once for every frame or every API request. Load it once per worker/process lifecycle where appropriate.
- Do not load heavyweight ML artifacts at module import unless the project explicitly accepts that startup/test cost.
- Separate model loading from model inference so failures and tests are easy to isolate.
- Validate the model file/path and the expected tensor contract before running inference.
- Make inference errors observable and actionable without dumping huge tensors or images into logs.
- Do not claim a model is accurate based on one successful prediction. Evaluate on representative, labeled examples and report limitations.
- Do not introduce online training or model mutation into the camera-processing loop unless that is an explicit requirement.

### 7.6 Confidence and acceptance thresholds

Confidence values can have different meanings depending on the detector/OCR implementation. Document the source and semantics of each score. Do not interpret a raw model score as a calibrated probability unless that is justified by the model and evaluation.

If a confidence threshold is used:

- Put it in named configuration or a clearly named constant as appropriate.
- Explain whether it applies to detection, OCR, or a combined result.
- Test values below, at, and above the threshold.
- Select the value using representative examples and a documented trade-off between false positives and missed readings.
- Avoid claiming that confidence alone guarantees the plate text is correct.

### 7.7 Plate normalization and regional formats

- Keep text normalization in one testable function/component.
- Decide the target country's plate formats explicitly. If the requirement is Brazilian plates, test the formats the project actually needs to recognize rather than relying on a broad, undocumented regular expression.
- Preserve the raw prediction when it is useful for diagnostics; distinguish it from normalized/accepted text where practical.
- Do not blindly replace ambiguous characters such as `O`/`0`, `I`/`1`, or similar pairs without a documented policy supported by the model and target format.
- Do not modify a prediction merely to make it match an expected plate pattern. If it cannot be accepted reliably, mark it uncertain or reject it according to the contract.
- Avoid logging full plate text in general-purpose logs unless needed and approved. Use limited or redacted details for diagnostics where feasible.

### 7.8 Performance approach

Start with a correct, measurable baseline. Measure at least the useful stages where feasible: capture wait/read time, preprocessing time, detector latency, OCR latency, total frame processing time, and HTTP publication latency.

Only then consider optimizations such as frame-size reduction, lower sampling rate, batching, model warm-up, or moving capture/inference into separate execution resources. Do not introduce GPU-specific code, multiprocessing, complicated queues, or asynchronous wrappers merely to sound production-ready.

---

## 8. Pipeline and Recognition Result

### 8.1 Pipeline responsibility

The pipeline should coordinate a single frame or observation through a clear sequence, such as:

1. Receive a valid frame from the configured source.
2. Prepare it for the detector.
3. Obtain zero, one, or multiple plate detections.
4. Crop and prepare a selected plate region.
5. Run character recognition.
6. Normalize and evaluate the prediction.
7. Drop, mark uncertain, or accept the result according to the documented policy.
8. Apply duplicate suppression where appropriate.
9. Build the agreed event object.
10. Submit the event through the backend client and handle the outcome.

The exact order may vary according to the model and business integration contract. Keep the sequence explicit and each failure understandable.

### 8.2 Data contracts inside the service

Define a typed, framework-independent internal result rather than returning unstructured dictionaries from every layer. Fields may eventually include a plate string, detection/OCR scores, source/camera identifier, observation timestamp, and a unique event identifier, but only include fields that are useful and defined.

Do not mix these different concepts:

- An image/frame (a large array of pixels).
- A raw detector prediction.
- An OCR prediction.
- An accepted recognition observation.
- An outbound event for the external backend.
- An HTTP response from this microservice's own API.

Each boundary should have an explicit shape and clear ownership.

### 8.3 Event identity and timestamps

- Use an explicit event identifier if required for safe retries and duplicate prevention. The format and uniqueness guarantees must be documented.
- Use timezone-aware timestamps. Prefer UTC for event transport unless the backend contract says otherwise.
- Make clear whether a timestamp represents frame capture time, inference completion, or event publication. These times are not identical.
- Do not create a timestamp only after a slow inference and then describe it as the exact capture time.
- Make event payload fields match the agreed external contract exactly.

### 8.4 Example payloads are not contracts

A sample payload can be useful during discussion, for example:

```json
{
  "eventId": "example-unique-id",
  "cameraId": "entrance-camera",
  "plate": "ABC1D23",
  "confidence": 0.96,
  "capturedAt": "2026-10-08T22:00:00Z"
}
```

This is illustrative only. The field names, confidence definition, timestamp semantics, identifier generation, HTTP route, and required fields are **not** final unless the backend team has accepted them. Keep the integration schema and client aligned with the actual agreement; never present a hypothetical example as an implemented or agreed API.

---

## 9. Duplicate Suppression and Delivery Reliability

### 9.1 Two different duplication problems

Keep these concepts separate:

- **Observation deduplication:** the camera may see the same plate in many consecutive frames. The service may suppress repeated recognitions within a configured interval or tracking window.
- **Delivery idempotency:** the service may transmit one event, fail to receive the response, and retry. The backend should be able to recognize the same event identifier and avoid processing it twice.

One mechanism does not replace the other.

### 9.2 Deduplication policy

Before coding, define the intended semantics:

- What constitutes the same observation: plate text alone, camera plus plate, or a tracked vehicle/object?
- How long should a repeated result be suppressed?
- Under what conditions should the same plate be allowed again, such as leaving and re-entering the observation area?
- Does the policy differ per camera?
- How does the service behave after a restart?

For an initial small service, an in-memory time-based cache may be adequate, but it is lost on restart and does not coordinate multiple worker processes. State that limitation instead of implying durable or distributed deduplication.

Use monotonic time for elapsed-duration comparisons. Do not compare wall-clock timestamps for elapsed intervals if a monotonic clock is appropriate.

### 9.3 HTTP sending and failures

- Reuse an HTTPX client for repeated requests where practical so connection pooling works.
- Set explicit, configurable request timeouts appropriate to the operation. Do not disable timeouts by default.
- Check the response status and handle non-success responses deliberately.
- Handle network exceptions, timeouts, malformed responses, and service unavailability without crashing the capture loop unexpectedly.
- Do not retry every error indefinitely. Retries should be bounded, delayed appropriately, and used only for errors likely to be transient.
- Reuse the same event identifier when retrying the same event.
- Do not silently discard failed events while reporting a successful delivery.
- Do not claim “reliable delivery” if failed events are only written to a log. If a durable buffer/queue is not part of the approved stack, document the limitation and agree on how much loss is acceptable for the project.
- Do not add Redis, a message broker, or local database as an assumed fix. Present the trade-off and request approval if durable queuing becomes a requirement.

### 9.4 Contract ownership

The backend team owns the receiving API. Before final integration, confirm:

- Base URL and route path.
- HTTP method.
- Required JSON fields, field types, and field names.
- Authentication and authorization mechanism.
- Meaning of each response status/body.
- Expected behavior for duplicate event IDs.
- Rate limits, timeouts, and retry expectations.
- Whether the backend expects a single observation event, a vehicle tracking event, or another semantic unit.

Build the client around that agreement. Avoid inventing an endpoint such as `/recognitions` and treating it as confirmed merely because it seems reasonable.

---

## 10. FastAPI, Uvicorn, and Worker Lifecycle

### 10.1 API responsibilities

The FastAPI layer is for the microservice's own HTTP interface, operational visibility, and explicitly requested diagnostics. It is not the main parking backend.

Possible endpoints, only where useful:

- `GET /health/live`: process is alive and can respond.
- `GET /health/ready`: essential components are initialized and the service can perform its intended work.
- A diagnostic route for testing a submitted image, if this is required by the deliverable.

These endpoint names are suggestions. Follow existing routes and deployment conventions. A liveness check should not fail merely because an external backend is briefly unavailable; a readiness check may account for essential worker/model/camera state when that state is available to the API process.

Do not expose camera credentials, full connection URLs, access tokens, raw stack traces, model internals, or sensitive images in health responses.

### 10.2 Do not block the async event loop

OpenCV capture and TensorFlow inference can be synchronous and computationally expensive. Do not run an unbounded blocking `while` loop directly inside an `async def` route or other event-loop path.

Choose a clear execution strategy:

- Run a worker in a suitable thread/task with explicit cancellation and resource cleanup for a modest initial deployment; or
- Run the worker as a separate process when independent lifecycle, fault isolation, or deployment control is needed.

Do not add concurrency prematurely. Understand which operations are blocking before deciding how to execute them. Be cautious about thread safety of shared OpenCV captures and model objects.

### 10.3 Lifespan and resource management

Use FastAPI's supported application lifespan mechanism for resources that truly belong to the API process, such as shared clients or a model intentionally loaded for API inference. Close resources on shutdown.

If the camera worker is started during app lifespan:

- Make startup failure behavior explicit.
- Keep a reference to the worker/task/thread so it can be stopped and joined safely.
- Signal shutdown and release the camera.
- Ensure exceptions update the reported worker state.
- Avoid spawning duplicate workers on reload or when multiple Uvicorn processes are configured.
- Test startup and shutdown behavior.

If the worker will be independently deployed, give it a separate entry point and lifecycle. Do not pretend that local in-memory state is automatically shared between separate processes.

### 10.4 Diagnostic image uploads

`python-multipart` is needed only if a route accepts multipart form uploads, for example an `UploadFile` image test endpoint.

If adding such an endpoint:

- Validate allowed media types and actual image decoding; do not trust the filename or `Content-Type` alone.
- Set a reasonable configurable upload-size limit.
- Handle empty or malformed payloads cleanly.
- Avoid writing uploaded content to disk unless necessary; clean up temporary files if used.
- Keep this endpoint for diagnostic/manual image recognition, not as a replacement for continuous live-camera processing.
- Do not expose it publicly without the appropriate access controls.

---

## 11. Testing Strategy

Tests should reduce dependence on physical hardware and external services. Most tests must run without access to a real camera, a production backend, or heavyweight model artifacts.

### 11.1 Unit tests

Use small, deterministic tests for pure logic and isolated components, including:

- Configuration validation and environment overrides.
- Source-string parsing (camera index, video path, and URL).
- Frame-sampling calculations, including missing/zero/invalid FPS.
- Recognition result construction and validation.
- Bounding-box validation and crop boundaries.
- Plate normalization and format checks.
- Confidence/acceptance thresholds.
- Duplicate suppression before, at, and after the configured window.
- Event ID and timestamp behavior, where implemented.
- Payload serialization.
- HTTP response/error classification.

Do not load a real model for every unit test. Inject a fake or stub for model outputs where the behavior under test is not the model itself.

### 11.2 Integration tests

Use controlled components to test meaningful interactions, for example:

- Recorded test video -> frame source -> mocked or small test detector -> OCR adapter -> result.
- Pipeline -> deduplicator -> fake event publisher.
- Backend client -> mocked HTTP response, timeout, non-2xx response, and connection failure.
- FastAPI routes -> service dependency override or test client.
- Worker startup and shutdown -> fake source, including resource release on exception.

Do not make automated tests call the live parking-lot camera or a production backend.

### 11.3 End-to-end verification

When safe equipment and approved data are available, test the complete flow with a short recorded clip before connecting a live camera. Then verify the live source separately.

An end-to-end check should record:

- Source opens and frames are read.
- Expected frame-sampling behavior occurs.
- Plate detection returns valid coordinates where a plate is visible.
- OCR produces a structured result.
- The acceptance/deduplication policy behaves as specified.
- The outbound request matches the agreed contract.
- Network or camera failures are reported clearly.
- Resources are released when stopping the worker.

A successful import or a server that starts is not proof that plate recognition works.

### 11.4 Suggested edge cases

Cover relevant cases as each component is introduced:

- Invalid camera path or inaccessible URL.
- Capture fails to open.
- `read()` fails at end-of-file or due to a simulated source failure.
- Empty frame or malformed image.
- Zero/unavailable FPS.
- No plate in the frame.
- Multiple plate candidates.
- Bounding box extends outside the image.
- Model loading failure.
- Inference exception.
- Low-confidence prediction.
- Ambiguous characters or unsupported plate format.
- Same plate repeated across nearby frames.
- Same plate seen again after the suppression window.
- Backend timeout, connection refusal, non-2xx response, malformed response, and retry exhaustion.
- Shutdown while waiting for a frame or during a retry delay.
- Invalid configuration and missing required settings.

### 11.5 Test data and fixtures

- Prefer small, deterministic, legally usable fixtures.
- Use synthetic or consented examples where possible.
- Check dataset/model terms before redistributing any external imagery or weights.
- Do not commit large videos, production surveillance footage, secrets, or unnecessary identifiable data.
- If a fixture is too large for ordinary Git, agree on an artifact/download strategy and document reproducible setup.
- Tests should not require an internet connection unless the test explicitly verifies an external integration and is marked accordingly.

### 11.6 Test quality

- Assert observable behavior, not irrelevant implementation details.
- Avoid tests that pass only because a method was called; validate the result and relevant side effects.
- Keep random/model-specific behavior isolated where deterministic assertions are impossible.
- When mocking, patch the symbol where the tested code looks it up, or pass a dependency explicitly.
- Fix flaky tests rather than adding sleeps and retry loops to hide race conditions.

### 11.7 Definition of done for a change

A change is complete when:

- Its responsibility and behavior are clear.
- Relevant unit/integration tests exist or the absence of a test is justified.
- The targeted tests pass in the available environment.
- Configuration, exceptions, and resource cleanup are considered.
- No credentials or unrelated files were added.
- The appropriate documentation is updated.
- The assistant reports what was actually run and any remaining limitations.

---

## 12. Logging, Diagnostics, and Observability

### 12.1 Useful operational signals

Prefer structured, concise logs for events such as:

- Service startup and shutdown.
- Camera open/close and reconnect attempt.
- Worker state transitions.
- Model load success/failure.
- Frames captured and frames selected for analysis.
- Detection/OCR processing failures.
- Recognition accepted/rejected due to an explicit criterion.
- Duplicate observation suppressed.
- Backend event attempted, accepted, rejected, or failed.
- Processing latency and repeated error counts when measured.

Use appropriate log levels. Expected “no plate found” outcomes should not necessarily be warnings on every frame. Avoid producing a line for every high-frequency event if it makes logs unusable.

### 12.2 Do not leak sensitive information

- Do not log secrets, authorization headers, RTSP passwords, or full environment contents.
- Do not log complete frames, base64 image payloads, or full surveillance images.
- Avoid logging full plate text by default. If it is required for debugging, make the behavior explicit, restricted, and temporary, and follow the team's data-handling policy.
- Return safe error messages to HTTP clients; keep diagnostic details in appropriately controlled logs.

### 12.3 Health vs. readiness

Keep the meanings distinct:

- **Liveness:** is the process alive and capable of responding?
- **Readiness:** are the resources required to perform the intended recognition work available?

Do not report the entire service as healthy if its worker has died silently. Equally, do not report liveness failure just because the external backend briefly returns an error. Define status according to the endpoint's explicit purpose.

### 12.4 Avoid inventing monitoring infrastructure

Logging and simple health endpoints are enough until there is a real deployment requirement for metrics, tracing, alerting, or dashboards. Do not add a monitoring stack as a default project dependency.

---

## 13. Security, Privacy, and Data Handling

License-plate imagery and recognition output can reveal information about identifiable people depending on context. Handle it deliberately.

- Process only footage/data the team is allowed to use.
- Do not bypass camera access controls or obtain credentials without authorization.
- Keep credentials in environment configuration or the deployment's secret mechanism, not source files.
- Use HTTPS when required by the deployment environment; do not disable TLS verification to “fix” certificate issues.
- Authenticate the service-to-backend call using the mechanism agreed with the backend team.
- Keep diagnostic/upload endpoints restricted to their intended users and network.
- Validate data received through HTTP endpoints; do not trust filenames, MIME types, or JSON blindly.
- Send only the fields needed by the receiving backend. Do not upload raw images by default.
- Define whether frames, crops, or intermediate data are kept, where they are stored, who can access them, and when they are deleted.
- Do not introduce persistent storage of captured images without explicit requirements and approval.
- Do not place personal data in Git fixtures, examples, public logs, or bug reports unnecessarily.
- Do not claim legal compliance solely because these precautions are present. Follow the project's institutional policies and applicable data-protection requirements.

---

## 14. Error Handling and Failure Semantics

Errors should be classified so the application can react appropriately instead of crashing indiscriminately or silently ignoring failures.

Consider separate categories for:

- Configuration error (invalid/missing required value).
- Source open/read failure.
- Expected absence of a plate in a frame.
- Invalid detector/OCR output.
- Model initialization/inference failure.
- Rejected recognition (for example, below an agreed quality threshold).
- Backend connection/timeout failure.
- Backend rejection of a payload or unauthorized call.
- Worker shutdown/cancellation.

Guidelines:

- Use explicit exceptions or result types when they make control flow clearer.
- Catch exceptions at the layer that can handle them meaningfully. Do not wrap every function in `except Exception: pass`.
- Preserve the cause when re-raising exceptions where useful.
- Do not convert infrastructure failures into “no plate found”; those mean different things.
- Do not crash the whole process because one malformed frame or transient HTTP failure occurred unless that is an intentional fail-fast policy.
- Conversely, do not keep running indefinitely in a known broken state while health endpoints report readiness.
- Make startup-critical failures fail clearly, and define whether later runtime failures trigger retry, degraded readiness, or worker termination.
- Avoid unbounded retries, unbounded in-memory queues, and tight error loops.

---

## 15. Performance and Resource Management

### 15.1 Measure first

Identify the actual bottleneck before optimizing. Record processing latency, frame rate, memory use, and send latency where practical. Do not guess that TensorFlow, OpenCV, or HTTPX is the bottleneck without evidence.

### 15.2 Memory

- Process frames incrementally instead of loading an entire long video into memory.
- Do not keep all frames or every intermediate image in a growing list.
- Bound any queue or cache introduced for buffering/deduplication.
- Release capture resources and close HTTP clients.
- Be conscious of copying large NumPy arrays; optimize after correctness is established.

### 15.3 Model and capture lifecycle

- Load reusable models and clients once per relevant process/lifecycle, not once per frame.
- Do not share a camera handle across unrelated threads without a clear ownership model.
- Ensure cancellation can stop the worker even if the backend is unavailable.
- If a library call can block indefinitely in the selected environment, document the limitation and investigate a safe timeout or process-isolation strategy.

### 15.4 Concurrency

Do not add asynchronous code just because FastAPI is async. OpenCV and model inference may be synchronous. Use a thread, process, or worker loop only for a measured/understood reason, and test startup, exceptions, stop behavior, and resource ownership.

If a production requirement later includes several cameras, stateful tracking, or high event throughput, reassess isolation and concurrency then. Do not overbuild that future architecture into the first deliverable.

---

## 16. API and Integration Contract

### 16.1 The external backend is a dependency

The Python service is a producer of recognition events. The external Node.js/Express backend is the owner of the receiving endpoint and parking-system business rules.

Keep an integration note in the README or a dedicated document once the teams agree on the contract. It should record:

- Backend base URL configuration name.
- Route and method.
- Authentication mechanism, without exposing secret values.
- Request schema and field definitions.
- Timestamp and confidence semantics.
- Expected success status/body.
- Error status interpretation.
- Retry/idempotency policy.
- Integration test procedure.

Do not store a second handwritten schema that silently diverges from the agreed contract. If the backend team changes the contract, update the client and tests in one deliberate change.

### 16.2 HTTPX client expectations

- Prefer a reusable `httpx.Client` or `httpx.AsyncClient` that matches the chosen execution model.
- Set explicit timeouts; handle connect/read/write/pool timeout cases where relevant.
- Reuse connections when sending repeated events.
- Check HTTP status codes and parse responses only according to the actual contract.
- Treat non-2xx responses as explicit outcomes, not success.
- Do not log credentials or entire sensitive payloads.
- Close the client at the end of its lifecycle.
- Avoid mixing synchronous and asynchronous HTTP APIs arbitrarily. Match them to the caller and worker design.

### 16.3 No implicit delivery guarantees

An HTTP POST attempt is not the same as confirmed delivery, and a successful HTTP response is not necessarily proof that a business operation has completed unless the backend contract says so. Report only what the response actually confirms.

If the current implementation cannot buffer events durably, document that data may be lost when the process stops during an outage. Do not hide this trade-off.

---

## 17. Development Workflow and Commands

The commands below are examples. Verify the actual package names, route paths, and test configuration before using them.

### 17.1 Environment and dependencies

```bash
uv sync
uv run python --version
uv run python -c "import cv2, numpy, tensorflow; print('core ML imports succeeded')"
```

Use the import check only after the dependencies are installed and supported in the current environment. A failure can indicate a missing dependency or a binary/platform compatibility issue; diagnose it rather than randomly reinstalling packages.

### 17.2 Start the API in development

```bash
uv run uvicorn app.main:app --reload
```

Adjust `app.main:app` if the project's real ASGI import path differs. Do not use auto-reload in production. If app startup owns a physical camera worker, understand reload behavior before starting it.

### 17.3 Run tests

```bash
uv run pytest -q
uv run pytest tests/unit -q
uv run pytest tests/integration -q
```

Run the narrowest relevant test during iteration, then the broader suite before reporting completion where runtime permits.

### 17.4 Check environment reproducibility

Use the lockfile-aware workflow supported by the installed `uv` version, for example:

```bash
uv sync --locked
uv run --locked pytest -q
```

If the installed version has different command support, check `uv --help` or the official `uv` documentation rather than silently dropping reproducibility checks.

### 17.5 Manual video checks

Keep a small development video in the agreed fixture/experiment location. The manual check should print or log:

- Whether the source opened.
- Reported metadata and any fallback used.
- Number of frames read and number selected for inference.
- First frame shape/dtype for sanity checking.
- Result count and processing errors.
- Confirmation that the source is released after completion or Ctrl+C.

Do not print full raw frame arrays. Do not rely on manual checks as a replacement for automated unit tests.

---

## 18. Git, File Changes, and Documentation

- Modify only files required for the task.
- Do not erase or rewrite user changes to “clean up” the repository without permission.
- Do not reformat the entire repository to make a small feature change.
- Keep commits, if asked to create them, focused and descriptive; do not commit secrets or unapproved footage/models.
- Update documentation when configuration, commands, model artifacts, endpoint contracts, or operational behavior changes.
- Keep README instructions runnable and consistent with `pyproject.toml`/`uv.lock`.
- Do not create an extra `requirements.txt`, Docker Compose file, CI workflow, ADR directory, or complex deployment stack unless it addresses an actual need.
- Where a decision is significant and not obvious (model choice, accepted plate formats, event semantics, loss policy during outage), record the rationale and remaining trade-offs in the project's documentation.

### 18.1 Avoid giant generated changes

Do not create every directory and fill every module before there is a working end-to-end path. Build one vertical slice at a time—for example, test-video capture, then a replaceable detection interface, then recognition, then publication.

A partially implemented module should not expose many unused classes or placeholder methods that look complete but do nothing. Prefer a small working implementation with explicit limitations.

---

## 19. Recommended Implementation Milestones

The order can be adjusted to the repository's current state. Each milestone should produce a testable capability.

### Milestone 1 — Confirm project foundations

- Inspect and validate Python 3.11 / `uv` setup.
- Review `pyproject.toml`, the lockfile, package import path, `.gitignore`, and `.env.example`.
- Ensure tests can be executed with the project environment.
- Avoid changing ML strategy before establishing the environment.

**Exit condition:** the environment is reproducible and the current application/test entry points are understood.

### Milestone 2 — Configurable frame source

- Implement or refine configuration with Pydantic Settings.
- Implement the camera/video-source abstraction.
- Support a short local test video first.
- Test open failure, read failure, sampling, and release.
- Keep live-camera-specific behavior separate from offline-file expectations where needed.

**Exit condition:** a small script/test can read and sample frames without recognition logic, and resources are reliably released.

### Milestone 3 — Recognition result contract

- Define the smallest useful internal result structure.
- Decide how coordinates, plate text, score(s), source ID, and timestamps are represented.
- Keep undecided backend payload fields out of unrelated recognition code.

**Exit condition:** a deterministic fake result can travel through the application without requiring a real model.

### Milestone 4 — Detection and OCR

- Confirm model strategy, weights/source, environment compatibility, input/output contract, license, and dataset constraints.
- Implement preprocessing that matches the chosen model.
- Implement detector and OCR adapters with small tests.
- Test no-plate and malformed-output conditions.
- Record a repeatable evaluation procedure.

**Exit condition:** a fixed, approved set of test images produces structured outputs, including appropriate behavior for uncertain or absent detections.

### Milestone 5 — Pipeline orchestration

- Wire source, detector, OCR, post-processing, and result creation together.
- Keep dependencies replaceable in tests.
- Make component failures distinguishable.
- Measure baseline latency.

**Exit condition:** a recorded clip passes through the pipeline and yields observable results without the external backend being required.

### Milestone 6 — Duplicate suppression

- Agree on what counts as the same event and when a plate may be reported again.
- Implement a bounded, testable initial policy.
- Document in-memory/restart limitations if persistence is not part of scope.

**Exit condition:** tests demonstrate both suppression of repeated observations and acceptance after the configured policy permits a new observation.

### Milestone 7 — Backend integration

- Obtain the accepted request/response contract from the other team.
- Implement the HTTPX client and settings.
- Test success, non-success responses, timeouts, and network errors using mocks.
- Agree on event identity, authentication, retries, and delivery-loss behavior.

**Exit condition:** a contract test matches the agreed API and failure outcomes are handled explicitly.

### Milestone 8 — Operational API and worker lifecycle

- Add health/readiness routes that reflect actual state.
- Decide whether worker execution is app-lifespan-managed or a separate process.
- Ensure camera/model/client resources start and stop cleanly.
- Test no duplicate capture initialization.

**Exit condition:** service startup, readiness, failure reporting, and shutdown are predictable.

### Milestone 9 — End-to-end validation and deployment readiness

- Test the service with an approved clip and then the available live camera.
- Measure recognition performance and operational latency.
- Document setup, configuration, model acquisition, limitations, and integration procedure.
- Add Docker or CI only as required for handoff/deployment.

**Exit condition:** another team member can set up the service, run tests, perform a documented recognition check, and understand what is and is not guaranteed.

---

## 20. Common Mistakes to Avoid

Do **not**:

- Implement the main backend's visitor, vehicle, reservation, authorization, or database logic in Python.
- Assume the backend endpoint or JSON contract without confirmation.
- Put continuous capture/inference in an HTTP route.
- Load TensorFlow models per frame or per request without a specific reason.
- Open the camera when modules are imported.
- Forget `VideoCapture.release()` on exceptions or shutdown.
- Assume webcam/file/RTSP sources report reliable identical metadata.
- Treat end-of-file, camera disconnection, “no plate found,” and model failure as the same condition.
- Hard-code camera URLs, credentials, model paths, server URLs, or operational thresholds throughout the code.
- Assume OpenCV BGR images already match the model's input format.
- Invent a TensorFlow architecture, pretrained weight file, dataset, or reported accuracy.
- Confuse confidence scores with calibrated probabilities without evidence.
- Automatically “correct” ambiguous license-plate characters to force a valid result.
- Send every frame or every repeated plate without an explicit event policy.
- Treat observation deduplication as a substitute for backend idempotency.
- Disable HTTP timeouts or retry forever.
- Add a queue, database, cache server, or orchestrator without demonstrated need.
- Depend on a physical camera or real backend in the normal unit-test suite.
- Catch and suppress every exception.
- Claim the service is ready because FastAPI returns HTTP 200 when the worker/model/camera is broken.
- Log secrets or dump entire images/NumPy arrays.
- Bulk-generate the whole architecture before proving one end-to-end slice.
- Say tests passed when they were not executed.

---

## 21. Decision Checklist for Significant Changes

Before a significant design or dependency change, answer the applicable questions:

### Scope
- Does this belong to plate recognition and event publication, or to the external backend?
- Does the change alter the agreed backend API contract?
- Can the requirement be met without adding another service or dependency?

### Architecture
- Which component owns the behavior and lifecycle?
- Can the component be tested without camera hardware and without the network?
- Does the change create shared mutable state, a second camera reader, or process-specific state?
- Is an abstraction solving a real variation or testability need?

### ML correctness
- Is the model and its input/output contract known?
- Are color order, shape, dtype, normalization, and coordinate conventions correct?
- Are thresholds supported by tests or evaluation data?
- What happens when the plate is missing, blurred, partly occluded, or ambiguous?

### Reliability
- What happens on startup failure, frame-read failure, model exception, HTTP timeout, and shutdown?
- Can retries create duplicates?
- Is the failure visible to logs/health checks?
- Are memory buffers/caches bounded?

### Learning and maintainability
- Can the student explain what the changed file does and why it exists?
- Are the relevant new concepts explained in context?
- Is the implementation no more complex than the requirement calls for?
- Is there a concrete way to verify the behavior?

---

## 22. AI Response Template for Project Tasks

Use this compact structure when it helps organize a development response. Adapt it to the size of the task; do not rigidly repeat headings for trivial changes.

### Goal
One or two sentences describing the capability being added or the defect being corrected.

### Current code and design
Identify the relevant files and the ownership boundary. State any assumptions that matter.

### Implementation steps
A small, ordered sequence with the exact files, functions, contracts, and behavior involved. Do not use vague instructions.

### Explanation
Explain new language/library concepts alongside their use. Explain why each component owns its responsibility and what failure cases matter.

### Verification
Provide runnable commands or specific tests and describe the expected observable result. Do not fabricate output.

### Review / next step
Summarize what is complete, what remains uncertain, and the next logical project task.

When the student is writing the code manually, provide focused guidance and review the code they submit. When the student asks the AI to make edits directly, make a focused change, preserve existing work, then explain the diff and verification results.

---

## 23. Official Documentation References

Check the documentation for the installed versions and the actual target platform. Official documentation should take precedence over stale snippets, blog posts, or assumptions copied from other projects.

- **Python 3.11:** https://docs.python.org/3.11/
- **uv — projects, dependencies, and lockfiles:** https://docs.astral.sh/uv/guides/projects/
- **uv — dependency synchronization:** https://docs.astral.sh/uv/concepts/projects/sync/
- **FastAPI — larger applications and routers:** https://fastapi.tiangolo.com/tutorial/bigger-applications/
- **FastAPI — application lifespan:** https://fastapi.tiangolo.com/advanced/events/
- **Pydantic Settings:** https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- **HTTPX — clients and connection pooling:** https://www.python-httpx.org/advanced/clients/
- **HTTPX — timeouts:** https://www.python-httpx.org/advanced/timeouts/
- **OpenCV — `VideoCapture` reference:** https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- **TensorFlow — pip installation and environment compatibility:** https://www.tensorflow.org/install/pip
- **pytest — fixtures:** https://docs.pytest.org/en/stable/how-to/fixtures.html
- **pytest — monkeypatching:** https://docs.pytest.org/en/stable/how-to/monkeypatch.html

Documentation links can change. If a link is unavailable or does not match the installed version, find the corresponding current official page and confirm the behavior before writing version-sensitive code.

---

## 24. Final Rule

Build a **small, testable, understandable plate-recognition service** that reads configured video sources, produces defensible recognition results, and publishes them through the backend team's agreed contract.

Make real progress with each iteration. Explain the relevant code and decisions. Keep ownership boundaries clear. Verify behavior before claiming success. Prefer working, measured, incremental delivery over speculative infrastructure or a large amount of untested code.
