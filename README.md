# Customer Feedback Support Agent

An AI-powered customer support workflow built with **Python, Streamlit, LangGraph, LangChain, and LLMs**.

The application accepts customer feedback, validates that the input is in scope, analyzes sentiment, evaluates negative feedback for urgency and severity, and prepares an email response. For higher-priority complaints, it also prepares a support ticket and notifies the appropriate internal recipients. Before an email is sent to a customer, a human reviewer can approve it, request a revision, or stop the workflow.

The goal is to combine useful LLM automation with **deterministic guardrails, explicit human approval, and controlled workflow execution**.

---

## Table of contents

- [At a glance](#at-a-glance)
- [What problem does this project solve](#what-problem-does-this-project-solve)
- [How the application works](#how-the-application-works)
- [Final workflow graph](#final-workflow-graph)
- [Priority and escalation rules](#priority-and-escalation-rules)
- [Guardrails and safety](#guardrails-and-safety)
- [Human review and approval](#human-review-and-approval)
- [Technology stack](#technology-stack)
- [Database and migrations: step-by-step](#database-and-migrations-step-by-step)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Environment configuration](#environment-configuration)
- [Run the application](#run-the-application)
- [Run tests](#run-tests)
- [State, checkpointing, and limitations](#state-checkpointing-and-limitations)
- [Screenshots and visual documentation](#screenshots-and-visual-documentation)
- [Possible future enhancements](#possible-future-enhancements)
- [License](#license)

---

## At a glance

| Area | What the application does |
|---|---|
| User interface | Collects customer name, email, and feedback in Streamlit |
| Input validation | Checks feedback with a deterministic Python guardrail |
| Scope validation | Uses a separate LLM-based guardrail with structured output |
| Sentiment | Classifies feedback as positive or negative |
| Complaint analysis | Evaluates negative feedback, including issue details, urgency, and severity |
| Prioritization | Assigns P1, P2, or P3 handling |
| Ticket handling | Prepares a support ticket for P1/P2 cases |
| Email drafting | Generates a customer-facing email |
| Human oversight | Allows a reviewer to accept, request evaluation/revision, or stop |
| Delivery | Sends the accepted email using SMTP |
| Workflow state | Persists LangGraph checkpoints and idempotency records in PostgreSQL |

## What problem does this project solve?

Customer feedback can arrive in large volumes and may contain anything from a simple compliment to an urgent service complaint. Manually reviewing every message, deciding which issue needs attention, drafting a response, and notifying the right people can be time-consuming.

This project demonstrates how an AI-assisted workflow can help:

1. **Understand feedback:** identify whether a message is positive or negative.
2. **Recognize important complaints:** extract the issue and assess urgency and severity.
3. **Prioritize work:** distinguish lower-priority feedback from issues that need support or management attention.
4. **Prepare consistent communication:** draft a relevant response for the customer.
5. **Keep a human in control:** require a reviewer to approve the email before delivery.

The application assists the support process; it does not replace the human reviewer.

## How the application works

### 1. Customer submits feedback

A user enters the customer's details and feedback through the Streamlit interface:

- Customer name
- Customer email address
- Feedback message

### 2. Input guardrails validate the message

Before the main feedback workflow runs, the application checks whether the submitted text is appropriate for this customer-feedback use case.

- A deterministic custom Python guardrail runs first.
- An LLM-based scope guardrail runs next.
- If a guardrail rejects the input, the main feedback workflow does not proceed.

### 3. Sentiment is classified

The workflow determines whether the feedback is positive or negative.

- **Positive feedback:** the application prepares a thank-you email.
- **Negative feedback:** the application proceeds to complaint analysis and prioritization.

### 4. Negative feedback is analyzed

The complaint analysis evaluates details such as the issue category, urgency, and severity. The workflow uses these results to determine the appropriate priority and escalation path.

### 5. The response is drafted

The application generates a customer-facing email. Depending on the priority, it may also prepare a support ticket and internal notifications.

### 6. A human reviews the email

The workflow pauses for a reviewer to choose an outcome:

- Accept the draft.
- Request further evaluation/revision with clarification.
- Stop the workflow.

### 7. The approved email is sent

Only an accepted draft proceeds to SMTP email delivery. A stopped workflow does not send the customer email.

---

## Final workflow graph

The following Mermaid diagram is the high-level workflow represented by the current application. GitHub renders Mermaid diagrams directly in Markdown.

```mermaid
flowchart TD
    A([Start]) --> B[Customer submits details and feedback in Streamlit]
    B --> C[Deterministic custom input guardrail]
    C --> D{Input allowed?}

    D -- No --> X[Reject input and show safe message]
    X --> Z([End])

    D -- Yes --> E[LLM scope guardrail]
    E --> F{In-scope feedback?}

    F -- No --> Y[Reject out-of-scope or unsafe request]
    Y --> Z

    F -- Yes --> G[Sentiment analysis]
    G --> H{Positive or negative?}

    H -- Positive --> I[Generate thank-you email]
    H -- Negative --> J[Analyze complaint, urgency, and severity]
    J --> K{Priority?}

    K -- P3 --> M[Prepare negative email]
    K -- P2 --> N[Create support ticket]
    K -- P2 --> O[Prepare negative email]
    K -- P1 --> P[Create support ticket]
    K -- P1 --> Q[Prepare negative email]

    N --> R[Notify support team]
    P --> S[Notify support team and manager]
    I --> T[Human email review interrupt]
    M --> T
    O --> T
    Q --> T

    T --> U{Reviewer decision}
    U -- Accepted --> V[Send approved email through SMTP]
    U -- Needs Evaluation --> W[Validate reviewer clarification]
    W --> X{Revision allowed and limit not reached?}
    X -- Yes --> Y[Regenerate email draft]
    Y --> T
    X -- No --> AA[Stop or return for final reviewer decision]

    U -- Stop --> AB[End without sending customer email]
    V --> AC([Completed])
    AA --> AB
    AB --> Z
```

### Graph walkthrough

| Graph stage | Purpose |
|---|---|
| Streamlit submission | Collects the customer's details and feedback |
| Custom guardrail | Performs deterministic checks before LLM processing |
| LLM scope guardrail | Checks whether the request belongs to the supported feedback use case |
| Sentiment analysis | Routes positive and negative feedback differently |
| Complaint analysis | Evaluates issue details, urgency, and severity for negative feedback |
| Priority routing | Selects the P1, P2, or P3 handling path |
| Email generation | Creates the customer-facing draft |
| Human review interrupt | Pauses the graph until a reviewer responds |
| Clarification validation | Checks reviewer feedback before a revision is generated |
| SMTP delivery | Sends the accepted customer email |

> **Implementation note:** This is a documentation-level view of the workflow. The exact node names, edge names, and conditional routing should be kept in sync with the graph implementation in `src/app/agents/` and the pipeline entry points in `src/app/pipelines/`.

## Priority and escalation rules

The current workflow uses three priority levels for negative feedback.

| Priority | Typical handling | Internal action | Customer response |
|---|---|---|---|
| **P3** | Lower urgency | No support-ticket escalation in the described flow | Apology-focused response |
| **P2** | Requires support attention | Prepare support ticket and notify support team | Relevant response draft |
| **P1** | Critical issue requiring escalation | Prepare critical ticket and notify support team and manager | Relevant response draft |

The exact priority decision is made by the complaint-analysis logic. Priority should be based on the application's configured rules and analysis output, not on the sentiment label alone.

## Guardrails and safety

The application uses more than one validation layer so that untrusted text is checked before it reaches the main LLM workflow.

### 1. Deterministic custom input guardrail

The custom Python guardrail runs before LLM processing of customer feedback.

It is checked:
- In the Streamlit form, so the user can receive immediate feedback.
- Again in the pipeline, as a safeguard for callers that do not use the Streamlit interface.

Example:

```python
from app.guardrail import custom_guardrail

result = custom_guardrail(feedback)

if not result.allowed:
    raise ValueError(result.reason)
```

This layer provides predictable, code-controlled checks. Keyword matching is a preliminary signal; it is not a complete security solution and does not prove that an input is safe.

### 2. LLM-based scope guardrail

A second validator runs after the deterministic check and before the customer-feedback workflow.

It uses:
- A separate system prompt.
- Structured Pydantic output.
- A fail-closed policy for rejected or uncertain input.

It is intended to reject unrelated requests, prompt-injection attempts, and ambiguous requests before the main workflow starts.

Example:

```python
from app.guardrail import llm_guardrail

result = llm_guardrail(feedback)

if not result.allowed:
    raise ValueError(result.safe_response or result.reason)
```

### 3. Reviewer clarification validation

Human review comments are also checked before the graph resumes and before a revision reaches the email-generation LLM.

Review-specific wording checks allow normal feedback about tone, phrasing, or content. The reviewer decision and score are structured controls rather than free-text instructions to the LLM.

### Guardrail boundary

The current application does not provide inbound free-text forms for support-team or manager responses; those roles receive generated notifications. If such input is added later, apply suitable validation before passing it into the workflow.

## Human review and approval

The email review step is a Human-in-the-Loop (HITL) checkpoint. LangGraph pauses execution while a person reviews the generated draft.

| Reviewer choice | Workflow behavior |
|---|---|
| `Accepted` | Continues to SMTP delivery |
| `Needs Evaluation` | Validates clarification and may regenerate the draft |
| `Stop` | Ends the workflow without sending the customer email |

A configurable `MAX_EMAIL_REVISIONS` setting limits the revision loop and helps prevent endless regeneration.

The review step is important because generated text can be inaccurate, incomplete, or unsuitable for a particular customer situation. Human approval is the final control before customer email delivery.

## Technology stack

| Technology | Role in this project |
|---|---|
| Python 3.13+ | Main application language |
| Streamlit | User interface for submitting feedback and reviewing workflow output |
| LangGraph | Stateful workflow orchestration, branching, interrupts, and checkpointing |
| LangChain | LLM integration and model invocation |
| Google Gemini | Configured LLM provider used for negative-feedback analysis |
| Groq | Configured LLM provider used for sentiment and email generation |
| PostgreSQL | Durable workflow checkpoints, support tickets, and email delivery records |
| SQLAlchemy 2.x and Psycopg 3 | PostgreSQL access and connection pooling |
| LangGraph Checkpoint PostgreSQL | Persistent graph checkpointing, including human-review interrupts |
| Alembic | Versioned database schema migrations |
| Pydantic and pydantic-settings | Structured data models and environment-based configuration |
| SMTP | Outbound customer and internal email delivery |
| `uv` | Dependency and environment management |

Provider usage can be controlled through the application's environment configuration.

## Database and migrations: step-by-step

This project uses PostgreSQL for two related but distinct kinds of persistence:

1. **Application records**: support tickets and email delivery status. These
   tables are described with SQLAlchemy and created or changed by Alembic.
2. **Workflow checkpoints**: LangGraph's saved state, including state needed
   to resume after human approval. LangGraph's PostgreSQL checkpointer manages
   its own tables.

A missing `support_tickets` table is an application schema/migration issue; a
workflow that cannot resume a human review is a checkpointer or configuration
issue. Separating the two makes failures easier to diagnose.

### The real problem this solves

If all information lived only in application memory, a server restart could
erase a generated ticket or lose a workflow paused while a person reviews an
email. If two workers processed the same feedback at once, both could create a
ticket or send the same email. PostgreSQL persists state across restarts and
uses shared constraints to coordinate workers. The design reduces lost work
and duplicate side effects; it cannot make an external SMTP server and the
database one atomic transaction.

### 1. Configure a PostgreSQL connection

Set `DATABASE_URL` in the local `.env` file and keep the real credential out of
Git:

```dotenv
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require
```

`src/app/configs/config.py` reads this as a secret. In
`src/app/database/connection.py`, `_database_url()` rejects a missing URL,
requires PostgreSQL, and normalizes its driver to `postgresql+psycopg`. This
fails early with a useful message if persistence is not configured.

`get_engine()` creates and caches SQLAlchemy's engine, which reuses a small
connection pool instead of reconnecting for each query. `pool_pre_ping=True`
checks pooled connections before reuse to catch stale connections.
`_get_session_factory()` creates database sessions. `session_scope()` wraps a
unit of work in a transaction: success commits, exceptions roll back. The
shutdown hook disposes the pool. `get_checkpointer_url()` formats the same
database URL for the psycopg interface expected by LangGraph.

### 2. Describe application tables with models

`src/app/database/models.py` is the Python description of the current schema.
`Base` is the metadata registry Alembic uses to inspect these models.

**`SupportTicketRecord` (`support_tickets`)**

- `thread_id` is the primary key: one workflow thread can have at most one
  persisted ticket. A retry can reuse its existing ticket.
- `ticket_id` is required and unique, preventing two rows from sharing a
  customer/support-facing ticket identifier.
- `ticket_data` is PostgreSQL `JSONB`, keeping the ticket's structured fields
  together while allowing PostgreSQL to query JSON values.
- `created_at` has a database default, so callers do not need to supply a
  timestamp and all rows use the database clock.

**`EmailDeliveryRecord` (`email_deliveries`)**

- `idempotency_key` is the primary key. A logical email has one delivery
  record, even when workers race to claim it.
- `status` tracks `sending`, `sent`, or `failed`. A check constraint rejects
  invalid values, including spelling mistakes.
- `error` optionally stores failure details for diagnosis.
- `created_at` and `updated_at` track when delivery began and last changed.

### 3. Create a versioned migration

Models describe the schema in Python; they do not modify a live database.
`migrations/versions/20261118_01_create_persistence_tables.py` is the ordered
instruction Alembic runs to create these tables.

The migration's `revision` is its unique identifier. `down_revision = None`
means this is currently the first migration in the chain. Alembic stores
applied revision identifiers in its own version table, so it knows which
changes are pending.

In `upgrade()`, `op.create_table("support_tickets", ...)` creates the ticket
columns and constraints. `op.create_table("email_deliveries", ...)` creates
the delivery record columns and allowed-status constraint. The definitions
mirror the SQLAlchemy models so a newly migrated database matches what the
application expects.

In `downgrade()`, Alembic drops `email_deliveries` and then `support_tickets`.
This reverses table creation, but dropping tables deletes their contents. A
production rollback should therefore be planned and backed up appropriately.

When the schema changes, update the model and add a new migration. Do not
rewrite a migration that has already been applied to shared databases: a new
revision gives every environment the same auditable sequence of changes.

### 4. Configure Alembic

`alembic.ini` points Alembic to `migrations/` and the migration template. The
URL there is a placeholder. `migrations/env.py` replaces it with the validated
application URL, avoiding a second copy of the database credential.

`env.py` imports `Base` and sets `target_metadata = Base.metadata`, making the
models visible to Alembic. `compare_type=True` asks Alembic to compare column
types when suggesting schema differences. Online mode connects to PostgreSQL
and applies changes in a transaction. Offline mode renders SQL without
connecting, useful for review or a DBA workflow. `NullPool` prevents a
short-lived migration command from keeping an application connection pool.

### 5. Apply the migration

With PostgreSQL running and `.env` configured, run this from the repository
root:

```bash
uv run alembic upgrade head
```

`upgrade` applies pending revisions; `head` means the latest revision in this
repository. Alembic records the current revision and invokes each migration's
`upgrade()` function. Running the command again is safe: if the database is
already at `head`, there are no pending revisions.

To inspect the database revision or repository history:

```bash
uv run alembic current
uv run alembic history
```

`current` reports the revision applied to the configured database; `history`
lists the migration chain. During deployment, apply migrations before starting
application code that depends on newly added columns or tables.

### 6. Use repositories for safe operations

`src/app/database/repositories.py` keeps SQL statements out of workflow code
and exposes operations in terms of the task the app needs to perform.

`SupportTicketRepository.create_or_get()` inserts a ticket with
`ON CONFLICT DO NOTHING` on `thread_id`, then loads the row for that thread.
The primary key resolves races: if a graph node retries or two workers create
the same ticket concurrently, PostgreSQL allows only one row and both can use
the persisted ticket. If the record cannot be loaded afterward, the repository
raises an error rather than claiming it was saved.

`EmailDeliveryRepository.claim()` tries to insert a `sending` record. Exactly
one concurrent worker can create a row for the idempotency key; other workers
read its existing status and do not claim the same email as new.
`mark_sent()` and `mark_failed()` record the result and verify that a row was
updated. Each operation commits through `session_scope()`.

The claim is committed before SMTP is called. If the process crashes after the
email provider accepts a message but before the database records `sent`, the
database cannot know whether delivery occurred. This project leaves that
unresolved `sending` record blocked from automatic retry to avoid a possible
duplicate. This is a deliberate safety tradeoff, not exactly-once delivery.

### 7. Save and resume LangGraph workflow state

`src/app/database/checkpointer.py` creates a `PostgresSaver` with a psycopg
connection pool. A lock and cached instance prevent repeated initialization
inside one process. `checkpointer.setup()` prepares LangGraph's checkpoint
tables when the checkpointer is first initialized.

Those checkpoint tables are separate from the Alembic-managed application
tables. They store graph state by thread, so a workflow interrupted for human
review can resume with its previous analysis and draft after a restart. The
ticket table stores a business record; the checkpoint tables store resumable
workflow state. Both use the configured PostgreSQL database but serve
different purposes.

### Benefits in this project

- **Durability:** ticket records, email outcomes, and workflow checkpoints
  survive application restarts.
- **Safer retries:** database keys prevent duplicate tickets and duplicate
  claims for the same logical email.
- **Worker coordination:** PostgreSQL constraints handle concurrent attempts
  across processes, unlike an in-memory check local to one worker.
- **Repeatable schema changes:** versioned migrations let local, test, and
  production environments follow the same ordered changes.
- **Easier troubleshooting:** application records and graph resume state have
  distinct owners and can be investigated separately.

These mechanisms reduce common failures; backups, secure credentials,
monitoring, and careful schema deployment are still needed.

## Project structure

```text
.
├── main.py                         # Streamlit application entry point
├── README.md                       # Project documentation
├── .env.example                    # Example environment configuration
├── requirements.txt                # Dependency list
├── pyproject.toml                  # Project metadata and dependency definitions
├── src/
│   └── app/
│       ├── agents/                 # LangGraph workflow and agent logic
│       ├── configs/                # Settings and environment validation
│       ├── constants/              # App defaults and company details
│       ├── database/               # PostgreSQL connections, models, and repositories
│       ├── guardrail/              # Deterministic and LLM-based input checks
│       ├── helpers/                # LLM factory and helper functions
│       ├── pipelines/              # Workflow entry points
│       ├── schemas/                # Pydantic state and output models
│       └── tools/                   # Placeholder/extension area for tools
├── alembic.ini                     # Alembic configuration
├── migrations/
│   ├── env.py                      # Migration environment
│   └── versions/                   # Versioned database schema changes
├── tests/
│   └── test_priority_escalation.py # Priority and escalation tests
└── uv.lock                         # Locked dependency versions
```

## Quick start

### Prerequisites

- Python 3.13 or later.
- `uv` installed.
- API credentials for at least one enabled LLM provider.
- An SMTP account or SMTP relay with permission to send email.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your actual GitHub repository URL and directory name.

### 2. Install dependencies

```bash
uv sync
```

### 3. Create your environment file

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

### 4. Configure credentials

Open `.env` and provide the values required by your setup. See [Environment configuration](#environment-configuration).

### 5. Create or update the PostgreSQL schema

```bash
uv run alembic upgrade head
```

The LangGraph PostgreSQL checkpointer initializes its own checkpoint tables when
the graph is first created.

### 6. Start the Streamlit application

```bash
uv run streamlit run main.py
```

Streamlit will print a local URL in the terminal. Open that URL in your browser.

## Environment configuration

The application reads configuration from `.env` using `pydantic-settings`.

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection URL, for example `postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require` |
| `ENABLE_GEMINI` | Required for Gemini usage | Enables Google Gemini |
| `ENABLE_GROQ` | Required for Groq usage | Enables Groq |
| `GOOGLE_API_KEY` | If Gemini is enabled | Gemini API credential |
| `GROQ_API_KEY` | If Groq is enabled | Groq API credential |
| `SMTP_HOST` | Usually required | SMTP server host; default is `smtp.gmail.com` |
| `SMTP_PORT` | Usually required | SMTP server port; default is `587` |
| `SMTP_USER` | Yes for SMTP delivery | Sender email address |
| `SMTP_PASSWORD` | Yes for SMTP delivery | SMTP app password or provider authentication secret |
| `SUPPORT_EMAILS` | Yes for support notifications | JSON list of support recipient email addresses |
| `MANAGER_EMAILS` | Yes for manager escalation | JSON list of manager recipient email addresses |
| `MAX_EMAIL_REVISIONS` | No | Maximum email revision attempts; default is `3` |
| `LLM_MAX_RETRIES` | No | Maximum LLM request retries; default is `3` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | No | LLM request timeout; default is `60` |

The application validates that at least one LLM provider is enabled and that configured email recipient lists contain valid addresses.

Example recipient-list format:

```dotenv
SUPPORT_EMAILS=["support@example.com","support-lead@example.com"]
MANAGER_EMAILS=["manager@example.com"]
```

Use real values only in your local `.env` file. Do not commit `.env`, API keys, SMTP passwords, or other secrets to Git.

## Run the application

```bash
uv run streamlit run main.py
```

Expected user journey:

1. Open the Streamlit application.
2. Enter customer name and email.
3. Enter the customer's feedback.
4. Submit the form.
5. Review the analysis and generated email.
6. Choose an approval action when prompted.
7. If accepted, the application sends the email through SMTP.

The exact screen layout depends on the current Streamlit implementation.

## Run tests

Run the unit tests with Python:

```bash
python -m unittest discover -s tests -v
```

Or use the project environment managed by `uv`:

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover guardrails, routing, revision limits, and email-delivery behavior.

## Custom evaluation framework (no LangSmith)

The separate [`evaluation/`](./evaluation/) package evaluates the real
LangGraph workflow without changing production nodes. It uses the actual
`FeedbackState` input fields, the `sentiment` result, and the negative-feedback
`analysis` structure in `src/app/schemas/schema.py`. The graph does not emit a
standalone intent field: evaluation derives a coarse intent from the existing
sentiment/category values. Positive runs do not produce an `analysis` category,
so category accuracy is reported only for labeled negative cases.

The current graph has no LangChain tool nodes. Tool-like metrics therefore
measure the real support-ticket and internal-notification graph nodes and are
clearly labeled as such. Graph execution stops at the existing human-review
interrupt. PostgreSQL checkpointing and ticket/email persistence are replaced
with in-memory test doubles, and SMTP is always mocked; the customer email is
never sent.

Install the test extra if pytest is not already available:

```powershell
uv sync --extra test
```

Run the isolated unit/integration suite (no external model calls):

```powershell
uv run pytest -q
```

Run evaluation against the configured model providers. This sends only the
synthetic feedback in the evaluation datasets to the providers. Model use is
intentionally blocked unless `--allow-model-calls` is specified:

```powershell
uv run python -m evaluation --allow-model-calls
```

Add `--judge` to score relevance, helpfulness, empathy, accuracy, and clarity
with a structured LLM judge. Compare a run to a saved JSON report using
`--baseline evaluation/reports/evaluation-<run-id>.json`. Choose a `--version`
label for each model/prompt version. Reports include prediction scores,
execution errors, graph node traces, guardrail false acceptance/rejection,
simulated side-effect counts, and per-case regression details. JSON and CSV
files are written under `evaluation/reports/`, which is git-ignored.

Folder guide:

- `evaluation/datasets/`: synthetic classification and guardrail examples with
  expected labels.
- `evaluation/models.py` and `data.py`: validated records and dataset loading.
- `evaluation/adapter.py`: actual graph integration with memory checkpointing
  and mocked SMTP/database operations.
- `evaluation/metrics.py`, `guardrails.py`, and `judge.py`: deterministic
  metrics, guardrail confusion metrics, and optional LLM scoring.
- `evaluation/runner.py`, `reports.py`, and `regression.py`: execution,
  JSON/CSV reports, and version-to-version comparison.
- `tests/test_evaluation_*.py`: offline behavior, graph, failure, and
  concurrency checks.

The CLI prints a summary shaped like this (metric values depend on model
outputs):

```json
{
  "sentiment_accuracy": {"accuracy": 0.9, "evaluated": 8},
  "category_accuracy": {"accuracy": 0.8, "evaluated": 6},
  "conditional_route_accuracy": {"accuracy": 1.0, "evaluated": 8},
  "executions": {
    "total": 8,
    "completed": 8,
    "guardrail_rejected": 0,
    "failed": 0,
    "tool_nodes_available": 0
  }
}
```

The evaluation runner disables LangSmith tracing in its process. It never
modifies the production pipeline or weakens its guardrails. Guardrail tests use
the separate synthetic cases in `evaluation/datasets/guardrails.json`. Existing
tests plus evaluation integration tests cover retry configuration, database
failure propagation, email idempotency, and concurrent send claims. The
concurrency test uses an in-memory atomic claim; validating PostgreSQL locking
under real database contention remains an opt-in infrastructure test.

## State, checkpointing, and limitations

### LangGraph state and checkpoints

The workflow uses LangGraph state to carry customer details, analysis results, priority information, email content, and review decisions between nodes.

LangGraph checkpoints are stored in PostgreSQL, so pauses and resumes at the human
approval `interrupt()` survive process restarts and work across application
workers using the same database. Application ticket and email-delivery records
are managed separately through SQLAlchemy repositories and Alembic migrations.

Ticket rows are unique by `thread_id`, and each generated ticket ID remains
deterministic. Email delivery claims are unique by idempotency key and committed
before SMTP is called, preventing concurrent workers from sending the same
message simultaneously. SMTP is an external system, so a process failure after
the provider accepts a message but before the database records `sent` cannot be
made transactionally exactly-once; an unresolved `sending` record is blocked
from automatic retry to avoid a possible duplicate.

### Current project boundaries

- The application is designed for an internal customer-support workflow.
- It does not currently integrate with an external ticketing platform; ticket creation is handled within the application's current workflow logic.
- Support-team and manager roles receive generated notifications; the current application does not expose inbound free-text forms for those roles.
- Web search and OpenAI integrations may appear in settings or dependencies, but they are not actively wired into the current graph flow.
- Email delivery requires valid SMTP configuration and provider permission.
- Keep `DATABASE_URL` in environment-managed secrets and use TLS for remote
  PostgreSQL connections.

## Screenshots and visual documentation

Screenshots help visitors understand the application before cloning it. Add real screenshots from your running Streamlit application to `docs/images/`.

Suggested repository layout:

```text
docs/
└── images/
    ├── streamlit-feedback-form.png
    ├── feedback-analysis-result.png
    ├── priority-and-ticket-result.png
    └── human-email-review.png
```

Recommended screenshots:

1. **Feedback form:** show the customer input fields and feedback submission area.
2. **Analysis result:** show sentiment and complaint analysis output.
3. **Priority and escalation:** show a representative P1/P2/P3 result and related ticket/notification information, if displayed by the UI.
4. **Human review:** show the generated email and the reviewer actions.

After adding the actual screenshots, insert them into this README with Markdown:

```markdown
![Customer feedback form](docs/images/streamlit-feedback-form.png)

![Feedback analysis result](docs/images/feedback-analysis-result.png)

![Priority and ticket result](docs/images/priority-and-ticket-result.png)

![Human email review](docs/images/human-email-review.png)
```

Only add screenshots that reflect the current application. Do not create placeholder screenshots that could be mistaken for real application output.

## Possible future enhancements

The following are ideas for future development, not claims about features already implemented:

- Integrate with an external ticketing system such as Jira or Zendesk.
- Add authentication and role-based access for reviewers.
- Add audit logs for approval decisions and email delivery.
- Add observability for graph execution, model latency, retries, and failures.
- Add automated tests for guardrail bypass attempts and prompt-injection cases.
- Add a dashboard for feedback volume, sentiment trends, and priority distribution.
- Add evaluation datasets to measure classification and response quality.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
