# Contract Intake & Group Allocation

A lightweight weekly training project built with **Python, Streamlit and SQLite**,
with optional live OpenAI analysis. Use synthetic data only.

**Current workshop rehearsal: MOCK AI.** This is a labelled local simulation
and needs no API key. Start with the [workshop walkthrough](#increment-4-workshop-rehearsal).

The user journey is:

**Upload a PDF → Submit case → See validation findings and allocated group.**

The submission screen has no API-key entry, model selector, or AI setup controls.
Developer configuration happens once outside the interface. The request and
original PDF are always saved before AI analysis starts.

## Run the app

Open PowerShell in the **uvdemo** folder:

    uv sync --locked
    uv run streamlit run app.py

Open http://127.0.0.1:8501. To restart after code or setup changes, press Ctrl+C
in that terminal, run the command again, and refresh the browser.

## Optional one-time OpenAI setup

For real AI analysis later, create **.streamlit/secrets.toml** inside the project.
If the file already exists, edit it rather than replacing its other settings.
Add your own key privately in your editor:

    OPENAI_API_KEY = "your-own-api-key"
    WORKBENCH_AI_MODE = "live"

The checked-in **.streamlit/secrets.toml.example** shows the format. The actual
**secrets.toml** file is excluded from Git. Never put the real key in app.py,
README, a screenshot, or chat. Restart the app after configuration.

OPENAI_API_KEY may alternatively be supplied by the server environment.
Environment variables take precedence over the secrets file.
The same principle applies after deployment: configure the hosting platform's
private secrets once; people submitting documents never enter a key.

The app uses **gpt-4.1-mini** by default. OPENAI_MODEL is an optional developer
override. API usage may incur charges. The app sends submitted fields and
extracted PDF text to OpenAI, using structured JSON output and store=False.

If the connection is not configured, **submission still works**. The saved case
waits for analysis. After setup, select it in **Review & Allocate** and click
**Retry analysis and group allocation**.

- [OpenAI API setup](https://developers.openai.com/api/docs/quickstart)
- [API authentication and server-side key storage](https://developers.openai.com/api/reference/overview#authentication)

## Use the three screens

1. **Submit Request:** enter fictional customer details, attach a readable PDF,
   and click **Submit case**. No API key is requested on this screen.
2. **Review & Allocate:** inspect AI findings and supporting excerpts. Retry an
   interrupted or failed attempt after resolving its cause.
3. **Case Tracker:** search by customer or case ID, filter by status and group,
   and track work on an allocated case.

## Increment 3: track work after allocation

The earlier increments save a mandatory PDF, analyse it, and automatically
allocate a group when required facts are resolved. This increment adds a small
follow-up workflow in **Case Tracker**:

| Current status | Button | New status |
|---|---|---|
| Allocated | Start work | In progress |
| In progress | Mark completed | Completed |
| Completed | Reopen case | In progress |

These are demonstration work statuses, not a claim that a contract has been
drafted or legally approved. Unallocated cases must resolve their analysis or
document issues first. Historical specialist assignments remain read-only.

Changing status does not call AI again or change the group, original request,
PDF, or analysis. Each change and its previous status are recorded in decision
history. Repeating the same update creates no duplicate history entry. A stale
page cannot overwrite a different current status; reload it before trying again.

The Python concept is an **allowed-transition dictionary**: each current status
maps to its permitted next statuses. The interface asks a plain Python function
to make a change. A SQLite **transaction** saves the new status and its history
entry together, or saves neither if an error occurs.

To try the feature, submit a synthetic request that can be allocated, then open
**Case Tracker**. Search for its case ID, click **Start work**, then **Mark
completed**. Refresh the browser and check that **Completed**, the allocated
group, and the history remain visible. **Reopen case** returns it to **In
progress**. Status and group filters can be combined with the search field.

## Increment 4: workshop rehearsal

The current local app is configured with `WORKBENCH_AI_MODE = "mock"` in the
private `.streamlit/secrets.toml` file. That file stays out of Git. On another
computer, create the same file with this one line and restart Streamlit:

    WORKBENCH_AI_MODE = "mock"

Check that the app shows **Development mode: MOCK AI simulation** before the
demonstration. An environment variable with the same name takes precedence.
Mock analysis is deterministic and recognises a few explicit labels; it does
not demonstrate live language-model reasoning. Live OpenAI verification was
deferred at your request on **11 October 2026**.

### Three ready-to-upload examples

Use **Start another request** between examples. The following PDFs are fictional
intake briefs for testing the upload workflow, not drafted or signed contracts.
They contain selectable text and do not contain expected routing answers.

| Field | 1. Matching request | 2. Conflicting segment | 3. Missing eligibility |
|---|---|---|---|
| PDF | [Government match](output/pdf/01-government-match.pdf) | [Segment conflict](output/pdf/02-segment-conflict.pdf) | [Missing eligibility](output/pdf/03-missing-eligibility.pdf) |
| Customer legal name | Fictional Harbour Research Department | Fictional Cedar Research Department | Fictional Acorn Services Pty Ltd |
| Reported customer segment | Government | Medium Business | Medium Business |
| Reported handling approach | Unknown | Unknown | Standard/Digital-eligible |
| Eligible for standard-template path? | Unknown | Unknown | Unknown |
| Expected status after submission | Allocated | Clarification needed | Clarification needed |
| Expected group | Government | Not allocated | Not allocated |

For examples **1 and 2**, paste this description:

> The contracting customer is a fictional government department requesting a services agreement.

For example **3**, paste this description:

> The contracting customer requests a services agreement. Digital eligibility has not been provided.

The same input values and expected outcomes are saved in
[demo/scenarios.json](demo/scenarios.json) for automated rehearsal tests.

### Five-minute presentation

1. **Explain the goal:** "This app captures a contract request, checks the supplied
   information, routes it to a group, and tracks progress. Today I am showing the
   clearly labelled mock simulation."
2. **Submit example 1:** upload its PDF and use the matching fields above. Show
   the saved case ID, Government group, original input, extracted text and
   supporting excerpts.
3. **Track work:** open **Case Tracker**, search for that case ID, then click
   **Start work** and **Mark completed**. Refresh and show the preserved group,
   Completed status and decision history.
4. **Submit example 2:** point out that the selected Medium Business segment
   conflicts with the Government description and PDF. The case is saved but
   stays unallocated. Saving intake is not approval of its contents.
5. **Submit example 3:** show why reporting standard-template handling does not
   establish digital eligibility. The app keeps the missing fact Unknown and
   requests clarification.

### Explain the code

| Part | Responsibility |
|---|---|
| Streamlit | Collect inputs and show the three screens |
| PDF extraction | Read selectable text; it does not perform OCR |
| AI analysis | In live mode, suggest a summary, categories and inconsistencies using one structured API call |
| Mock analysis | Simulate selected examples locally, with visible mock labels |
| Python rules | Check required facts, choose the group and permit progress transitions |
| SQLite | Preserve original input, PDF, analysis, decisions and history |

The Python concept for this increment is a **test fixture**: a known input and
its expected outcome. A **test double** replaces an external service during
testing. The rehearsal tests exercise the real Streamlit form and PDF files in
mock mode, and fail if they attempt to contact OpenAI.

Run the workshop rehearsal tests from the `uvdemo` folder:

    uv run python -m unittest discover -s tests -p test_demo_workshop.py -v

These tests use temporary databases and do not alter your saved demonstration
cases. Expected outcomes above are verified in mock mode; live model behaviour
has not been verified. The full test suite also includes simulated live SDK
success and failure responses, which are not real API calls.

Verified on **11 October 2026**: all **145 automated tests passed**, including
the three workshop PDF rehearsals. All three PDFs were rendered and visually
checked. No live OpenAI call was made.

The PDFs are already provided. If you intentionally edit the sample generator,
rebuild them with `uv run --link-mode copy --with reportlab python demo/generate_samples.py`.
ReportLab is only needed for rebuilding, not for running the app.

## Group rules

| Resolved facts | Group |
|---|---|
| Government | Government |
| Enterprise | Enterprise |
| Medium Business + Standard/Digital-eligible + explicit eligibility Yes | Digital |
| Medium Business + Assisted/Custom + explicit eligibility No | Medium Business |

These are configurable demo rules, not company policy. Digital means the
standard-template handling path, not electronic signing. Contract type alone
does not establish eligibility. The form lets you state eligibility explicitly
or choose Unknown.

AI summarises the document, suggests categories, and identifies discrepancies.
Python chooses the group. Missing required facts, detected conflicts, unreadable
PDFs, or failed analysis leave the case unallocated. Government and Enterprise
do not require a handling approach or digital eligibility. An unknown contract
type alone does not block routing.

New cases are allocated to groups only. There is no specialist selection or
capacity management in this workflow. Earlier saved cases and specialist
assignments remain preserved as history.

## Failures and data

The app retains the request and PDF when AI fails. New failures distinguish
authentication, account/model access, quota, rate limits, connectivity, and
service errors using safe messages. An older generic failure cannot reveal its
original cause retrospectively.

Refreshing a saved case does not make another model call. Retrying analysis is
an explicit button click. Repeated submission tokens and allocation attempts
cannot create duplicate cases or group allocations. Successful allocations are
fixed in this prototype; start a new request for corrected information.

Data is stored in **data/workbench.sqlite3**, excluded from Git. Tests use
temporary databases. WORKBENCH_DB_PATH can choose a different local database.

Supported uploads: nonempty, unencrypted PDFs with selectable text on every
page, up to 5 MiB, 25 pages, and 60,000 extracted characters. Analysis has a
100,000-character combined input limit. These are technical limits, not routing
criteria. OCR, legal verification, contract drafting, email, and production
integration are outside this project.

## Development without a live service

For an explicitly labelled **mock-AI simulation**, configure this in your local
secrets file or environment:

    WORKBENCH_AI_MODE = "mock"

Restart the app. Mock mode recognises a few explicit category labels and conflict
patterns; it does not understand arbitrary contract language. The interface
labels mock results, and no model API is called. It never silently replaces
failed live analysis. Remove this setting or change it to "live" for real AI.

## Learn the code and test

| File | What to learn |
|---|---|
| app.py | Streamlit inputs and the save → analyse → allocate sequence |
| src/uvdemo/intake.py | Python dictionaries and field validation |
| src/uvdemo/documents.py | Extracting text from a PDF |
| src/uvdemo/analysis.py | One structured AI call, evidence checks, safe exceptions |
| src/uvdemo/group_allocation.py | Saving group decisions atomically |
| src/uvdemo/allocation.py, route_analysis | The four Python routing rules |
| src/uvdemo/tracking.py | Allowed status transitions and filtering lists of cases |
| src/uvdemo/storage.py | SQLite persistence and duplicate prevention |

Run automated tests:

    uv run python -m unittest discover -s tests -v

Tests simulate OpenAI responses and require no API key. They do not verify an
actual account's credentials, billing, or network access.

To run just the increment 3 storage and transition tests:

    uv run python -m unittest discover -s tests -p test_tracking.py -v

Small exercise: using the missing-eligibility PDF, create a new request with
digital eligibility **Yes**. Write down the expected group before submitting,
then compare it with the result and explain which supplied fact changed it.
