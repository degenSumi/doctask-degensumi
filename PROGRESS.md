# Progress and assumptions

Log of what is built, what is not, and the calls made along the way.

---

## 2026-08-10

### The five non-cuttable behaviours

| # | Behaviour | Where it lives | Proven by |
|---|---|---|---|
| 1 | Works in steps we can watch | `graph/` — seven named stages, every decision appended to `stage_log` | `test_an_escalation_is_recorded_by_name` |
| 2 | Survives being stopped | LangGraph checkpointer, written after every stage | `test_resuming_redoes_no_model_work` |
| 3 | A human holds the gate | `interrupt_before=["commit"]`; decisions are per item | `test_committing_undecided_items_is_refused`, `test_rejecting_one_leaves_the_rest` |
| 4 | A machine can drive it | HTTP (`api/app.py`) and MCP (`mcp_server.py`), approval an explicit call in both | `test_a_program_can_drive_the_whole_flow`, `test_an_agent_can_drive_a_run_to_completion` |
| 5 | Never bluffs | `domain/grounding.py` plus constructor invariants | `test_every_citation_points_at_text_that_is_really_there` |

Three decisions genuinely change the path, each named in the log: **skip** an
unreadable document, **retry** a document whose opening passage carried nothing,
**escalate** a document that addresses the system.

### Of the five that separate strong from competent

| # | Behaviour | State |
|---|---|---|
| 6 | A stranger can run it | `uv sync && uv run analyst run corpus`, or `docker compose up` for the API alone. No key needed. |
| 7 | It proves itself | 68 tests, no key, no database, no network |
| 8 | Takes no orders from documents | 12 deterministic patterns; quarantined sources still read with hostile passages removed |
| 9 | Two runs stay two runs | separate threads in the checkpointer; tested |
| 10 | Knows what it cost | per-stage deltas summed in state; a resumed run keeps what the first attempt spent |

### Verified against the corpus

```
5 sources classified from content, not filenames
15 grounded facts, each located in the text it claims to come from
2 supersessions   amendment raised the fee, extended payment terms
2 conflicts       both on INV-1002, cited to clause and line
3 findings        one missing purchase order, two billing mismatches
1 quarantine      file note matching 8 hostile patterns
```

### Verified in the container

`docker compose up` brings up the API, and the whole flow was driven through
HTTP with no shell access to the container:

```
POST /runs                    -> 14 items awaiting a decision, nothing committed
POST /runs/demo/commit        -> HTTP 409, names all 14 undecided items
POST /runs/demo/decisions     -> rows approved, findings deferred
POST /runs/demo/commit        -> committed: true, 8 rows, each cited
```

### Bugs the corpus and the tests caught

1. **Four conflicts where there should be two.** INV-1001 is dated 3 April and
   the amendment takes effect 1 May, so it was being judged against terms that
   did not yet exist. Fixed by keeping a **timeline** per term and measuring each
   invoice against what applied on its own issue date. An invoice with an
   unreadable date is now not judged at all rather than judged against a guess.
2. **A retry loop that would never terminate.** The retry marker was read but
   never cleared. Now raised to spent *before* the retry runs, so a document is
   read again at most once whatever the outcome.
3. **A false `AMENDMENT-EFFECTIVE-DATE` finding** on a document that states its
   effective date plainly. The extractor had no `effective_date` subject, so the
   rule reported a failure that was really a gap in the extractor.
4. **Cost reported as zero after a resume.** It was read from the model object,
   which is fresh in a new process. Now accumulated in state as per-stage deltas.
5. **An update cost what a full run cost.** Classification and extraction ran
   over every source on every pass and then discarded the duplicates. Now both
   skip what they have already done, and a test fails if an arrival ever costs
   more than a third of a full run. Measured: 4 calls against 20.
6. **An update silently lost a finding.** Findings about documents nobody
   re-examined disappeared, because the register is rebuilt from facts on every
   reconcile. They are now carried in state independently of the register.
7. **The container image would not build.** `pyproject.toml` names `README.md`
   as the package readme and the Dockerfile never copied it. Worth noting that
   the failure was invisible at first because the build ran through a pipe, so
   the shell reported the exit code of `tail` rather than of the build.

### Decisions made

- **Domain: vendor contracts, amendments, invoices.** Conflicts are crisp and
  checkable, so "notices where the documents disagree" is demonstrable.
- **Supersession vs conflict is the central distinction.** An amendment changing
  a term is the documents working as intended. An invoice disagreeing with the
  term in force on its date is a person's problem. Collapsing the two would bury
  the second in noise.
- **The model reads; the code decides.** The model is asked only what a passage
  says. Whether the pile agrees is settled in `reconcile.py`, over facts already
  grounded in text.
- **A quote that cannot be located is discarded**, not recorded with a caveat.
  Exact match, then whitespace-insensitive, and no further — fuzzier matching
  would start accepting quotes the source does not contain.
- **Hostile text is detected deterministically.** A component that can be
  reasoned with can be reasoned out of its own defence.
- **SQLite checkpointing.** A fresh clone runs with no database to install, and
  run state survives a restart under `docker compose` on its own volume.
- **The pattern model is the default.** The system is complete without a key;
  a key swaps one adapter.

### Assumptions logged

- Dates are written in the formats these documents use. An unrecognised date
  returns nothing rather than a guess.
- Two agreement documents of equal standing that disagree are reported as a
  conflict rather than resolved by date. Only an amendment supersedes.
- `SourceKind.UNKNOWN` is read for facts but never allowed to supersede.
- Approving a conflict, finding or quarantine records that a person saw it. It
  changes no row, because those are reported rather than applied.

### Cut, and why

- **pgvector retrieval.** Extraction reads the opening passage and retries with
  the whole document instead of retrieving the relevant part. Adequate at this
  corpus size, not at a hundred pages. Cut in favour of the five non-cuttable
  behaviours being complete, and Postgres was dropped with it rather than left
  running unread.
- **Watching a folder.** A document that arrives is added with `analyst ingest`,
  which reads only that document and re-checks only what it touched. Nothing
  polls, so a new file waits until someone names it.
- **A web review interface.** The gate exists at a terminal, over HTTP and over
  MCP. A browser is another adapter.

### Blocked on

- **A model API key** for a live demo. Everything runs on the pattern model.
  Google AI Studio or Groq, free, no card.
- **Repository name.** Using `doctask-degensumi`; the brief's example suggests a
  personal name rather than a handle.

---

## 2026-08-18

### Run against a real model for the first time

Google AI Studio and Groq, both on the free tier. Three defects only a hosted
model could expose, none of which the pattern stand-in could ever have shown:

- **Subjects were sent as bare names.** Asked to choose between `payment_terms`
  and `invoice_terms` with nothing to tell them apart, both models filed an
  invoice's terms under `payment_terms` — the words printed on the page. The
  reconciler then had no invoice term to compare against the term in force, so a
  real conflict silently never fired. Subjects now carry a description each, and
  the prompt says to choose on the description rather than the wording.
- **Register rows assumed a bare value.** The row templates read
  `"Be bound by the laws of {value}"`, which the regex stand-in satisfied because
  its capture groups return the value alone. A model returns the phrase it read,
  so rows came out as "Be bound by the laws of the laws of England and Wales".
  The port never promised a shape; `RowSpec` now takes the part the sentence
  needs rather than whatever arrived.
- **Both models under-reported.** Before the fixes: 1 conflict and 2 findings
  against the stand-in's 2 and 3. After: identical on both.

The lesson worth keeping is that a fake which is convenient can hide a contract
that was never written down. These were bugs in the seam, not in either model.

### Found by running it rather than testing it

- **The review prompt printed no keys.** `[a] approve  [r] reject` is Rich markup;
  the tags were parsed and stripped, leaving four words and no way to know what
  to press. Unrecognised input also defaulted to defer, so a stray keystroke was
  indistinguishable from a decision. It now re-asks.
- **Two thirds of the register were invisible.** Conflicts and findings were
  counted in the footer and never shown. A conflict now prints both documents
  side by side with each value, its clause and line, and its own words — which
  is what someone settling it actually needs. Reported, not resolved, stays the
  position, but a report nobody can act on is not a report.
- **A finished run had no route back to it.** Run ids were random, so resuming or
  adding a document meant scrolling for a hex string. A run is now named after
  the corpus and rules it reads, on all three surfaces: the same documents land
  on the same run and are paid for once. `--new` starts over deliberately.
- **Container state died with the container.** `runs.db` had no volume; Postgres
  was masking it by existing. Now on its own volume, proven by restarting the
  container mid-run and resuming.

### Cut since

- **Postgres.** It was declared in settings, passed to the container, and read by
  nothing: the checkpointer was always SQLite. What it existed to serve was
  pgvector retrieval, which is cut. Removed rather than shipped unread, along
  with three dependencies and a schema file nothing applied.
