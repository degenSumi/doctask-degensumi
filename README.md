# analyst

Reads a pile of related documents, builds one register where every row cites the
text it came from, checks it against rules you supply, and surfaces the places
the documents disagree rather than picking a side.

Nothing reaches the register without a person deciding on it, item by item.

---

## The problem

A vendor relationship accumulates paper: a master agreement, amendments that
change its terms, invoices raised against it, correspondence about all three.
They describe the same arrangement and they do not quite agree.

Somebody has to know what is currently true. Which fee applies this month. Which
payment terms are in force. Whether the invoice that arrived on Tuesday matches
the agreement as it stood when it was raised.

The trap is that a naive reading gets this **confidently wrong**. An amendment
raising a fee looks exactly like a contradiction, and an invoice raised before an
amendment took effect looks exactly like a breach. Report both as problems and
the real problem is buried in noise.

This system distinguishes them:

- an amendment changing a term is a **supersession** — the documents working as intended
- an invoice disagreeing with the term in force **on its own issue date** is a **conflict** — a person's problem

## What a run does

```
corpus/  ─►  ingest      read every readable document
             classify    decide what each one is, from content not filename
                         ├─ escalate: text addressed to the system is quarantined
             extract     pull values, each quoting the document
                         ├─ retry: nothing in the opening passage, read it all
                         └─ discard: a quote not in the document is dropped
             reconcile   apply amendments; find real disagreements
             examine     check against your rules
             propose     prepare every item for review
                    ─────  the run stops here  ─────
             commit      apply only what was approved
```

## Run it

From a clone:

```bash
uv sync
uv run analyst run corpus
```

That is the whole system: it reads the corpus, stops, and asks about each item.

### With a model

Put a key in `.env`:

```bash
cp .env.example .env      # then set LLM_PROVIDER=groq and LLM_API_KEY=...
```

Groq and Google AI Studio both have a free tier with no card. Leave `LLM_MODEL`
blank and the cheapest model that provider offers which answers in JSON is used.
Supplying a key swaps one adapter and changes nothing else.

### Without one

It also runs with no key at all. The default is a pattern-backed stand-in that
reads the same passages a model would and answers in the same shape, quoting the
passage rather than paraphrasing it, so grounding is exercised for real. That is
what the tests use, which is why the suite needs no credentials and no network.

The caveat, plainly: the stand-in recognises the phrasing of the sample corpus
and nothing else. Give it the same clause worded differently and it reports
nothing rather than guessing. It stands in for the one question a model is asked
— what does this passage say — while everything the answer feeds into runs
identically either way.

### Or without a Python toolchain

```bash
docker compose up
```

The API is then on `http://localhost:8000`, docs at `/docs`. This runs the HTTP
surface only; the terminal and MCP surfaces need the clone above. Run state is
kept on a volume, so a restarted container resumes rather than losing the run.

Compose reads `.env` for the model settings, so a key placed there is used by
the container too. Without one it runs on the stand-in, like everything else.

### At a terminal

```bash
uv run analyst run corpus          # runs, then asks about each item
uv run analyst run corpus          # again: resumes if stopped, reports if finished
uv run analyst run corpus --new    # read the same corpus as a separate run

uv run analyst ingest <run-id> inbox/AMD-02-MSA-2026-014.md   # a document arrives
uv run analyst show <run-id>       # stages, register, cost
```

![The gate. The register above is what would be committed; below it each item is
put separately, carrying the clause and line it came from, and nothing moves
until a key is pressed.](docs/gate.png)

The register above the panels is a draft. It becomes the deliverable only from
the decisions taken on the items below it, one at a time.

**A run is named after what it reads**, not after a random id, so the second
command is the first one again. A run stopped part way continues; a run already
decided is reported rather than repeated, and any document sitting in the folder
that it never read is named. Reading the same corpus twice by accident is the
easiest way to spend twice.

### From a program

```bash
curl -X POST localhost:8000/runs -H 'content-type: application/json' -d '{}'
curl localhost:8000/runs/<id>/proposals
curl -X POST localhost:8000/runs/<id>/decisions -H 'content-type: application/json' \
     -d '{"decisions":[{"proposal_id":"conflict:0","decision":"approve"}]}'
curl -X POST localhost:8000/runs/<id>/commit
```

Posting the same corpus twice returns the run that already read it, with **200**
rather than 201, and no second reading. `{"fresh": true}` reads it again as a
separate run.

### From an agent

```bash
uv run python -m analyst.mcp_server
```

Seven MCP tools: `start_run`, `get_run`, `get_stages`, `list_proposals`,
`decide`, `commit`, `get_register`.

**A run is named the same way in every surface.** The terminal, HTTP and MCP all
derive it from the corpus and rules being read, so the same documents land on the
same run whichever door they arrive through, and are paid for once.

**The gate is in every surface.** `commit` is refused while any item is
undecided and the reply names the outstanding ones. There is no flag, endpoint or
tool that writes to the register without decisions.

## What it does on the sample corpus

Five synthetic documents: an agreement, an amendment, two invoices, a file note.

![A committed register. Every row names the clause and line it came from, and
the two conflicts below it show both documents side by side with what each one
says in its own words.](docs/register.png)

```
5 sources classified from content
19 grounded facts, each located in the text it claims to come from
2 supersessions   the amendment raised the fee and extended payment terms
2 conflicts       both on INV-1002, cited to a clause and a line
3 findings        one missing purchase order, two billing mismatches
1 quarantine      a file note matching 8 hostile patterns
```

**INV-1001 produces no conflict.** It is dated 3 April; the amendment takes
effect 1 May. It was correct when it was raised, and the system measures it
against the terms that applied then rather than the terms that apply now.

## Claims, and where each is proven

98 tests, no key, no database, no network.

| Claim | Test |
|---|---|
| Stages can be watched, decisions named | `test_an_escalation_is_recorded_by_name` |
| A supersession is not a conflict | `test_a_supersession_is_distinguished_from_a_conflict` |
| An invoice is judged against its own date | `test_an_invoice_predating_an_amendment_is_not_reported_as_wrong` |
| Nothing is asserted without a source | `test_every_citation_points_at_text_that_is_really_there` |
| A stopped run resumes and redoes nothing | `test_resuming_redoes_no_model_work` |
| The same corpus twice is not paid for twice | `test_a_finished_run_is_reported_not_repeated` |
| And not over HTTP or MCP either | `test_the_same_corpus_is_not_read_twice` |
| Nothing commits without a decision | `test_committing_undecided_items_is_refused` |
| Rejecting one leaves the rest | `test_rejecting_one_leaves_the_rest` |
| A program can drive the whole flow | `test_a_program_can_drive_the_whole_flow` |
| Documents do not give orders | `test_its_instructions_are_not_obeyed` |
| Two runs stay two runs | `test_two_runs_stay_separate` |
| A clean corpus honestly reports nothing | `test_a_clean_corpus_reports_no_findings` |
| A row reads the same whichever model wrote it | `test_both_phrasings_give_one_sentence` |
| An update asks only about what moved | `test_it_asks_only_about_what_moved` |

```bash
uv run pytest
```

## How it never bluffs

A model reports a value and the text it read it from. That text is then **located
in the source**. If it cannot be found, the answer is dropped rather than
recorded — a value whose quote is not in the document is invented, however
plausible it reads.

Matching is exact, then whitespace-insensitive, and stops there. Fuzzier matching
would begin accepting quotes the source does not contain, which is the failure
the step exists to prevent.

Above that, the types refuse to hold an uncited claim:

```python
def __post_init__(self) -> None:
    if not self.support:
        raise ValueError(f"the fact {self.subject!r} has no source span; it cannot be asserted")
```

`Fact`, `Finding`, `Obligation` and `Proposal` all do this. An uncited claim is
not rejected later — it cannot be constructed.

## How it resists documents that give orders

A source describes the world. Text that instead addresses the software reading
it, redefines its instructions, or commands its approval workflow is quarantined
and reported.

Detection is deterministic rather than delegated to a model, because a component
that can be reasoned with can be reasoned out of its own defence. False positives
are the direction worth failing in: a wrongly flagged passage costs a person one
glance; a missed one puts hostile text into a prompt.

A quarantined document is still read, with the hostile passages removed, so one
sentence cannot hide the ordinary content around it. The sample corpus contains a
file note demanding that a discrepancy go unreported — and the discrepancy is
reported.

## Design

```
domain/     pure. no network, no files, no database.
  models      Source, Span, Fact, Conflict, Rule, Finding, Obligation, Register, Proposal
  grounding   locates a claimed quote, or discards the claim
  rules       patterns identifying text aimed at the system
  reconcile   supersession vs conflict, against the term in force on the date
  register    facts to rows, driven by a table
  examine     rules to findings

ports/      interfaces
  language_model   what a model is asked, and what it may answer

adapters/
  fake_model    patterns; the default, so it runs with no key
  hosted_model  Gemini or Groq behind the same seam
  loader        .md .txt .docx .pdf into text

graph/      LangGraph: stages, the retry edge, the checkpointer
api/        HTTP
mcp_server  MCP tools
```

The domain layer imports nothing from adapters. The model is asked only what a
passage says; whether the pile agrees is decided in `reconcile.py`, in code, over
facts that are already grounded.

**Calls made about the pile**, where a different answer was defensible:

- Only an amendment supersedes. Two agreements of equal standing that disagree
  are a conflict for a person to settle, not something to resolve by date.
- A document whose kind could not be worked out is still read for facts, but is
  never allowed to supersede another document.
- A date written in a format these documents do not use returns nothing rather
  than a guess.
- Approving a conflict, a finding or a quarantine records that a person saw it.
  It moves no row, because those are reported rather than applied.

## Formats

`.md`, `.txt`, `.docx`, `.pdf`. Anything else is refused by name rather than read
as bytes and half-understood. Text only — layout and images are discarded,
because every claim is located by character offset and the text a citation points
into has to be the text the model was shown.

## Limitations

- **Extraction subjects are a fixed vocabulary.** The system reports the terms it
  knows to look for. A document about something else is read and produces
  nothing, which is honest but not general.
- **Dates are read in the formats these documents use.** An unrecognised date
  returns nothing rather than a guess, so an invoice with an unreadable date is
  not judged at all.
- **Retrieval is not implemented.** Extraction reads the opening passage and
  retries with the whole document rather than retrieving the relevant part.
  Retrieval would mean chunking each source, embedding the chunks and selecting
  by similarity, with citations still carried as character offsets into the
  original text. Adequate for documents of this size, and not for a
  hundred-page agreement.
- **Run state is checkpointed to SQLite, not Postgres.** The checkpointer is one
  line either way. What Postgres was there to serve was retrieval, which is cut,
  so the database went with it rather than being shipped unread.
- **No web review interface.** The gate is exposed at a terminal, over HTTP, and
  over MCP. A browser interface would be another adapter, not a rewrite.
- **An update is driven by hand, not watched for.** A document that arrives is
  added with `analyst ingest`, which reads only that document, re-checks only
  what it touched, and asks only about the rows that moved. Nothing polls a
  folder, so a new file waits until someone names it.
- **Concurrency is per run, not within one.** Documents are read in sequence.

## Cost

Every run reports what it spent, and a resumed run keeps what the first attempt
spent, because that work was paid for whether or not the process survived to use
it.

```
6 current rows · 2 conflicts · 3 findings · digest 9db93eab45f31596
model calls 20 · characters in 18506
```
