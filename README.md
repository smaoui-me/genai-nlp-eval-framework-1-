## Data overview

This repository contains several customer-support ticket CSV datasets (including language-specific files in the
`customer-support-tickets-dataset/` folder). The primary, cleaned dataset is located at
data/dataset_tickets_clean.csv and includes fields such as subject, body, answer, type, queue, priority, language,
version, and tag columns.

We will primarily use data/dataset_tickets_clean.csv as the main source for experiments and evaluations.

## Ticket extraction evaluation

For extraction, we evaluate on `data/processed/ticket_extraction_eval.csv`. Each row contains:

- `ticket_id`: unique ticket identifier
- `text`: full ticket text used as model input, which is a conctenation of the subject and body from main dataset in this format: subject + "\n\n" + body
- `gold_type`: gold issue type
- `gold_queue`: gold support queue
- `gold_tags`: gold tag list

The allowed label space is loaded from `data/processed/ticket_allowed_labels.json`, which contains the full lists of
valid `types`, `queues`, and `tags`.

### Why we do candidate tag selection

The tag inventory is large, so we do not send all tags to the LLM for every ticket. Instead, for each ticket we build
a smaller candidate-tag list:

- tags whose words or phrases appear in the ticket text
- frequent tags from the dataset as fallback
- a configurable limit, typically around 30 to 50 tags

This keeps prompts smaller and more stable while still covering most gold tags.

### What the model predicts

For each ticket, the model predicts:

- exactly one `type`
- exactly one `queue`
- zero to a few `tags`

We save a JSONL debug record per ticket with the input text, candidate tags, raw LLM response, parsed output,
validated output, JSON validity flags, and invalid labels if any.

### Validation before scoring

Predictions are validated against the allowed labels:

- predicted `type` must be in allowed types
- predicted `queue` must be in allowed queues
- predicted `tags` must be in allowed tags

Invalid labels are removed and flagged in the debug output rather than silently accepted.

### Metrics

We score extraction at three levels:

- `type`: accuracy and macro F1
- `queue`: accuracy and macro F1
- `tags`: micro precision, micro recall, and micro F1 using predicted vs. gold tag sets

We also report:

- `candidate_tag_recall`: how often gold tags were present in the candidate shortlist
- `invalid_json_rate`: fraction of tickets where the model output could not be parsed as valid JSON
- `invalid_label_rate`: fraction of tickets containing unsupported labels
- `evidence_valid_rate`: fraction of evidence snippets that appear in the original ticket text

### Error analysis outputs

After each run, we save:

- a scores CSV with the aggregate metrics
- an errors CSV with ticket-level mismatches for type, queue, and tags
- a JSONL file with full per-ticket debug information

For debugging and iteration, we usually run the pipeline first on a small subset such as `max_rows = 20` before
scaling to a larger sample.
