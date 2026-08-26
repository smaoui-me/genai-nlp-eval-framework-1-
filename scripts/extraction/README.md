# Few-NERD extraction scripts

These entry points run and compare the zero-shot structured, zero-shot
free-form, few-shot structured, few-shot free-form, and verifier-agent
location-extraction methods.

The default evaluation input is the compact public
`data/extraction/processed/fewnerd_location_test_1000.csv` file. Few-shot
methods use the three public demonstrations in
`sample_data/extraction/fewnerd_few_shot_examples.csv` and no longer depend
on an ignored raw training download.

For Docker commands that exercise and compare every method, see
[`docs/TESTING.md`](../../docs/TESTING.md).
