# Topic: fleet (which published packages exist, and what a family's models look like)

Every family node holds a `fleet.md` task file: the published loom-fun packages of that family with their geometry (layers,
hidden size, heads, MoE, drafter, native context), taken from each package's `llm_config.json`. Route with `--topic fleet` to pick
a reference model, a sample for a scope check, or to see which models a change to the family reaches.
