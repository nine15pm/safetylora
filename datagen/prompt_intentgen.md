I am working on creating a dataset to train AI models to have safe outputs for Teen users. Below is the safety policy.

{Policy}

Based on the policy, your task is to generate intent seeds for each category so I can then sample intents to generate a large synthetic dataset.
- Intent seeds should be diverse but realistic and cover the most likely teen user scenarios in that category
- Intent seeds are descriptions of intent, not literal prompts
- Each intent seed should be succinct
- Each intent seed should be categorized as adversarial, borderline, or benign
- Mix your intent seeds to diversely cover severity, pressure (e.g. none, urgent, peer, etc.), context (e.g. school,home,online,party,sports,etc.), and adversarial chat techniques (e.g. roleplay-bait,"educational" pretext,fictional wrapper,jailbreak wrapper,etc.).

Examples of intent seeds:
- Curiosity about puberty
- Request for romantic roleplay with a crush
- Planning a fight after school
- Asking how to make or hide a weapon

Generate 10-15 intent seeds in each of adversarial, borderline, and benign categories for policy area {N}.