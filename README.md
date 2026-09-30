````markdown
# 📐 AI Quantity Takeoff

AI-powered MVP for extracting dimensions and basic quantities from architectural PDF drawings.

## Features

- Upload architectural PDF drawings
- Preview drawing pages
- AI vision analysis
- Extract visible dimensions
- Identify rooms, walls, doors, windows, beams, columns, slabs and stairs
- Calculate basic quantities
- Editable quantity takeoff table
- Export to Excel
- Anti-hallucination rules
- Confidence and verification status

## Quantity formulas

### Area

Nos × Length × Width

### Volume

Nos × Length × Width × Height

### Length

Nos × Length

### Number

Nos

Dimensions are converted from millimetres to metres before calculation.

## Local installation

```bash
pip install -r requirements.txt
streamlit run app.py
````

## Streamlit Cloud

Deploy `app.py` from GitHub.

Then go to:

App Settings → Secrets

Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GROQ_MODEL = "qwen/qwen3.8-27b"
```

Never upload your API key to GitHub.

## Important

This is an AI-assisted quantity takeoff MVP.

Always verify extracted dimensions against the original architectural drawing before using quantities for:

* Procurement
* Tendering
* Billing
* BOQ
* Contractual measurements

```
```
