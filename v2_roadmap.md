# STOCKY V2 ARCHITECTURE ROADMAP

## OBJECTIVE
upgrade the temporal fusion transformer from a pure technical model to a macro-technical engine by integrating alternative data streams.

## NEW FEATURE PIPELINES
1. **MACRO ANCHORS:** integrate VIX (yfinance) and Federal Funds Rate (FRED API) into the past-observed tensor.
2. **SENTIMENT SCORING:** deploy a local NLP pipeline to parse daily financial headlines and append a -1 to 1 polarity score to the target ticker.
3. **EVENT FLAGS:** map binary earnings date flags into the future-known tensor.

## INFRASTRUCTURE CHANGES
- implement python-dotenv for API key management.
- expand temporalfusiontransformer input dictionaries to handle the increased dimensionality.