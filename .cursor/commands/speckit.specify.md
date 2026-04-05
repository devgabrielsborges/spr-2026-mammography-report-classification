---
description: Create a research or ML project specification using the Stargazer preset
scripts:
  sh: .specify/scripts/bash/create-new-feature.sh "{ARGS}"
  ps: .specify/scripts/powershell/create-new-feature.ps1 "{ARGS}"
---


<!-- Source: stargazer -->
## User Input

```text
$ARGUMENTS
```

You are creating a specification for a Statistics, Machine Learning, Data Science, or research project using the **Stargazer** preset.

Given the project description above, follow this process:

1. **Create the feature branch** by running the script:
   - Bash: `{SCRIPT} --json --short-name "<short-name>" "<description>"`
   - The JSON output contains BRANCH_NAME and SPEC_FILE paths.

2. **Read the spec-template** to see the sections you need to fill.

3. **Write the specification** to SPEC_FILE by filling each section according to these guidelines:

   - **Problem Statement**: Frame it as a clear question or optimization objective. What is the measurable outcome of success?

   - **Hypothesis**: State H₀ and H₁ formally. For ML projects, describe the expected input-output relationship. For statistical studies, state what effect or difference you expect to observe.

   - **Background and Motivation**: Summarize relevant prior work, existing solutions, and the gap this project fills. Include references to papers, benchmarks, or production baselines.

   - **Data Requirements**: Be specific about sources, volume, format, labeling status, and sensitivity classification. List assumptions about the data (distribution, stationarity, independence).

   - **Methodology**: Describe the modeling approach or statistical method. Justify why it fits the problem. Include planned feature engineering and the baseline to beat.

   - **Evaluation**: Define primary and secondary metrics with target thresholds. Specify the validation strategy (cross-validation, temporal split, etc.) and how statistical significance will be determined.

   - **Scope and Constraints**: Clearly separate what is in scope from what is not. State compute, data, and timeline constraints.

   - **Risks**: Identify data quality risks, model underperformance risks, and distribution shift risks with likelihood, impact, and mitigation strategies.

   - **Deliverables**: List concrete outputs — trained models, reports, notebooks, reproducibility artifacts.

   - **References**: Cite relevant papers, documentation, or prior work.

4. **Validate** the specification:
   - Every metric in Evaluation has a justification
   - The hypothesis is falsifiable
   - Data requirements are concrete enough to start acquisition
   - At least one baseline is defined
   - Risks include at least one data-related and one model-related entry