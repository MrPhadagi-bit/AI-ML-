# Repository Guide

This repository is intended to hold multiple AI and machine learning projects under one clean structure. The aim is to make every future addition easy to navigate, easy to run, and easy to present.

## Recommended Folder Layout

Each new project should use a consistent structure like this:

```text
projects/
|-- 01_project_name/
|   |-- README.md
|   |-- requirements.txt
|   |-- notebooks/
|   |-- src/
|   |-- data/
|   |   |-- raw/
|   |   |-- processed/
|   |-- models/
|   |-- reports/
```

## Naming Convention

- Use numbered folders to keep projects ordered.
- Use lowercase names with underscores.
- Keep project names short and descriptive.

Examples:

- `01_housing_price_prediction`
- `02_movie_review_sentiment_analysis`
- `03_image_classification_cnn`

## What Each Project README Should Contain

Every project-level `README.md` should include:

- Project title
- Problem statement
- Dataset description
- Tools and libraries used
- Setup instructions
- Training or execution steps
- Evaluation metrics
- Sample results
- Future improvements

## Documentation Standards

To keep the repo polished, each project should document:

- What problem it solves
- What data it uses
- How the model was trained
- How performance was measured
- What worked well and what can be improved

## Development Workflow

When starting a new project in this repo:

1. Create a new folder inside `projects/`.
2. Add a project-specific `README.md`.
3. Keep experiments in `notebooks/`.
4. Move reusable code into `src/`.
5. Save evaluation charts, reports, or screenshots in `reports/`.
6. Track dependencies in `requirements.txt`.

## Source Usage

The external repositories listed in the roadmap are intended as reference material only.

- Use them to understand datasets, workflows, and model choices.
- Keep the implementation in this repository original and well documented.
- Cite inspiration clearly when a project is based on an external idea or notebook.

## Good Repository Hygiene

To keep this repository maintainable:

- Do not commit large datasets unless necessary.
- Do not commit virtual environments.
- Do not commit trained model files unless they are intentionally versioned.
- Keep notebooks readable and paired with a short written summary.
- Prefer clean, reusable Python scripts for final implementations.

## Next Step Recommendation

A strong first build for this repository is `Predicting Housing Prices` because it establishes:

- A clean end-to-end ML workflow
- Dataset handling and preprocessing habits
- Model training and evaluation standards
- Documentation patterns that can be reused for the rest of the repo

