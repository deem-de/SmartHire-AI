# SmartHire AI

SmartHire AI is a resume parsing and interview assistant project. It combines a
BERT-based named entity recognition pipeline with a FastAPI backend and a React
demo interface. The app can extract candidate information from uploaded resumes,
compare baseline and section-aware parsers, generate interview questions, and
evaluate interview answers.

## Features

- Upload and parse PDF, DOCX, or TXT resumes.
- Extract candidate profile fields such as name, contact details, education,
  skills, companies, designations, and experience highlights.
- Compare baseline BERT and section-aware BERT resume parsers.
- Generate interview questions from the extracted resume profile.
- Evaluate interview answers with AI feedback or an offline fallback.
- Serve the backend API and React UI from one FastAPI application.

## Project Structure

```text
SmartHire_Model_And_Interface_Code/
+-- 01_Model_Pipeline/
|   +-- main.py
|   +-- data_loading.py
|   +-- preprocessing.py
|   +-- training.py
|   +-- evaluation.py
|   +-- trained_models/
+-- 02_Interface_Application/
|   +-- app_server.py
|   +-- react-ui/
|   +-- outputs/
|   +-- start_smarthire.ps1
|   +-- requirements.txt
+-- .gitattributes
+-- .gitignore
```

## Requirements

- Python 3.10 or newer
- Git LFS, required for the saved `.safetensors` model files
- Optional: `GEMINI_API_KEY` or `OPENAI_API_KEY` for AI-generated interview
  questions and answer evaluation

Install Git LFS before cloning, or run `git lfs pull` after cloning:

```powershell
git lfs install
git clone https://github.com/Ghalaxxx/SmartHire-AI.git
cd SmartHire-AI
git lfs pull
```

## Run the Interface Application

From the project root:

```powershell
cd 02_Interface_Application
python -m pip install -r requirements.txt
.\start_smarthire.ps1
```

Then open:

```text
http://127.0.0.1:3001
```

The FastAPI server also exposes:

- `GET /api/health`
- `GET /api/bootstrap`
- `POST /api/process-resume`
- `POST /api/start-interview`
- `POST /api/evaluate-answer`
- `POST /api/final-summary`

## AI Interview Settings

The app works without an AI key by using offline template-based interview logic.
For AI-generated questions and feedback, set one of these environment variables:

```powershell
$env:GEMINI_API_KEY="your_gemini_key"
```

or:

```powershell
$env:OPENAI_API_KEY="your_openai_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

If `GEMINI_API_KEY` is present, the app uses Gemini. Otherwise, it uses OpenAI.

## Run the Model Pipeline

The model pipeline trains and evaluates the baseline and section-aware BERT NER
models.

```powershell
cd 01_Model_Pipeline
python -m pip install -r requirements.txt
python main.py --dataset_path data\resume_ner_training_data.json
```

Useful options:

```powershell
python main.py --sample_size 50 --epochs 1
python main.py --skip_training
python main.py --output_dir trained_models
```

## Data and Model Files

Saved model weights are tracked with Git LFS through `.gitattributes`.

Training data folders such as `data/` and `datasets/` are ignored by Git. If you
want to train from scratch, place the dataset locally and pass its path with
`--dataset_path`.

## Notes

- The runtime app expects model files under
  `02_Interface_Application/outputs/`.
- The model pipeline stores training outputs under
  `01_Model_Pipeline/trained_models/` or the directory passed to
  `--output_dir`.
- Resume uploads are temporarily written under
  `02_Interface_Application/outputs/temp_uploads/`.
  ## Team Members
- Deem Alrashoud
- Ghala hazazi
- renad hazazi
- rahaf saadAldeen
- mace marzogi

## My Contribution
- Model development and testing
- Resume parsing pipeline
- Evaluation and experimentation
- Documentation
