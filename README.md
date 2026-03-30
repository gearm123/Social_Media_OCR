# Social Media OCR Translation Pipeline

This project extracts chat conversations from screenshots, reconstructs the conversation structure, and renders a translated chat-style output image.

It is currently optimized for Facebook Messenger-style conversations, but the pipeline is being kept modular so other chat platforms can be supported later.

## What It Does

Given a folder of input chat screenshots, the pipeline:

1. Cleans the images and prepares them for analysis.
2. Uses Gemini vision to transcribe the conversation structure from the screenshots.
3. Uses OCR hints to refine the source-language message text.
4. Runs a reference-resolution pass to improve who is speaking about whom before final English translation.
5. Extracts status-bar/header information separately.
6. Renders the final conversation as a clean chat image, along with debug comparison outputs.

## Current Pass Structure

1. `Pass 1`: source transcription and conversation structure
2. `Pass 2`: OCR-guided source-text polishing
3. `Pass 3`: reference resolution plus final English translation
4. `Pass 4`: status bar extraction for the rendered header

System metadata such as timestamps and call notices is handled separately from normal chat bubbles and merged back in only at the final rendering stage.

## Main Files

- `main.py`: pipeline entry point and orchestration
- `ocr_translate.py`: Gemini/OCR pipeline logic
- `chat_renderer.py`: rendered chat output
- `config.py`: paths, environment, and model configuration
- `pass1_bubble_input.txt`: manual bubble-count/order guidance for Pass 1

## Expected Local Structure

- `input_images/`: place source screenshots here
- `rendered_chat/`: generated images
- `result/`: prompt/debug text outputs
- `result_json/`: structured JSON debug outputs

The generated runtime folders are git-ignored so the repository stays clean. `input_images/` is kept as an empty tracked directory, but actual user screenshots should remain local and should not be committed.

## Running

Typical flow:

1. Put screenshots into `input_images/`
2. Update `pass1_bubble_input.txt` if needed
3. Run `python main.py`
4. Check `rendered_chat/`, `result/`, and `result_json/`

## Goal

The main goal of the project is accurate conversation reconstruction and translation, especially in difficult cases where OCR is noisy, chat UI artifacts are present, or subject/reference resolution is ambiguous.
