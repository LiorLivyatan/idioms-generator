# Confusing Context Variant Generator

A Python tool that generates confusing context variants for idiomatic expressions using AI language models. This tool is designed for linguistic research and natural language processing tasks that require ambiguous contexts for idiom comprehension testing.

## Features

- **Automated variant generation**: Creates multiple confusing context variants for each input sentence
- **BIO tag preservation**: Maintains proper BIO (Beginning-Inside-Outside) tagging for idioms
- **Flexible input options**: Process all sentences or specify a subset (random or sequential)
- **Multiple output formats**: Export results as JSON, CSV, or Pickle files
- **Interactive configuration**: User-friendly prompts for customizing generation parameters

## Requirements

- Python 3.7+
- OpenAI API access via GitHub Models or direct OpenAI API
- Input data in BIO-tagged TSV format

## Installation & Setup

1. **Clone or download** this repository

2. **Install dependencies** (handled automatically by main.py):

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:

   ```
   GITHUB_TOKEN_MODEL=your_github_token
   GITHUB_BASE_URL=https://models.inference.ai.azure.com
   ```

   Or for direct OpenAI API usage, modify `samples_generator.py` to use:

   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Prepare input data**:
   Ensure `test_english.tsv` is in the project directory. The file should contain BIO-tagged sentences in TSV format:

   ```
   After	O
   months	O
   of	O
   hard	O
   work	O
   he	O
   kicked	B-IDIOM
   the	I-IDIOM
   bucket	I-IDIOM

   ```

## Usage

### Quick Start

Run the main script and follow the interactive prompts:

```bash
python main.py
```

The script will:

1. Install required dependencies
2. Check environment configuration
3. Validate input files
4. Prompt for configuration options:
   - Number of variants per sentence (default: 3)
   - Number of sentences to process (default: all)
   - Random vs sequential sentence selection
   - Output format (json/csv/pickle)
5. Generate variants and save results

### Configuration Options

- **Variants per sentence**: How many alternative contexts to generate for each input sentence
- **Sentence selection**:
  - Process all sentences in the file
  - Process first N sentences (sequential)
  - Process N random sentences
- **Output format**: Choose between JSON (default), CSV, or Pickle formats

### Example Output

For input sentence: "After months of hard work, he finally kicked the bucket."

Generated variants might include:

1. "At the farm discussing pig slaughter, after months of hard work, he finally kicked the bucket."
2. "During a poker game with a rusty bucket nearby, after months of hard work, he finally kicked the bucket."
3. "While children played with pails and cans, after months of hard work, he finally kicked the bucket."

## Output Structure

The generated DataFrame contains:

- `original_sentence`: The input sentence
- `variant_number`: Variant identifier (1, 2, 3, ...)
- `variant_sentence`: Generated confusing context variant
- `tokens`: Tokenized sentence
- `tags`: BIO tags for each token
- `tag_ids`: Numeric tag identifiers
- `true_idioms`: Preserved idioms from the original sentence

## Files

- `main.py`: Main entry point with user interface and dependency management
- `samples_generator.py`: Core variant generation logic and BIO processing
- `test_english.tsv`: Input data file (BIO-tagged sentences)
- `requirements.txt`: Python dependencies
- `.env.example`: Template for environment variables
- `.gitignore`: Version control exclusions

## Advanced Usage

You can also import and use the generator programmatically:

```python
from samples_generator import generate_variants_dataframe

# Generate variants
df = generate_variants_dataframe(
    tsv_file_path="./test_english.tsv",
    num_variants=5,
    max_sentences=10
)

# Save results
df.to_json("my_variants.json", orient="records", indent=2)
```

## Troubleshooting

### Common Issues

1. **Missing API credentials**: Ensure environment variables are set correctly
2. **Input file not found**: Verify `test_english.tsv` exists in the project directory
3. **Dependency errors**: Run `pip install -r requirements.txt` manually if automatic installation fails

### API Configuration

The tool supports both GitHub Models and direct OpenAI API access. To switch to OpenAI:

1. Modify `samples_generator.py` line 31-35:

   ```python
   model = OpenAIChat(id="gpt-4o")
   ```

2. Set `OPENAI_API_KEY` in your `.env` file
