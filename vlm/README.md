# MLX Vision Language Model Checker

This script provides image analysis and caption generation using MLX Vision Language Models (VLMs). It processes images, extracts metadata, and benchmarks multiple VLMs, outputting results in both HTML and Markdown formats.

## Features

- Processes images with one or more MLX Vision Language Models
- Extracts EXIF and GPS metadata from images
- Generates captions and diagnostics for images
- Outputs results as a colorized CLI table, HTML, and Markdown reports
- Handles timeouts, errors, and memory statistics
- Supports verbose/debug mode for detailed output

## Requirements

- Python 3.9+
- [mlx](https://github.com/ml-explore/mlx)
- [mlx-vlm](https://github.com/ml-explore/mlx-examples/tree/main/vision_language)
- [Pillow](https://python-pillow.org/)
- [huggingface_hub](https://huggingface.co/docs/huggingface_hub)
- [tzlocal](https://pypi.org/project/tzlocal/)

## Usage

```sh
python check_models.py --folder /path/to/images --models model1 model2 --output-html results.html --output-markdown results.md --verbose
```

## Arguments

- `--folder`: Folder to scan for images (default: `~/Pictures/Processed`)
- `--models`: List of model IDs or paths to process
- `--output-html`: Output HTML report file
- `--output-markdown`: Output Markdown report file
- `--prompt`: Custom prompt for the models
- `--max-tokens`: Maximum tokens to generate
- `--temperature`: Sampling temperature
- `--timeout`: Timeout in seconds for model operations
- `--verbose`: Enable verbose and debug output
- `--trust-remote-code`: Allow custom code from Hub models (SECURITY RISK)

## Output

- CLI: Colorized tables and logs
- HTML: `results.html` (default)
- Markdown: `results.md` (default)

## Notes

- The script uses ANSI color codes for CLI output. Colors may not display correctly in all terminals.
- Timeout functionality requires UNIX (not available on Windows).
- For best results, ensure all dependencies are installed and models are downloaded/cached.

## License

MIT
