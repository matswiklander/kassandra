# Kassandra - Team Delivery Capacity Analyzer

[![Python](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Python CLI application that calculates 95% confidence intervals for team delivery capacity based on historical sprint data from markdown tables.

## Features

- ✅ Parses markdown tables containing sprint data (Sprint Name, Stories, Weeks)
- ✅ Calculates Stories/Weeks ratios for delivery capacity
- ✅ Computes 95% confidence intervals using Student's t-distribution
- ✅ Handles edge cases (insufficient data, zero variation, etc.)
- ✅ Provides user-friendly output with clear error messages
- ✅ Implements the algorithm specified in SPECIFICATION.md

## Installation

### Prerequisites
- Python 3.6+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/kassandra.git
cd kassandra

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python kassandra.py sprint_data.md

# Show help
python kassandra.py --help
```

## Example

Given a markdown file `sprint_data.md` with this table:

```markdown
| Sprint Name       | Stories | Weeks |
| :---------------- | :-----: | ----: |
| Sprint 1          |   21    |     3 |
| Sprint 2          |   21    |     3 |
| Sprint 3          |   25    |     3 |
| Sprint 4          |   23    |     3 |
```

Run the analysis:

```bash
python kassandra.py sprint_data.md
```

Output:

```
Team delivery capacity 95% confidence:
  Lower limit: 7.00 stories/week
  Upper limit: 8.30 stories/week
```

## Algorithm

The application implements the algorithm from SPECIFICATION.md:

1. **Data Filtering**: Uses only the 10 most recent sprints
2. **Ratio Calculation**: Calculates Stories/Weeks for each sprint
3. **Statistics**: Computes mean (μ) and sample standard deviation (σ)
4. **T-score Calculation**: Uses `stats.t.ppf(0.975, df)` for 95% confidence
5. **Confidence Interval**: Returns `μ ± t_score × (σ/√n)`

## Error Handling

The application handles various edge cases:
- **Insufficient data**: Requires at least 2 sprints with valid data
- **Zero variation**: Returns identical lower and upper bounds when all values are identical
- **Division by zero**: Skips sprints with zero weeks
- **Invalid data**: Skips rows with non-numeric Stories or Weeks values
- **Missing columns**: Validates that required columns (Stories, Weeks) exist

## Dependencies

- `click` - Command-line interface library
- `beautifulsoup4` - HTML parser for extracting table data
- `mistune` - Markdown parser with table support
- `numpy` - Numerical computing library
- `scipy` - Scientific computing and statistical functions

## Development

### Running Tests

Create test files with markdown tables and run:

```bash
python kassandra.py your_test_file.md
```

### Adding Features

The codebase is modular and easy to extend:
- Add new output formats in the main function
- Extend parsing logic in `read_and_parse_sprint_data()`
- Add more CLI options using Click decorators

## License

[MIT License](LICENSE) - Feel free to use, modify, and distribute.

## Contributing

Contributions are welcome! Please open issues or pull requests for:
- Bug fixes
- Feature requests
- Documentation improvements

## Author

Created with ❤️ by Mats Wiklander
