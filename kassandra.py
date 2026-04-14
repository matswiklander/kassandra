#!/usr/bin/env python3
"""
Kassandra - Team Delivery Capacity Analyzer

Calculates 95% confidence intervals for team delivery capacity based on
historical sprint data from markdown tables. Implements the algorithm
specified in SPECIFICATION.md using Student's t-distribution for small samples.

Features:
- Parses markdown tables containing sprint data (Sprint Name, Stories, Weeks)
- Calculates Stories/Weeks ratios for delivery capacity
- Computes 95% confidence intervals using t-distribution
- Handles edge cases (insufficient data, zero variation, etc.)
- Provides user-friendly output with clear error messages
"""


import click
from typing import List, Tuple, Optional, Dict
from bs4 import BeautifulSoup
import mistune
from mistune.plugins.table import table
from scipy import stats
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_and_parse_sprint_data(file_path: str) -> List[List[str]]:
    """
    Read markdown file and parse sprint data in one operation.

    Combines file reading, markdown parsing, and basic validation into
    a single cohesive function for better error handling and simplicity.

    Args:
        file_path: Path to markdown file

    Returns:
        Parsed table data as list of rows

    Raises:
        Exception: If file cannot be read or table cannot be parsed
    """
    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()

        # Parse markdown table
        markdown_parser = mistune.create_markdown(plugins=[table])
        html_content = markdown_parser(markdown_content)

        # Parse HTML to extract table
        soup = BeautifulSoup(html_content, "html.parser")
        table_html = soup.find("table")

        if not table_html:
            raise Exception("No table found in the markdown file.")

        # Extract table data
        table_data = []
        for row in table_html.find_all("tr"):
            columns = []
            for cell in row.find_all(["th", "td"]):
                columns.append(cell.get_text(strip=True))
            table_data.append(columns)

        # Validate basic structure
        if not table_data or len(table_data) <= 1:
            raise Exception(
                "Table must contain at least one header row and one data row."
            )

        # Validate required columns
        header = table_data[0]
        try:
            header.index("Stories")
            header.index("Weeks")
        except ValueError as e:
            raise Exception(
                f"Table must contain 'Stories' and 'Weeks' columns: {str(e)}"
            )

        return table_data

    except Exception as e:
        raise Exception(f"Failed to read and parse sprint data: {str(e)}")


def validate_table_structure(table_data: List[List[str]]) -> Dict[str, int]:
    """
    Validate table structure and return column indices.

    Args:
        table_data: Parsed table data

    Returns:
        Dictionary with column indices for 'Stories' and 'Weeks'

    Raises:
        Exception: If table structure is invalid
    """
    if not table_data or len(table_data) <= 1:
        raise Exception("Table must contain at least one header row and one data row.")

    header = table_data[0]

    try:
        stories_idx = header.index("Stories")
        weeks_idx = header.index("Weeks")
        return {"stories_idx": stories_idx, "weeks_idx": weeks_idx}
    except ValueError as e:
        raise Exception(f"Table must contain 'Stories' and 'Weeks' columns: {str(e)}")


def generate_plot(lower_limit: float, upper_limit: float, output_file: str = "capacity_diagram.png") -> str:
    """
    Generate a diagram showing the confidence interval as a band over 0-52 weeks.
    
    Args:
        lower_limit: Lower bound of confidence interval (stories/week)
        upper_limit: Upper bound of confidence interval (stories/week)
        output_file: Path to save the generated plot
        
    Returns:
        Path to the generated plot file
    """
    weeks = np.arange(0, 53)
    cumulative_lower = weeks * lower_limit
    cumulative_upper = weeks * upper_limit
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Fill the confidence band
    ax.fill_between(weeks, cumulative_lower, cumulative_upper, alpha=0.3, color='blue', label=f'95% Confidence Band ({lower_limit:.2f}-{upper_limit:.2f} stories/week)')
    
    # Plot the bounds
    ax.plot(weeks, cumulative_lower, color='blue', linestyle='--', linewidth=1, label='Lower bound')
    ax.plot(weeks, cumulative_upper, color='blue', linestyle='--', linewidth=1, label='Upper bound')
    
    # Plot the mean line
    mean_line = weeks * ((lower_limit + upper_limit) / 2)
    ax.plot(weeks, mean_line, color='blue', linewidth=2, label='Expected capacity')
    
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Stories')
    ax.set_title('Team Delivery Capacity 95% Confidence Interval')
    ax.set_xticks(np.arange(0, 53, 4))
    ax.set_yticks(np.arange(0, int(cumulative_upper[-1]) + 1, max(1, int(cumulative_upper[-1]) // 8)))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    return output_file


def calculate_confidence_interval(
    sprint_data: List[List[str]],
) -> Optional[Tuple[float, float]]:
    """
    Calculate a 95% confidence interval for team delivery capacity.

    Args:
        sprint_data: List of rows from parsed table (including header)

    Returns:
        tuple: (lower_bound, upper_bound) for the confidence interval, or None if insufficient data

    Raises:
        Exception: If data validation fails
    """
    try:
        # Validate table structure
        column_indices = validate_table_structure(sprint_data)

        # Extract all sprint data (skip header)
        all_sprints = []
        for row in sprint_data[1:]:
            if len(row) > max(
                column_indices["stories_idx"], column_indices["weeks_idx"]
            ):
                try:
                    stories = int(row[column_indices["stories_idx"]])
                    weeks = int(row[column_indices["weeks_idx"]])
                    all_sprints.append({"Stories": stories, "Weeks": weeks})
                except (ValueError, TypeError) as e:
                    # Skip invalid rows silently
                    continue

        if not all_sprints:
            raise Exception("No valid sprint data found in table.")

        # Use only the 10 most recent sprints
        recent_sprints = all_sprints[-10:]

        # Calculate Stories/Weeks ratios
        stories_per_week = []
        for sprint in recent_sprints:
            if sprint["Weeks"] != 0:  # Avoid division by zero
                stories_per_week.append(sprint["Stories"] / sprint["Weeks"])

        # Need at least 2 data points for meaningful confidence interval
        if len(stories_per_week) < 2:
            raise Exception(
                "At least 2 sprints with valid Stories and Weeks data are required."
            )

        # Calculate statistics
        mean_value = np.mean(stories_per_week)
        standard_deviation = np.std(stories_per_week, ddof=1)
        n = len(stories_per_week)

        # Calculate confidence interval with T-score
        df = n - 1  # Degrees of freedom

        # Handle edge case: if all values are identical
        if standard_deviation == 0:
            return (float(mean_value), float(mean_value))

        t_score = stats.t.ppf(0.975, df)
        margin_of_error = t_score * (standard_deviation / np.sqrt(n))

        lower_bound = mean_value - margin_of_error
        upper_bound = mean_value + margin_of_error

        return (float(lower_bound), float(upper_bound))

    except Exception as e:
        raise Exception(f"Statistical calculation failed: {str(e)}")


@click.command()
@click.argument("markdown_file", type=click.Path(exists=True))
@click.option("--plot/--no-plot", default=False, help="Generate a diagram showing confidence interval over 0-52 weeks")
@click.option("--output", "-o", default="capacity_diagram.png", help="Output filename for the plot (default: capacity_diagram.png)")
def main(markdown_file: str, plot: bool, output: str):
    """
    Main function to analyze team delivery capacity from markdown file.

    Args:
        markdown_file: Path to markdown file containing sprint data
        plot: Whether to generate a diagram
        output: Output filename for the plot
    """

    try:
        # Read, parse, and validate markdown file in one step
        table_data = read_and_parse_sprint_data(markdown_file)

        # Calculate confidence interval
        confidence_interval = calculate_confidence_interval(table_data)

        if confidence_interval:
            lower_limit, upper_limit = confidence_interval
            click.echo(f"Team delivery capacity 95% confidence:")
            click.echo(f"  Lower limit: {lower_limit:.2f} stories/week")
            click.echo(f"  Upper limit: {upper_limit:.2f} stories/week")
            
            if plot:
                plot_file = generate_plot(lower_limit, upper_limit, output)
                click.echo(f"\nDiagram saved to: {plot_file}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


if __name__ == "__main__":
    main()
