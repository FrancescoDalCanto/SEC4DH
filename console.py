"""Console formatting helpers for experiment output."""

LOG_WIDTH = 72


def print_section(title):
    """
    Print a section header with fixed-width separators.

    Parameters
    ----------
    title : str
        Text to display between separator lines.
    """
    print("\n" + "=" * LOG_WIDTH)
    print(title)
    print("=" * LOG_WIDTH)


def print_message(scope, message):
    """
    Print a single-line status message.

    Parameters
    ----------
    scope : str
        Short stage label displayed in brackets.
    message : str
        Message describing the current action or result.
    """
    print(f"[  {scope}  ] {message}")


def print_metric(label, value):
    """
    Print an aligned metric or configuration value.

    Parameters
    ----------
    label : str
        Name of the metric or setting.
    value : object
        Value to display next to ``label``.
    """
    print(f"  {label:<30}: {value}")
