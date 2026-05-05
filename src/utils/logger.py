from rich.console import Console

console = Console()


def log_info(message: str) -> None:
    """LOG INFO MESSAGE. **"""
    console.print(f"[bold cyan][INFO][/bold cyan] {message}")


def log_success(message: str) -> None:
    """LOG SUCCESS MESSAGE. **"""
    console.print(f"[bold green][SUCCESS][/bold green] {message}")


def log_warning(message: str) -> None:
    """LOG WARNING MESSAGE. **"""
    console.print(f"[bold yellow][WARNING][/bold yellow] {message}")


def log_error(message: str) -> None:
    """LOG ERROR MESSAGE. **"""
    console.print(f"[bold red][ERROR][/bold red] {message}")