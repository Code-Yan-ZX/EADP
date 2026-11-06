The first direct CPU-only admission check found the reviewer script at the
wrong guessed path (`review_author_graph_scatter_result.py`) and failed with
FileNotFoundError before any GPU use. The actual supervised reviewer command
records `review_author_graph_scatter.py`; the gate was corrected accordingly.
No completed or failed GPU experiment was produced by that check.
