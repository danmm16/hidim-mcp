"""hidim-mcp: an MCP server that lets Claude work with high-dimensional data
without pulling the data itself into the conversation.

The data stays in this process. Tools return statistics, low-dimensional
projections, neighborhoods, tensor slices and rendered pictures, all capped in size.
"""

__version__ = "0.1.1"
