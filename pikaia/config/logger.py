import logging
import sys

# Create a logger
logger = logging.getLogger("pikaia")
logger.setLevel(logging.INFO)

# Create a handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)
