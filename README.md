# Tsunami Wave Simulator

A Python application for simulating and visualizing tsunami wave motion in real-time.

## Features

- Interactive GUI for parameter adjustment
- Real-time wave visualization
- Physical parameters:
  - Wave amplitude
  - Wavelength
  - Water depth
- Scientific calculations based on linear wave theory

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python src/main.py
```

## Project Structure

```
tsunami_simulator/
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
└── src/
    ├── main.py         # Main application entry point
    └── models/         # Scientific models and calculations
```

## Development

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions and classes
- Run tests before committing changes

## License

MIT License
